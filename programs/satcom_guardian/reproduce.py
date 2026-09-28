"""Epitaxial-laser / T-chip optical SATCOM Phase-0 engineering model.

The modeled T device is an out-of-path link guardian.  It observes a pilot or
small receive-power tap while the payload continues through an ordinary modem.
The model is deliberately reduced order: it resolves decision windows and
modal powers rather than simulating every symbol of a multi-year link.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import numpy.linalg as la
from scipy.special import erf


HERE = Path(__file__).resolve().parent
FIGURE_PATH = HERE / "figures" / "satcom_guardian.png"
RESULTS_PATH = HERE / "results" / "satcom_guardian.json"

SPEED_OF_LIGHT_M_S = 299_792_458.0
PLANCK_J_S = 6.626_070_15e-34
RANGES_KM = np.asarray([500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0])
LINEWIDTHS_HZ = np.asarray([1e4, 1e5, 1e6, 1e7, 1e8, 5e9])
COHERENCE_DELAYS_PS = np.asarray([0.1, 1.0, 10.0, 100.0, 1_000.0])
DETUNING_DELAYS_PS = np.asarray([1.0, 10.0, 100.0])
RIN_LEVELS_DBC_HZ = np.asarray([-170.0, -155.0, -140.0, -125.0, -110.0, -100.0])
FAULT_BIASES_URAD = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
QPD_SPOT_RADII_URAD = np.asarray([6.75, 13.5, 27.0])
GUARDIAN_LOSS_SWEEP_DB = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])

RNG_STREAM_IDS = {
    "transmitter_latent": 11,
    "receiver_latent": 13,
    "common_source": 17,
    "scalar_detector": 29,
    "mode_sorter_detector": 31,
    "quadrant_detector": 37,
    "directional_t_detector": 41,
    "directional_t_phase_x": 43,
    "directional_t_phase_y": 47,
}


@dataclass(frozen=True)
class GuardianConfig:
    """Explicit exploratory assumptions, not specifications of a flight part."""

    wavelength_nm: float = 1550.0
    epitaxial_seed_monitored_carrier_power_w: float = 0.155
    boosted_monitored_carrier_power_w: float = 2.5
    beam_divergence_urad: float = 15.0
    receive_aperture_diameter_m: float = 0.10
    non_pointing_link_efficiency: float = 0.25
    guardian_tap_fraction: float = 0.01
    guardian_insertion_loss_db: float = 1.5
    directional_t_x_power_fraction: float = 0.5
    directional_t_complement_phase_rad: float = 0.0
    detector_quantum_efficiency: float = 0.80
    decision_window_ns: float = 100.0
    payload_lane_rate_gbps: float = 100.0
    illustrative_wdm_lanes: int = 10
    laser_linewidth_hz: float = 1.0e6
    laser_rin_dbc_per_hz: float = -150.0
    t_arm_delay_mismatch_ps: float = 1.0
    laser_frequency_drift_from_calibrated_carrier_ghz: float = 0.0
    residual_phase_jitter_rms_rad: float = 0.01
    detector_read_noise_e_rms: float = 5.0
    detector_total_two_port_dark_count_rate_hz: float = 1.0e6
    detector_gain_mismatch_fraction: float = 0.005
    transmitter_pointing_jitter_urad: float = 0.25
    receiver_aoa_jitter_urad: float = 0.25
    receiver_fault_bias_urad: float = 1.50
    qpd_equivalent_spot_radius_urad: float = 13.5
    polarization_jitter_deg: float = 1.0
    monte_carlo_trials_per_class: int = 30_000
    empirical_false_alarm_probability: float = 0.01
    seed: int = 20_260_909


def _named_rng(random_seed: int, stream_name: str) -> np.random.Generator:
    """Create an order-independent named PCG64 substream."""
    return np.random.default_rng(
        np.random.SeedSequence([random_seed, RNG_STREAM_IDS[stream_name]])
    )


def _laser_visibility(linewidth_hz: float, delay_s: float) -> float:
    """First-order visibility for a Lorentzian laser spectrum."""
    return float(np.exp(-np.pi * linewidth_hz * abs(delay_s)))


def _small_aperture_capture(
    config: GuardianConfig,
    range_km: float,
    pointing_rad: np.ndarray | float,
) -> np.ndarray:
    """Approximate Gaussian-beam capture including gross pointing loss."""
    range_m = range_km * 1e3
    divergence_rad = config.beam_divergence_urad * 1e-6
    beam_radius_m = divergence_rad * range_m
    diameter_squared = config.receive_aperture_diameter_m**2
    on_axis_capture = -np.expm1(-diameter_squared / (2.0 * beam_radius_m**2))
    pointing_factor = np.exp(-2.0 * (np.asarray(pointing_rad) / divergence_rad) ** 2)
    return on_axis_capture * pointing_factor


def _guardian_photoelectrons(
    config: GuardianConfig,
    transmit_power_w: float,
    range_km: float,
    pointing_rad: np.ndarray | float,
) -> np.ndarray:
    capture = _small_aperture_capture(config, range_km, pointing_rad)
    received_power_w = (
        transmit_power_w * config.non_pointing_link_efficiency * capture
    )
    guardian_power_w = (
        received_power_w
        * config.guardian_tap_fraction
        * 10.0 ** (-config.guardian_insertion_loss_db / 10.0)
    )
    window_s = config.decision_window_ns * 1e-9
    photon_energy_j = (
        PLANCK_J_S * SPEED_OF_LIGHT_M_S / (config.wavelength_nm * 1e-9)
    )
    return (
        guardian_power_w
        * window_s
        * config.detector_quantum_efficiency
        / photon_energy_j
    )


def _uniform_pupil_jinc(z: np.ndarray | float) -> np.ndarray:
    """Return 2 J1(z)/z by a stable dependency-free recurrence."""
    values = np.asarray(z, dtype=float)
    y = 0.25 * values**2
    term = np.ones_like(values)
    total = np.ones_like(values)
    for order in range(1, 64):
        term *= -y / (order * (order + 1.0))
        total += term
        if np.all(np.abs(term) <= 2e-16 * np.maximum(1.0, np.abs(total))):
            break
    return total


def _uniform_pupil_first_order_amplitude(
    z: np.ndarray | float,
) -> np.ndarray:
    """Return 4 J2(z)/z for a normalized circular-pupil tangent mode."""
    values = np.asarray(z, dtype=float)
    y = 0.25 * values**2
    term = 0.5 * values
    total = term.copy()
    for order in range(1, 64):
        term *= -y / (order * (order + 2.0))
        total += term
        if np.all(np.abs(term) <= 2e-16 * np.maximum(1.0, np.abs(total))):
            break
    return total


def _nominal_mode_power(
    config: GuardianConfig,
    point_x_urad: np.ndarray,
    point_y_urad: np.ndarray,
    polarization_angle_rad: np.ndarray,
) -> np.ndarray:
    """Exact uniform-circular-pupil piston-mode power."""
    radial_urad = np.hypot(point_x_urad, point_y_urad)
    z = (
        np.pi
        * config.receive_aperture_diameter_m
        * radial_urad
        * 1e-6
        / (config.wavelength_nm * 1e-9)
    )
    spatial_match = _uniform_pupil_jinc(z) ** 2
    polarization_match = np.cos(polarization_angle_rad) ** 2
    return np.clip(spatial_match * polarization_match, 0.0, 1.0)


def _directional_tangent_scores(
    config: GuardianConfig,
    point_x_urad: np.ndarray,
    point_y_urad: np.ndarray,
    polarization_angle_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed piston/tangent scores with retained residual-modal evidence."""
    radial_urad = np.hypot(point_x_urad, point_y_urad)
    z = (
        np.pi
        * config.receive_aperture_diameter_m
        * radial_urad
        * 1e-6
        / (config.wavelength_nm * 1e-9)
    )
    polarization_amplitude = np.cos(polarization_angle_rad)
    piston = _uniform_pupil_jinc(z) * polarization_amplitude
    tangent_radial = (
        _uniform_pupil_first_order_amplitude(z) * polarization_amplitude
    )
    direction_x = np.divide(
        point_x_urad,
        radial_urad,
        out=np.zeros_like(point_x_urad),
        where=radial_urad > 0.0,
    )
    direction_y = np.divide(
        point_y_urad,
        radial_urad,
        out=np.zeros_like(point_y_urad),
        where=radial_urad > 0.0,
    )
    tangent_x = tangent_radial * direction_x
    tangent_y = tangent_radial * direction_y
    complement_weight = np.cos(config.directional_t_complement_phase_rad)
    complement_x = np.clip(1.0 - piston**2 - tangent_x**2, 0.0, 1.0)
    complement_y = np.clip(1.0 - piston**2 - tangent_y**2, 0.0, 1.0)
    return (
        np.clip(
            2.0 * piston * tangent_x + complement_weight * complement_x,
            -1.0,
            1.0,
        ),
        np.clip(
            2.0 * piston * tangent_y + complement_weight * complement_y,
            -1.0,
            1.0,
        ),
    )


def _rin_common_power(
    config: GuardianConfig,
    rng: np.random.Generator,
    size: int,
    rin_dbc_per_hz: float | None = None,
) -> tuple[np.ndarray, float]:
    """Draw common power from one-sided white RIN and a rectangular window."""
    rin_db = (
        config.laser_rin_dbc_per_hz
        if rin_dbc_per_hz is None
        else rin_dbc_per_hz
    )
    equivalent_noise_bandwidth_hz = 1.0 / (
        2.0 * config.decision_window_ns * 1e-9
    )
    relative_variance = 10.0 ** (rin_db / 10.0) * equivalent_noise_bandwidth_hz
    log_sigma = float(np.sqrt(np.log1p(relative_variance)))
    factors = np.exp(rng.normal(-0.5 * log_sigma**2, log_sigma, size=size))
    return factors, float(np.sqrt(relative_variance))


def _simulate_population(
    config: GuardianConfig,
    *,
    range_km: float,
    transmit_power_w: float,
    transmitter_bias_urad: float,
    receiver_bias_urad: float,
    random_seed: int,
) -> dict[str, np.ndarray]:
    trials = config.monte_carlo_trials_per_class
    tx_rng = _named_rng(random_seed, "transmitter_latent")
    rx_rng = _named_rng(random_seed, "receiver_latent")
    source_rng = _named_rng(random_seed, "common_source")
    scalar_rng = _named_rng(random_seed, "scalar_detector")
    sorter_rng = _named_rng(random_seed, "mode_sorter_detector")
    qpd_rng = _named_rng(random_seed, "quadrant_detector")
    directional_rng = _named_rng(random_seed, "directional_t_detector")
    directional_phase_x_rng = _named_rng(random_seed, "directional_t_phase_x")
    directional_phase_y_rng = _named_rng(random_seed, "directional_t_phase_y")

    tx_x = tx_rng.normal(
        transmitter_bias_urad,
        config.transmitter_pointing_jitter_urad,
        trials,
    )
    tx_y = tx_rng.normal(0.0, config.transmitter_pointing_jitter_urad, trials)
    rx_x = rx_rng.normal(
        receiver_bias_urad,
        config.receiver_aoa_jitter_urad,
        trials,
    )
    rx_y = rx_rng.normal(0.0, config.receiver_aoa_jitter_urad, trials)
    polarization = rx_rng.normal(
        0.0, np.deg2rad(config.polarization_jitter_deg), trials
    )
    tx_radial_urad = np.hypot(tx_x, tx_y)
    expected_mode_power = _nominal_mode_power(
        config, rx_x, rx_y, polarization
    )
    ideal_directional_x, ideal_directional_y = _directional_tangent_scores(
        config, rx_x, rx_y, polarization
    )

    mean_photoelectrons = _guardian_photoelectrons(
        config,
        transmit_power_w,
        range_km,
        tx_radial_urad * 1e-6,
    )
    rin_factor, _ = _rin_common_power(config, source_rng, trials)
    delay_s = config.t_arm_delay_mismatch_ps * 1e-12
    visibility = _laser_visibility(config.laser_linewidth_hz, delay_s)
    static_phase = (
        2.0
        * np.pi
        * config.laser_frequency_drift_from_calibrated_carrier_ghz
        * 1e9
        * delay_s
    )
    dark_per_port = (
        0.5
        * config.detector_total_two_port_dark_count_rate_hz
        * config.decision_window_ns
        * 1e-9
    )
    total_signal_mean = mean_photoelectrons * rin_factor
    mismatch = config.detector_gain_mismatch_fraction
    plus_gain = 1.0 + 0.5 * mismatch
    minus_gain = 1.0 - 0.5 * mismatch

    directional_phase_x = static_phase + directional_phase_x_rng.normal(
        0.0, config.residual_phase_jitter_rms_rad, trials
    )
    directional_phase_y = static_phase + directional_phase_y_rng.normal(
        0.0, config.residual_phase_jitter_rms_rad, trials
    )
    observed_directional_x = (
        visibility * np.cos(directional_phase_x) * ideal_directional_x
    )
    observed_directional_y = (
        visibility * np.cos(directional_phase_y) * ideal_directional_y
    )
    directional_scores: list[np.ndarray] = []
    for cell_fraction, cell_score in (
        (config.directional_t_x_power_fraction, observed_directional_x),
        (1.0 - config.directional_t_x_power_fraction, observed_directional_y),
    ):
        cell_signal_mean = total_signal_mean * cell_fraction
        cell_plus = directional_rng.poisson(
            cell_signal_mean * 0.5 * (1.0 + cell_score) + dark_per_port
        ).astype(float)
        cell_minus = directional_rng.poisson(
            cell_signal_mean * 0.5 * (1.0 - cell_score) + dark_per_port
        ).astype(float)
        cell_plus += directional_rng.normal(
            0.0, config.detector_read_noise_e_rms, trials
        )
        cell_minus += directional_rng.normal(
            0.0, config.detector_read_noise_e_rms, trials
        )
        cell_plus = plus_gain * (cell_plus - dark_per_port)
        cell_minus = minus_gain * (cell_minus - dark_per_port)
        directional_scores.append(
            (cell_plus - cell_minus)
            / np.maximum(cell_plus + cell_minus, 1.0)
        )
    directional_x = directional_scores[0]
    directional_y = directional_scores[1]
    directional_radial = np.hypot(directional_x, directional_y)

    # A loss-matched direct mode sorter measures the same nominal-versus-residual
    # observable without a phase-sensitive T interferometer.  Matching its loss,
    # detector count, and readout noise isolates the T core's coherence burden.
    sorter_nominal = sorter_rng.poisson(
        total_signal_mean * expected_mode_power + dark_per_port
    ).astype(float)
    sorter_residual = sorter_rng.poisson(
        total_signal_mean * (1.0 - expected_mode_power) + dark_per_port
    ).astype(float)
    sorter_nominal += sorter_rng.normal(
        0.0, config.detector_read_noise_e_rms, trials
    )
    sorter_residual += sorter_rng.normal(
        0.0, config.detector_read_noise_e_rms, trials
    )
    sorter_nominal = plus_gain * (sorter_nominal - dark_per_port)
    sorter_residual = minus_gain * (sorter_residual - dark_per_port)
    sorter_total = sorter_nominal + sorter_residual
    normalized_sorter_score = (sorter_nominal - sorter_residual) / np.maximum(
        sorter_total, 1.0
    )

    boresight_guardian_photoelectrons = float(
        _guardian_photoelectrons(config, transmit_power_w, range_km, 0.0)
    )
    core_transmission = 10.0 ** (-config.guardian_insertion_loss_db / 10.0)
    pre_core_signal_mean = mean_photoelectrons * rin_factor / core_transmission
    scalar = scalar_rng.poisson(pre_core_signal_mean + dark_per_port).astype(float)
    scalar += scalar_rng.normal(0.0, config.detector_read_noise_e_rms, trials)
    scalar -= dark_per_port
    scalar_boresight_photoelectrons = (
        boresight_guardian_photoelectrons / core_transmission
    )
    normalized_scalar_power = scalar / max(scalar_boresight_photoelectrons, 1.0)

    # Practical pointing baseline: an ideal gapless four-quadrant detector on
    # the same pre-core tap.  For a Gaussian spot with 1/e^2 radius w, the
    # difference-over-sum response on either axis is erf(sqrt(2) * offset / w).
    # The angular spot radius is explicit because a real focal length, PSF,
    # detector gap, and deliberate defocus must ultimately replace it.
    qpd_scale = config.qpd_equivalent_spot_radius_urad
    true_qpd_x = erf(np.sqrt(2.0) * rx_x / qpd_scale)
    true_qpd_y = erf(np.sqrt(2.0) * rx_y / qpd_scale)
    right_fraction = 0.5 * (1.0 + true_qpd_x)
    upper_fraction = 0.5 * (1.0 + true_qpd_y)
    quadrant_fractions = np.column_stack(
        (
            right_fraction * upper_fraction,
            (1.0 - right_fraction) * upper_fraction,
            (1.0 - right_fraction) * (1.0 - upper_fraction),
            right_fraction * (1.0 - upper_fraction),
        )
    )
    qpd = qpd_rng.poisson(
        pre_core_signal_mean[:, None] * quadrant_fractions + dark_per_port
    ).astype(float)
    qpd += qpd_rng.normal(
        0.0, config.detector_read_noise_e_rms, size=(trials, 4)
    )
    # Use the same per-segment mismatch magnitude as the two-port detector.
    # The static right/left imbalance is retained so calibration must absorb it.
    qpd_gains = np.asarray(
        [plus_gain, minus_gain, minus_gain, plus_gain], dtype=float
    )
    qpd = (qpd - dark_per_port) * qpd_gains[None, :]
    qpd_total = np.sum(qpd, axis=1)
    qpd_denominator = np.maximum(qpd_total, 1.0)
    qpd_x = (qpd[:, 0] + qpd[:, 3] - qpd[:, 1] - qpd[:, 2]) / qpd_denominator
    qpd_y = (qpd[:, 0] + qpd[:, 1] - qpd[:, 2] - qpd[:, 3]) / qpd_denominator
    qpd_radial = np.sqrt(qpd_x**2 + qpd_y**2)
    return {
        "directional_t_radial_discriminant": directional_radial,
        "directional_t_x_discriminant": directional_x,
        "directional_t_y_discriminant": directional_y,
        "normalized_loss_matched_mode_sorter_score": normalized_sorter_score,
        "normalized_scalar_power": normalized_scalar_power,
        "qpd_radial_discriminant": qpd_radial,
        "qpd_x_discriminant": qpd_x,
        "qpd_y_discriminant": qpd_y,
        "ideal_directional_t_radial_discriminant": np.hypot(
            ideal_directional_x, ideal_directional_y
        ),
        "ideal_qpd_radial_discriminant": np.sqrt(true_qpd_x**2 + true_qpd_y**2),
        "mean_photoelectrons": mean_photoelectrons,
    }


def _rank_auc(normal: np.ndarray, fault: np.ndarray) -> float:
    """Tie-correct AUC when larger values mean more fault-like."""
    sorted_normal = np.sort(normal)
    strictly_lower = np.searchsorted(sorted_normal, fault, side="left")
    lower_or_equal = np.searchsorted(sorted_normal, fault, side="right")
    pairwise_wins = strictly_lower + 0.5 * (lower_or_equal - strictly_lower)
    return float(np.mean(pairwise_wins) / len(normal))


def _operating_point(
    calibration_anomaly: np.ndarray,
    normal_anomaly: np.ndarray,
    fault_anomaly: np.ndarray,
    target_false_alarm: float,
) -> dict[str, float | list[float]]:
    threshold = float(
        np.quantile(
            calibration_anomaly,
            1.0 - target_false_alarm,
            method="higher",
        )
    )
    calibration_exceedance = float(np.mean(calibration_anomaly > threshold))
    false_alarm_count = int(np.count_nonzero(normal_anomaly > threshold))
    detection_count = int(np.count_nonzero(fault_anomaly > threshold))
    return {
        "threshold": threshold,
        "threshold_operator": "strictly_greater_than",
        "calibration_exceedance_probability": calibration_exceedance,
        "evaluation_false_alarm_probability": float(
            false_alarm_count / len(normal_anomaly)
        ),
        "evaluation_false_alarm_probability_95_percent_interval": (
            _wilson_interval(false_alarm_count, len(normal_anomaly))
        ),
        "fault_detection_probability": float(detection_count / len(fault_anomaly)),
        "fault_detection_probability_95_percent_interval": _wilson_interval(
            detection_count, len(fault_anomaly)
        ),
        "area_under_roc": _rank_auc(normal_anomaly, fault_anomaly),
    }


def _wilson_interval(successes: int, trials: int) -> list[float]:
    """Two-sided 95% Wilson score interval for a binomial proportion."""
    z = 1.959_963_984_540_054
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / trials
            + z**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return [float(center - radius), float(center + radius)]


def _positive_hermitian_square_root(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = la.eigh(matrix)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.conj().T


def _burau_unitary_generator(index: int, omega: float) -> np.ndarray:
    """Return one Euclidean-unitary modified Burau generator for B3."""
    s = np.exp(0.5j * omega)
    if index == 1:
        beta = np.asarray([[-s**2, s], [0.0, 1.0]], dtype=complex)
    elif index == 2:
        beta = np.asarray([[1.0, 0.0], [s, -s**2]], dtype=complex)
    else:
        raise ValueError("Burau generator index must be 1 or 2")
    form = np.asarray(
        [[2.0 * np.cos(omega / 2.0), -1.0], [-1.0, 2.0 * np.cos(omega / 2.0)]],
        dtype=complex,
    )
    root = _positive_hermitian_square_root(form)
    return root @ beta @ la.inv(root)


def _exact_burau_t_mixer_checks() -> dict[str, float]:
    """Verify the exact balanced three-letter Burau path mixer."""
    omega = float(np.pi / 4.0)
    unitary_one = _burau_unitary_generator(1, omega)
    unitary_two = _burau_unitary_generator(2, omega)
    mixer = la.inv(unitary_two) @ unitary_one @ la.inv(unitary_two)
    identity = np.eye(2, dtype=complex)
    balanced = np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    left_phases = np.empty(2, dtype=complex)
    right_phases = np.empty(2, dtype=complex)
    right_phases[0] = 1.0
    left_phases[0] = balanced[0, 0] / mixer[0, 0]
    right_phases[1] = balanced[0, 1] / (left_phases[0] * mixer[0, 1])
    left_phases[1] = balanced[1, 0] / mixer[1, 0]
    phase_aligned = np.diag(left_phases) @ mixer @ np.diag(right_phases)
    return {
        "burau_mixer_omega_rad": omega,
        "burau_mixer_unitarity_residual": float(
            la.norm(mixer.conj().T @ mixer - identity)
        ),
        "burau_mixer_balanced_magnitude_residual": float(
            np.max(np.abs(np.abs(mixer) - 1.0 / np.sqrt(2.0)))
        ),
        "burau_mixer_generic_coupler_equivalence_residual": float(
            la.norm(phase_aligned - balanced)
        ),
    }


def _exact_t_checks() -> dict[str, float]:
    rng = np.random.default_rng(1701)
    dimension = 4
    identity = np.eye(dimension, dtype=complex)
    checks = {
        "tie_correct_auc_self_test_error": float(
            abs(
                _rank_auc(
                    np.asarray([0.0, 0.0, 1.0]),
                    np.asarray([0.0, 0.0, 1.0]),
                )
                - 0.5
            )
        ),
    }
    q_x = identity.copy().astype(complex)
    q_x[0, 0] = 0.0
    q_x[1, 1] = 0.0
    q_x[0, 1] = -1j
    q_x[1, 0] = 1j
    q_y = identity.copy().astype(complex)
    q_y[0, 0] = 0.0
    q_y[2, 2] = 0.0
    q_y[0, 2] = -1j
    q_y[2, 0] = 1j
    tangent_states = np.zeros((256, dimension), dtype=complex)
    b_x = rng.uniform(-0.25, 0.25, len(tangent_states))
    b_y = rng.uniform(-0.25, 0.25, len(tangent_states))
    residual = rng.uniform(0.0, 0.10, len(tangent_states))
    a_0 = np.sqrt(1.0 - b_x**2 - b_y**2 - residual**2)
    tangent_states[:, 0] = a_0
    tangent_states[:, 1] = 1j * b_x
    tangent_states[:, 2] = 1j * b_y
    tangent_states[:, 3] = residual
    for axis, branch, expected in (
        ("x", q_x, 2.0 * a_0 * b_x + b_y**2 + residual**2),
        ("y", q_y, 2.0 * a_0 * b_y + b_x**2 + residual**2),
    ):
        branch_states = tangent_states @ branch.T
        directional_plus = 0.5 * (tangent_states + branch_states)
        directional_minus = 0.5 * (tangent_states - branch_states)
        directional_score = np.sum(np.abs(directional_plus) ** 2, axis=1) - np.sum(
            np.abs(directional_minus) ** 2, axis=1
        )
        checks[f"directional_{axis}_branch_unitarity_residual"] = float(
            la.norm(branch.conj().T @ branch - identity)
        )
        checks[f"directional_{axis}_hybrid_score_residual"] = float(
            np.max(np.abs(directional_score - expected))
        )
    checks.update(_exact_burau_t_mixer_checks())
    residuals = {
        name: value
        for name, value in checks.items()
        if name.endswith("residual") or name.endswith("error")
    }
    if max(residuals.values()) > 2e-12:
        raise AssertionError("ideal T-cell identities failed")
    return checks


def _rin_sensitivity(
    config: GuardianConfig, rng: np.random.Generator
) -> dict[str, list[float]]:
    trials = config.monte_carlo_trials_per_class
    mean_photoelectrons = 1.0e6
    true_score = 0.8
    plus_fraction = 0.5 * (1.0 + true_score)
    minus_fraction = 1.0 - plus_fraction
    raw_rmse = []
    normalized_rmse = []
    fractional_rin_rms = []
    for rin_db in RIN_LEVELS_DBC_HZ:
        common, rin_rms = _rin_common_power(
            config, rng, trials, float(rin_db)
        )
        plus = rng.poisson(mean_photoelectrons * common * plus_fraction).astype(float)
        minus = rng.poisson(mean_photoelectrons * common * minus_fraction).astype(float)
        plus += rng.normal(0.0, config.detector_read_noise_e_rms, trials)
        minus += rng.normal(0.0, config.detector_read_noise_e_rms, trials)
        mismatch = config.detector_gain_mismatch_fraction
        plus *= 1.0 + 0.5 * mismatch
        minus *= 1.0 - 0.5 * mismatch
        raw_score = (plus - minus) / mean_photoelectrons
        normalized_score = (plus - minus) / np.maximum(plus + minus, 1.0)
        raw_rmse.append(float(np.sqrt(np.mean((raw_score - true_score) ** 2))))
        normalized_rmse.append(
            float(np.sqrt(np.mean((normalized_score - true_score) ** 2)))
        )
        fractional_rin_rms.append(rin_rms)
    return {
        "rin_dbc_per_hz": RIN_LEVELS_DBC_HZ.tolist(),
        "integrated_fractional_rin_rms": fractional_rin_rms,
        "raw_difference_score_rmse": raw_rmse,
        "normalized_difference_score_rmse": normalized_rmse,
        "test_photoelectrons_per_window": mean_photoelectrons,
        "true_normalized_score": true_score,
    }


def _population_anomalies(population: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return larger-is-more-anomalous statistics for every modeled receiver."""
    return {
        "burau_directional_t_guardian": population[
            "directional_t_radial_discriminant"
        ],
        "loss_matched_mode_sorter": -population[
            "normalized_loss_matched_mode_sorter_score"
        ],
        "quadrant_detector": population["qpd_radial_discriminant"],
        "scalar_power_monitor": -population["normalized_scalar_power"],
    }


def _paired_three_populations(
    config: GuardianConfig,
    *,
    range_km: float,
    transmit_power_w: float,
    base_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    calibration = _simulate_population(
        replace(
            config,
            laser_frequency_drift_from_calibrated_carrier_ghz=0.0,
        ),
        range_km=range_km,
        transmit_power_w=transmit_power_w,
        transmitter_bias_urad=0.0,
        receiver_bias_urad=0.0,
        random_seed=base_seed,
    )
    normal = _simulate_population(
        config,
        range_km=range_km,
        transmit_power_w=transmit_power_w,
        transmitter_bias_urad=0.0,
        receiver_bias_urad=0.0,
        random_seed=base_seed + 1,
    )
    fault = _simulate_population(
        config,
        range_km=range_km,
        transmit_power_w=transmit_power_w,
        transmitter_bias_urad=0.0,
        receiver_bias_urad=config.receiver_fault_bias_urad,
        random_seed=base_seed + 2,
    )
    return calibration, normal, fault


def _metrics_for_populations(
    config: GuardianConfig,
    calibration: dict[str, np.ndarray],
    normal: dict[str, np.ndarray],
    fault: dict[str, np.ndarray],
) -> dict[str, object]:
    calibration_anomalies = _population_anomalies(calibration)
    normal_anomalies = _population_anomalies(normal)
    fault_anomalies = _population_anomalies(fault)
    return {
        name: _operating_point(
            calibration_anomalies[name],
            normal_anomalies[name],
            fault_anomalies[name],
            config.empirical_false_alarm_probability,
        )
        for name in calibration_anomalies
    }


def _guardian_loss_sensitivity(config: GuardianConfig) -> dict[str, object]:
    """Paired counterfactual sweep of the unmeasured T-bank insertion loss."""
    rows = []
    paired_seed = config.seed + 10_400
    for loss_db in GUARDIAN_LOSS_SWEEP_DB:
        candidate = replace(config, guardian_insertion_loss_db=float(loss_db))
        calibration, normal, fault = _paired_three_populations(
            candidate,
            range_km=8_000.0,
            transmit_power_w=candidate.boosted_monitored_carrier_power_w,
            base_seed=paired_seed,
        )
        metrics = _metrics_for_populations(candidate, calibration, normal, fault)
        rows.append(
            {
                "guardian_insertion_loss_db": float(loss_db),
                "burau_directional_t_guardian": metrics[
                    "burau_directional_t_guardian"
                ],
                "pre_core_quadrant_detector": metrics["quadrant_detector"],
            }
        )
    return {
        "range_km": 8_000.0,
        "monitored_carrier_power_w": config.boosted_monitored_carrier_power_w,
        "receiver_aoa_bias_urad": config.receiver_fault_bias_urad,
        "paired_random_stream_base_seed": paired_seed,
        "rows": rows,
    }


def _directional_t_complement_phase_ablation(
    config: GuardianConfig,
) -> dict[str, object]:
    """Measure the value of retaining complementary residual-modal power."""
    rows = []
    paired_seed = config.seed + 10_400
    for label, phase_rad in (
        ("retained_residual_default", 0.0),
        ("pure_signed_quadrature_ablation", float(np.pi / 2.0)),
    ):
        candidate = replace(config, directional_t_complement_phase_rad=phase_rad)
        calibration, normal, fault = _paired_three_populations(
            candidate,
            range_km=8_000.0,
            transmit_power_w=candidate.boosted_monitored_carrier_power_w,
            base_seed=paired_seed,
        )
        metrics = _metrics_for_populations(candidate, calibration, normal, fault)
        rows.append(
            {
                "case": label,
                "complement_phase_rad": phase_rad,
                "burau_directional_t_guardian": metrics[
                    "burau_directional_t_guardian"
                ],
                "pre_core_quadrant_detector": metrics["quadrant_detector"],
            }
        )
    return {
        "paired_random_stream_base_seed": paired_seed,
        "interpretation": (
            "The default +1 complement preserves quadratic residual-modal "
            "evidence alongside the signed piston/tangent term. The pi/2 "
            "case suppresses that complement and is a pure-signed ablation."
        ),
        "rows": rows,
    }


def _directional_t_design_characterization(
    config: GuardianConfig,
) -> dict[str, object]:
    """Record local gain, information scale, and useful response range."""
    angles = np.linspace(0.0, 35.0, 35_001)
    zeros = np.zeros_like(angles)
    scores, _ = _directional_tangent_scores(config, angles, zeros, zeros)
    wavelength_m = config.wavelength_nm * 1e-9
    t_slope_per_urad = (
        np.pi * config.receive_aperture_diameter_m / wavelength_m * 1e-6
    )
    qpd_slope_per_urad = (
        2.0
        * np.sqrt(2.0)
        / (np.sqrt(np.pi) * config.qpd_equivalent_spot_radius_urad)
    )
    linear_reference = t_slope_per_urad * angles
    relative_error = np.zeros_like(angles)
    relative_error[1:] = np.abs(scores[1:] / linear_reference[1:] - 1.0)
    nonlinear_indices = np.flatnonzero(relative_error > 0.01)
    one_percent_linear_range = (
        float(angles[max(int(nonlinear_indices[0]) - 1, 0)])
        if len(nonlinear_indices)
        else float(angles[-1])
    )
    nonincreasing = np.flatnonzero(np.diff(scores) <= 0.0)
    monotonic_limit = (
        float(angles[nonincreasing[0]])
        if len(nonincreasing)
        else float(angles[-1])
    )
    transmission = 10.0 ** (-config.guardian_insertion_loss_db / 10.0)
    t_fisher_per_pre_core_photon = (
        transmission
        * config.directional_t_x_power_fraction
        * t_slope_per_urad**2
    )
    qpd_fisher_per_pre_core_photon = qpd_slope_per_urad**2
    fault_score = float(
        _directional_tangent_scores(
            config,
            np.asarray([config.receiver_fault_bias_urad]),
            np.asarray([0.0]),
            np.asarray([0.0]),
        )[0][0]
    )
    return {
        "architecture": (
            "two parallel hybrid tangent/residual T cells; x/y photon "
            "allocation sums to one and four photodiodes match the QPD "
            "channel count"
        ),
        "branch_map": (
            "sigma_y on the enrolled-piston/tangent pair and exp(i phi_c) "
            "on the orthogonal complement; default phi_c=0 retains "
            "residual-modal evidence"
        ),
        "directional_t_x_power_fraction": config.directional_t_x_power_fraction,
        "directional_t_y_power_fraction": (
            1.0 - config.directional_t_x_power_fraction
        ),
        "directional_t_complement_phase_rad": (
            config.directional_t_complement_phase_rad
        ),
        "boresight_signed_gain_per_urad": float(t_slope_per_urad),
        "qpd_boresight_signed_gain_per_urad": float(qpd_slope_per_urad),
        "directional_t_score_at_selected_x_fault": fault_score,
        "one_percent_small_angle_linearity_range_urad": one_percent_linear_range,
        "first_positive_axis_monotonic_limit_urad": monotonic_limit,
        "shot_noise_fisher_information_per_pre_core_photon_per_urad2": {
            "directional_t_x_axis_after_configured_loss_and_fanout": float(
                t_fisher_per_pre_core_photon
            ),
            "ideal_pre_core_qpd_x_axis": float(qpd_fisher_per_pre_core_photon),
            "directional_t_to_qpd_ratio": float(
                t_fisher_per_pre_core_photon / qpd_fisher_per_pre_core_photon
            ),
        },
        "warnings": [
            "local Fisher information excludes capture loss, aberration, detector gaps, and control-loop dynamics",
            "the retained complement adds calibrated quadratic cross-terms, so precision steering needs a two-dimensional lookup",
            "the tangent response becomes nonlinear and eventually nonmonotonic outside its local tracking range",
            "acquisition still requires the terminal PAT sensor or a wider-field mode",
        ],
    }


def _fault_severity_sweep(config: GuardianConfig) -> dict[str, object]:
    """Resolve how strongly the result depends on the assumed pointing fault."""
    range_km = float(RANGES_KM[-1])
    transmit_power_w = config.boosted_monitored_carrier_power_w
    seed = config.seed + 300_000
    calibration = _simulate_population(
        config,
        range_km=range_km,
        transmit_power_w=transmit_power_w,
        transmitter_bias_urad=0.0,
        receiver_bias_urad=0.0,
        random_seed=seed,
    )
    normal = _simulate_population(
        config,
        range_km=range_km,
        transmit_power_w=transmit_power_w,
        transmitter_bias_urad=0.0,
        receiver_bias_urad=0.0,
        random_seed=seed + 1,
    )
    calibration_anomalies = _population_anomalies(calibration)
    normal_anomalies = _population_anomalies(normal)
    rows = []
    for index, bias_urad in enumerate(FAULT_BIASES_URAD):
        fault = _simulate_population(
            config,
            range_km=range_km,
            transmit_power_w=transmit_power_w,
            transmitter_bias_urad=0.0,
            receiver_bias_urad=float(bias_urad),
            random_seed=seed + 100 + index,
        )
        fault_anomalies = _population_anomalies(fault)
        metrics = {
            name: _operating_point(
                calibration_anomalies[name],
                normal_anomalies[name],
                fault_anomalies[name],
                config.empirical_false_alarm_probability,
            )
            for name in calibration_anomalies
        }
        rows.append(
            {
                "pointing_bias_urad": float(bias_urad),
                "bias_over_nominal_jitter_sigma": float(
                    bias_urad / config.receiver_aoa_jitter_urad
                ),
                "bias_over_lambda_over_aperture": float(
                    bias_urad
                    / (
                        config.wavelength_nm
                        * 1e-3
                        / config.receive_aperture_diameter_m
                    )
                ),
                "mean_true_directional_t_radial_score": float(
                    np.mean(fault["ideal_directional_t_radial_discriminant"])
                ),
                "mean_total_power_reduction_fraction": float(
                    1.0
                    - np.mean(fault["mean_photoelectrons"])
                    / np.mean(normal["mean_photoelectrons"])
                ),
                **metrics,
            }
        )
    return {
        "range_km": range_km,
        "monitored_carrier_power_w": transmit_power_w,
        "interpretation": (
            "sensitivity study, not a fault-occurrence distribution; the zero-bias "
            "row is a negative control"
        ),
        "rows": rows,
    }


def _qpd_design_sensitivity(config: GuardianConfig) -> dict[str, object]:
    """Expose the sensitivity/FOV trade hidden by one assumed QPD spot size."""
    range_km = float(RANGES_KM[-1])
    rows = []
    for power_index, (power_name, transmit_power_w) in enumerate(
        (
            ("seed_only", config.epitaxial_seed_monitored_carrier_power_w),
            ("boosted", config.boosted_monitored_carrier_power_w),
        )
    ):
        common_seed = config.seed + 400_000 + power_index * 10_000
        for spot_radius_urad in QPD_SPOT_RADII_URAD:
            local_config = replace(
                config,
                qpd_equivalent_spot_radius_urad=float(spot_radius_urad),
            )
            calibration = _simulate_population(
                local_config,
                range_km=range_km,
                transmit_power_w=transmit_power_w,
                transmitter_bias_urad=0.0,
                receiver_bias_urad=0.0,
                random_seed=common_seed,
            )
            normal = _simulate_population(
                local_config,
                range_km=range_km,
                transmit_power_w=transmit_power_w,
                transmitter_bias_urad=0.0,
                receiver_bias_urad=0.0,
                random_seed=common_seed + 1,
            )
            fault = _simulate_population(
                local_config,
                range_km=range_km,
                transmit_power_w=transmit_power_w,
                transmitter_bias_urad=0.0,
                receiver_bias_urad=config.receiver_fault_bias_urad,
                random_seed=common_seed + 2,
            )
            metrics = _operating_point(
                calibration["qpd_radial_discriminant"],
                normal["qpd_radial_discriminant"],
                fault["qpd_radial_discriminant"],
                config.empirical_false_alarm_probability,
            )
            rows.append(
                {
                    "power_case": power_name,
                    "monitored_carrier_power_w": transmit_power_w,
                    "qpd_equivalent_spot_radius_urad": float(spot_radius_urad),
                    **metrics,
                }
            )
    return {
        "range_km": range_km,
        "rows": rows,
        "interpretation": (
            "smaller assumed spots have steeper difference/sum response but less "
            "physical field-of-view margin; the ideal gapless infinite detector in "
            "this model does not capture clipping, gaps, saturation, aberration, or "
            "the deliberate defocus trade used by real terminals"
        ),
    }


def _assessed_range_summary(
    detection: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Summarize the sampled range envelope without extrapolating a max range."""
    receiver_names = (
        "burau_directional_t_guardian",
        "loss_matched_mode_sorter",
        "quadrant_detector",
        "scalar_power_monitor",
    )
    targets = (0.5, 0.9, 0.99)
    by_power: dict[str, object] = {}
    for power_name, power_results in detection.items():
        rows = power_results["by_range"]
        receiver_summary: dict[str, object] = {}
        for receiver_name in receiver_names:
            receiver_summary[receiver_name] = {
                str(target): max(
                    (
                        float(row["range_km"])
                        for row in rows
                        if row[receiver_name]["fault_detection_probability"]
                        >= target
                    ),
                    default=None,
                )
                for target in targets
            }
        by_power[power_name] = receiver_summary
    return {
        "meaning": (
            "farthest sampled range meeting each detection-probability target for "
            "the selected 1.5-urad stress case at the diagnostic 1% false-alarm "
            "point; these are not maximum terminal ranges"
        ),
        "sampled_ranges_km": RANGES_KM.tolist(),
        "by_transmit_power": by_power,
    }


def _literature_range_context(config: GuardianConfig) -> dict[str, object]:
    """Relate the exploratory grid to explicit SDA OCT v4 range anchors."""
    core_transmission = 10.0 ** (-config.guardian_insertion_loss_db / 10.0)
    anchors = []
    for range_km, service in (
        (5_500.0, "continuous interoperable modes"),
        (20_000.0, "low-rate burst modes"),
    ):
        post_core = float(
            _guardian_photoelectrons(
                config,
                config.boosted_monitored_carrier_power_w,
                range_km,
                0.0,
            )
        )
        anchors.append(
            {
                "range_km": range_km,
                "sda_service_context": service,
                "one_way_light_time_ms": range_km * 1e3 / SPEED_OF_LIGHT_M_S * 1e3,
                "model_post_t_core_photoelectrons_per_100_ns_window": post_core,
                "model_pre_core_photoelectrons_per_100_ns_window": (
                    post_core / core_transmission
                ),
            }
        )
    return {
        "standard": "SDA Optical Communications Terminal Standard v4.0.0",
        "standard_url": (
            "https://www.sda.mil/wp-content/uploads/2024/07/"
            "SDA_OCT_Standard_4.0.0_final-20240701.pdf"
        ),
        "current_sda_resources_url": (
            "https://www.sda.mil/home/work-with-us/resources/"
        ),
        "deployment_context": (
            "SDA's public resources page identifies v3.2 as the standard of "
            "record for Tranche 3 and v4 for a smaller set of terminals; the "
            "20,000-km v4 mode is not a generic requirement for every link"
        ),
        "standard_facts": {
            "continuous_reference_range_km": 5_500.0,
            "continuous_receive_aperture_irradiance_uw_per_m2": 25.0,
            "burst_reference_range_km": 20_000.0,
            "burst_long_term_receive_aperture_irradiance_uw_per_m2": 6.0,
            "burst_duration_ns": 102.4,
            "minimum_maximum_transmit_power_w": 2.5,
        },
        "model_at_standard_ranges": anchors,
        "insertion_loss_range_scaling": {
            "t_core_transmission": core_transmission,
            "pre_core_to_post_core_photon_ratio": 1.0 / core_transmission,
            "t_range_fraction_at_equal_detected_photons_under_inverse_square_scaling": (
                np.sqrt(core_transmission)
            ),
            "interpretation": (
                "the assumed 1.5-dB T-core loss alone shortens equal-photon range "
                "to sqrt(transmission) of a lossless pre-core sensor; detector "
                "statistics and fault response can change the actual crossover"
            ),
        },
        "non_compliance_warning": (
            "The two model photon counts merely evaluate this repository's continuous "
            "Gaussian-envelope assumptions at the standard's distances. They do not "
            "model OCT waveforms, burst duty cycle, receiver sensitivity, PAT margin, "
            "or demonstrate SDA compliance."
        ),
    }


def _technology_trade_study(
    detection: dict[str, dict[str, object]],
    qpd_design_sensitivity: dict[str, object],
) -> dict[str, object]:
    """Record an explicit, literature-routed baseline decision matrix."""
    criteria = {
        "signed_pointing_control_output": (
            "Does the receiver directly provide signed azimuth/elevation error for "
            "a steering loop?"
        ),
        "calibration_and_control_ease": (
            "Higher means less active optical stabilization and simpler calibration."
        ),
        "production_readiness": (
            "Relative evidence for producible terminal hardware; this is not a formal TRL."
        ),
        "coherence_and_wavelength_robustness": (
            "Higher means less dependence on optical phase, linewidth, and arm delay."
        ),
        "photon_and_loss_efficiency": (
            "Higher means fewer added optical losses/readout channels for the stated role."
        ),
        "fault_coverage": (
            "Breadth across pointing, modal/polarization, power, and decoded-link faults."
        ),
        "incremental_swa_p": (
            "Higher means lower incremental size, weight, electrical power, and complexity."
        ),
        "evidence_maturity": (
            "Higher means flight or strong experimental evidence rather than simulation."
        ),
    }
    scores = {
        "burau_directional_t_guardian": {
            "signed_pointing_control_output": 5,
            "calibration_and_control_ease": 2,
            "production_readiness": 1,
            "coherence_and_wavelength_robustness": 2,
            "photon_and_loss_efficiency": 3,
            "fault_coverage": 3,
            "incremental_swa_p": 2,
            "evidence_maturity": 1,
        },
        "quadrant_detector_pat": {
            "signed_pointing_control_output": 5,
            "calibration_and_control_ease": 4,
            "production_readiness": 5,
            "coherence_and_wavelength_robustness": 5,
            "photon_and_loss_efficiency": 4,
            "fault_coverage": 2,
            "incremental_swa_p": 4,
            "evidence_maturity": 5,
        },
        "pixel_focal_plane_pat": {
            "signed_pointing_control_output": 5,
            "calibration_and_control_ease": 3,
            "production_readiness": 4,
            "coherence_and_wavelength_robustness": 5,
            "photon_and_loss_efficiency": 3,
            "fault_coverage": 3,
            "incremental_swa_p": 3,
            "evidence_maturity": 4,
        },
        "conventional_mode_sorter": {
            "signed_pointing_control_output": 1,
            "calibration_and_control_ease": 3,
            "production_readiness": 3,
            "coherence_and_wavelength_robustness": 4,
            "photon_and_loss_efficiency": 3,
            "fault_coverage": 4,
            "incremental_swa_p": 2,
            "evidence_maturity": 3,
        },
        "modem_fec_link_telemetry": {
            "signed_pointing_control_output": 1,
            "calibration_and_control_ease": 4,
            "production_readiness": 5,
            "coherence_and_wavelength_robustness": 4,
            "photon_and_loss_efficiency": 5,
            "fault_coverage": 5,
            "incremental_swa_p": 5,
            "evidence_maturity": 5,
        },
        "scalar_power_tap": {
            "signed_pointing_control_output": 1,
            "calibration_and_control_ease": 5,
            "production_readiness": 5,
            "coherence_and_wavelength_robustness": 5,
            "photon_and_loss_efficiency": 5,
            "fault_coverage": 1,
            "incremental_swa_p": 5,
            "evidence_maturity": 5,
        },
    }
    farthest_boosted = detection["boosted"]["by_range"][-1]
    simulated_receivers = {
        name: farthest_boosted[name]
        for name in (
            "burau_directional_t_guardian",
            "loss_matched_mode_sorter",
            "quadrant_detector",
            "scalar_power_monitor",
        )
    }
    seed_qpd_by_spot_radius = {
        str(row["qpd_equivalent_spot_radius_urad"]): row[
            "fault_detection_probability"
        ]
        for row in qpd_design_sensitivity["rows"]
        if row["power_case"] == "seed_only"
    }
    return {
        "score_scale": {
            "literature_review_as_of": "2026-09-28",
            "minimum": 1,
            "maximum": 5,
            "direction": "5 is more favorable",
            "warning": (
                "ordinal engineering judgement, not measured performance, formal TRL, "
                "or a weighted procurement score; do not sum columns"
            ),
        },
        "criteria": criteria,
        "scores": scores,
        "simulated_default_stress_at_farthest_assessed_range": {
            "range_km": float(farthest_boosted["range_km"]),
            "monitored_carrier_power_w": float(
                detection["boosted"]["monitored_carrier_power_w"]
            ),
            "receivers": simulated_receivers,
        },
        "seed_only_qpd_detection_by_spot_radius_urad": seed_qpd_by_spot_radius,
        "baseline_selection": {
            "primary_burau_candidate": "burau_directional_t_guardian",
            "primary_pointing_baseline": "quadrant_detector_pat",
            "mandatory_system_baseline": (
                "modem_fec_link_telemetry_plus_existing_pat"
            ),
            "matched_directional_control": "generic_two_cell_balanced_interferometer",
            "modal_baseline": "conventional_mode_sorter",
            "negative_control_only": "scalar_power_tap",
            "pixel_sensor_role": (
                "secondary baseline when acquisition field of view, multi-spot "
                "disambiguation, or non-Gaussian imagery matters"
            ),
        },
        "current_decision": {
            "burau_specific_advantage_demonstrated": False,
            "directional_t_architecture_passes_phase0_performance_gate": True,
            "reason": (
                "The optimized T bank now supplies signed x/y errors and matches "
                "the practical QPD at the selected 1.5-dB corner. A generic pair "
                "of balanced interferometers implements the same directional "
                "observable, so the simulation establishes a useful architecture "
                "but not a uniquely Burau hardware advantage. The conventional "
                "mode sorter remains a separate modal-sensing baseline."
            ),
            "reconsider_if": (
                "measured hardware shows a loss, bandwidth, stability, fault-coverage, "
                "or SWaP advantage over the quadrant detector and both generic "
                "matched implementations"
            ),
        },
    }


def _plot_results(
    config: GuardianConfig,
    link_budget: dict[str, object],
    detection: dict[str, dict[str, object]],
    rin: dict[str, object],
    severity_sweep: dict[str, object],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 8.0))

    axes[0, 0].loglog(
        RANGES_KM,
        link_budget["seed_only_post_t_core_photoelectrons_per_window"],
        marker="o",
        label=(
            f"{config.epitaxial_seed_monitored_carrier_power_w * 1e3:.0f} mW "
            "monitored carrier"
        ),
    )
    axes[0, 0].loglog(
        RANGES_KM,
        link_budget["boosted_post_t_core_photoelectrons_per_window"],
        marker="o",
        label=(
            f"{config.boosted_monitored_carrier_power_w:.1f} W monitored "
            "carrier"
        ),
    )
    axes[0, 0].set_xlabel("inter-satellite range (km)")
    axes[0, 0].set_ylabel("guardian photoelectrons / window")
    axes[0, 0].set_title("(a) Receive-tap photon budget")
    axes[0, 0].legend(fontsize=8)

    delay_curve_ps = np.logspace(-1, 3, 240)
    for linewidth in [1e5, 1e6, 1e7, 1e8, 5e9]:
        visibility = np.exp(-np.pi * linewidth * delay_curve_ps * 1e-12)
        axes[0, 1].semilogx(
            delay_curve_ps,
            visibility,
            label=f"{linewidth / 1e6:g} MHz",
        )
    axes[0, 1].axhline(0.99, color="black", linewidth=0.8, linestyle="--")
    axes[0, 1].set_ylim(-0.02, 1.03)
    axes[0, 1].set_xlabel("T-arm delay mismatch (ps)")
    axes[0, 1].set_ylabel("interference visibility")
    axes[0, 1].set_title("(b) Parameterized-source coherence")
    axes[0, 1].legend(fontsize=7, ncol=2)

    detuning_ghz = np.linspace(-5.0, 5.0, 501)
    for delay_ps in DETUNING_DELAYS_PS:
        gain = np.cos(2.0 * np.pi * detuning_ghz * 1e9 * delay_ps * 1e-12)
        axes[0, 2].plot(detuning_ghz, gain, label=f"{delay_ps:g} ps skew")
    axes[0, 2].axvspan(-2.5, 2.5, color="#cccccc", alpha=0.3, label="±2.5 GHz")
    axes[0, 2].set_xlabel("drift from calibrated carrier (GHz)")
    axes[0, 2].set_ylabel("multiplicative score gain")
    axes[0, 2].set_title("(c) Frequency-drift sensitivity")
    axes[0, 2].legend(fontsize=8)

    receiver_styles = {
        "burau_directional_t_guardian": (
            "directional Burau T",
            "#0072b2",
            "o",
        ),
        "loss_matched_mode_sorter": ("mode sorter", "#1b9e77", "^"),
        "quadrant_detector": ("quadrant detector", "#e7298a", "s"),
        "scalar_power_monitor": ("scalar power", "#666666", "x"),
    }
    for power_name, linestyle in [("seed_only", "--"), ("boosted", "-")]:
        for receiver_name, (label, color, marker) in receiver_styles.items():
            values = [
                item[receiver_name]["fault_detection_probability"]
                for item in detection[power_name]["by_range"]
            ]
            axes[1, 0].semilogx(
                RANGES_KM,
                values,
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=f"{label}, {power_name.replace('_', ' ')}",
            )
    axes[1, 0].axhline(
        config.empirical_false_alarm_probability,
        color="black",
        linewidth=0.8,
        linestyle=":",
    )
    axes[1, 0].set_ylim(-0.02, 1.03)
    axes[1, 0].set_xlabel("inter-satellite range (km)")
    axes[1, 0].set_ylabel("fault detection probability")
    axes[1, 0].set_title("(d) Detection at 1% FA target")
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].semilogy(
        rin["rin_dbc_per_hz"],
        rin["raw_difference_score_rmse"],
        marker="o",
        label="raw port difference",
    )
    axes[1, 1].semilogy(
        rin["rin_dbc_per_hz"],
        rin["normalized_difference_score_rmse"],
        marker="o",
        label="difference / sum",
    )
    axes[1, 1].set_xlabel("one-sided white laser RIN (dBc/Hz)")
    axes[1, 1].set_ylabel("score RMSE")
    axes[1, 1].set_title("(e) Common-power rejection")
    axes[1, 1].legend(fontsize=8)

    severity_rows = severity_sweep["rows"]
    biases = [row["pointing_bias_urad"] for row in severity_rows]
    severity_styles = {
        "burau_directional_t_guardian": (
            "directional Burau T",
            "#0072b2",
            "o",
            "-",
        ),
        "loss_matched_mode_sorter": (
            "loss-matched mode sorter",
            "#1b9e77",
            "^",
            "--",
        ),
        "quadrant_detector": ("quadrant detector", "#e7298a", "s", "-"),
        "scalar_power_monitor": ("scalar power", "#666666", "x", ":"),
    }
    for receiver_name, (label, color, marker, linestyle) in severity_styles.items():
        values = [
            row[receiver_name]["fault_detection_probability"]
            for row in severity_rows
        ]
        axes[1, 2].plot(
            biases,
            values,
            marker=marker,
            color=color,
            linestyle=linestyle,
            label=label,
        )
    axes[1, 2].axvline(
        config.receiver_fault_bias_urad,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="default stress case",
    )
    axes[1, 2].set_ylim(-0.02, 1.03)
    axes[1, 2].set_xlabel("mean pointing bias (urad)")
    axes[1, 2].set_ylabel("fault detection probability")
    axes[1, 2].set_title(f"(f) Fault severity at {RANGES_KM[-1]:g} km")
    axes[1, 2].legend(fontsize=8)

    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Phase-0 model: directional Burau T and practical SATCOM baselines",
        fontsize=12,
    )
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_satcom_guardian_study(
    config: GuardianConfig | None = None,
) -> dict[str, object]:
    """Run the reduced-order link, source, T-chip, and detector experiments."""
    if config is None:
        config = GuardianConfig()
    if not 0.0 < config.guardian_tap_fraction < 1.0:
        raise ValueError("guardian_tap_fraction must be between zero and one")
    if not 0.0 < config.directional_t_x_power_fraction < 1.0:
        raise ValueError("directional_t_x_power_fraction must be between zero and one")
    if config.qpd_equivalent_spot_radius_urad <= 0.0:
        raise ValueError("qpd_equivalent_spot_radius_urad must be positive")
    if config.monte_carlo_trials_per_class < 1_000:
        raise ValueError("at least 1,000 trials per class are required")

    exact_checks = _exact_t_checks()
    seed_counts = [
        float(
            _guardian_photoelectrons(
                config,
                config.epitaxial_seed_monitored_carrier_power_w,
                range_km,
                0.0,
            )
        )
        for range_km in RANGES_KM
    ]
    boosted_counts = [
        float(
            _guardian_photoelectrons(
                config,
                config.boosted_monitored_carrier_power_w,
                range_km,
                0.0,
            )
        )
        for range_km in RANGES_KM
    ]
    core_transmission = 10.0 ** (-config.guardian_insertion_loss_db / 10.0)
    link_budget = {
        "ranges_km": RANGES_KM.tolist(),
        "seed_only_post_t_core_photoelectrons_per_window": seed_counts,
        "boosted_post_t_core_photoelectrons_per_window": boosted_counts,
        "seed_only_pre_core_photoelectrons_per_window": [
            count / core_transmission for count in seed_counts
        ],
        "boosted_pre_core_photoelectrons_per_window": [
            count / core_transmission for count in boosted_counts
        ],
        "main_path_tap_penalty_db": float(
            -10.0 * np.log10(1.0 - config.guardian_tap_fraction)
        ),
        "booster_gain_from_seed_db": float(
            10.0
            * np.log10(
                config.boosted_monitored_carrier_power_w
                / config.epitaxial_seed_monitored_carrier_power_w
            )
        ),
        "power_interpretation": (
            "each power is the optical power in the one carrier scored by this "
            "model; it is not aggregate WDM terminal power"
        ),
    }

    visibility_grid = [
        [
            _laser_visibility(float(linewidth), float(delay_ps) * 1e-12)
            for delay_ps in COHERENCE_DELAYS_PS
        ]
        for linewidth in LINEWIDTHS_HZ
    ]
    detuning_ghz = np.linspace(-5.0, 5.0, 101)
    detuning_gain = {
        str(float(delay_ps)): np.cos(
            2.0 * np.pi * detuning_ghz * 1e9 * delay_ps * 1e-12
        ).tolist()
        for delay_ps in DETUNING_DELAYS_PS
    }

    detection: dict[str, dict[str, object]] = {}
    powers = {
        "seed_only": config.epitaxial_seed_monitored_carrier_power_w,
        "boosted": config.boosted_monitored_carrier_power_w,
    }
    for power_index, (power_name, monitored_carrier_power) in enumerate(
        powers.items()
    ):
        by_range = []
        for range_index, range_km in enumerate(RANGES_KM):
            local_seed = config.seed + power_index * 10_000 + range_index * 100
            calibration = _simulate_population(
                config,
                range_km=float(range_km),
                transmit_power_w=monitored_carrier_power,
                transmitter_bias_urad=0.0,
                receiver_bias_urad=0.0,
                random_seed=local_seed,
            )
            normal = _simulate_population(
                config,
                range_km=float(range_km),
                transmit_power_w=monitored_carrier_power,
                transmitter_bias_urad=0.0,
                receiver_bias_urad=0.0,
                random_seed=local_seed + 1,
            )
            fault = _simulate_population(
                config,
                range_km=float(range_km),
                transmit_power_w=monitored_carrier_power,
                transmitter_bias_urad=0.0,
                receiver_bias_urad=config.receiver_fault_bias_urad,
                random_seed=local_seed + 2,
            )
            calibration_anomalies = _population_anomalies(calibration)
            normal_anomalies = _population_anomalies(normal)
            fault_anomalies = _population_anomalies(fault)
            metrics = {
                name: _operating_point(
                    calibration_anomalies[name],
                    normal_anomalies[name],
                    fault_anomalies[name],
                    config.empirical_false_alarm_probability,
                )
                for name in calibration_anomalies
            }
            by_range.append(
                {
                    "range_km": float(range_km),
                    "boresight_photoelectrons_per_window": float(
                        _guardian_photoelectrons(
                            config,
                            monitored_carrier_power,
                            float(range_km),
                            0.0,
                        )
                    ),
                    "nominal_mean_true_directional_t_radial_score": float(
                        np.mean(
                            normal["ideal_directional_t_radial_discriminant"]
                        )
                    ),
                    "fault_mean_true_directional_t_radial_score": float(
                        np.mean(
                            fault["ideal_directional_t_radial_discriminant"]
                        )
                    ),
                    "fault_mean_true_qpd_radial_discriminant": float(
                        np.mean(fault["ideal_qpd_radial_discriminant"])
                    ),
                    "mean_fault_total_power_reduction_fraction": float(
                        1.0
                        - np.mean(fault["mean_photoelectrons"])
                        / np.mean(normal["mean_photoelectrons"])
                    ),
                    **metrics,
                }
            )
        detection[power_name] = {
            "monitored_carrier_power_w": monitored_carrier_power,
            "by_range": by_range,
        }

    rin = _rin_sensitivity(config, np.random.default_rng(config.seed + 99_999))
    severity_sweep = _fault_severity_sweep(config)
    qpd_design_sensitivity = _qpd_design_sensitivity(config)
    guardian_loss_sensitivity = _guardian_loss_sensitivity(config)
    directional_t_complement_phase_ablation = (
        _directional_t_complement_phase_ablation(config)
    )
    directional_t_design = _directional_t_design_characterization(config)
    assessed_range_summary = _assessed_range_summary(detection)
    literature_range_context = _literature_range_context(config)
    technology_trade_study = _technology_trade_study(
        detection, qpd_design_sensitivity
    )
    allowable_delay_99_visibility_ps = {
        str(float(linewidth)): float(
            -np.log(0.99) / (np.pi * linewidth) * 1e12
        )
        for linewidth in LINEWIDTHS_HZ
    }
    detuning_delay_99_gain_ps = float(
        np.arccos(0.99) / (2.0 * np.pi * 2.5e9) * 1e12
    )
    default_delay_s = config.t_arm_delay_mismatch_ps * 1e-12
    detuning_99_gain_at_default_delay_ghz = float(
        np.arccos(0.99) / (2.0 * np.pi * default_delay_s) / 1e9
    )
    representative_thermal_frequency_coefficient_ghz_per_k = 76.0
    diagnostic_flagged_windows_per_second = (
        config.empirical_false_alarm_probability
        / (config.decision_window_ns * 1e-9)
    )
    derived_scales = {
        "decision_rate_hz": 1.0 / (config.decision_window_ns * 1e-9),
        "raw_flagged_windows_per_second_at_target_false_alarm": (
            diagnostic_flagged_windows_per_second
        ),
        "payload_bits_per_lane_per_decision_window": (
            config.payload_lane_rate_gbps * config.decision_window_ns
        ),
        "illustrative_aggregate_raw_rate_tbps": (
            config.payload_lane_rate_gbps
            * config.illustrative_wdm_lanes
            / 1_000.0
        ),
        "maximum_delay_ps_for_99_percent_visibility_by_linewidth_hz": (
            allowable_delay_99_visibility_ps
        ),
        "maximum_delay_ps_for_99_percent_score_gain_at_2_5_ghz_detuning": (
            detuning_delay_99_gain_ps
        ),
        "representative_inp_on_si_frequency_shift_ghz_per_k": (
            representative_thermal_frequency_coefficient_ghz_per_k
        ),
        "illustrative_temperature_change_k_corresponding_to_2_5_ghz_shift": (
            2.5 / representative_thermal_frequency_coefficient_ghz_per_k
        ),
        "maximum_frequency_offset_ghz_for_99_percent_score_gain_at_default_delay": (
            detuning_99_gain_at_default_delay_ghz
        ),
        "temperature_change_k_at_99_percent_score_gain_for_default_delay": (
            detuning_99_gain_at_default_delay_ghz
            / representative_thermal_frequency_coefficient_ghz_per_k
        ),
        "thermal_shift_note": (
            "76 GHz/K is the low end inferred from the measured wavelength "
            "shift of one 1.55-um InP-on-Si Fabry-Perot prototype; it is a "
            "literature stage-temperature scale that includes self-heating, "
            "not a universal epitaxial-laser coefficient or a T-chip thermal "
            "model; the 2.5-GHz conversion is illustrative, whereas the "
            "default-delay limit is derived directly from the interferometer"
        ),
    }

    if not all(
        earlier > later
        for earlier, later in zip(boosted_counts, boosted_counts[1:])
    ):
        raise AssertionError("boresight photon budget must decrease with range")
    if rin["normalized_difference_score_rmse"][-1] >= rin[
        "raw_difference_score_rmse"
    ][-1]:
        raise AssertionError("balanced normalization did not reject strong RIN")
    boosted_mid = detection["boosted"]["by_range"][2]
    if (
        boosted_mid["burau_directional_t_guardian"]["area_under_roc"]
        <= boosted_mid["scalar_power_monitor"]["area_under_roc"]
    ):
        raise AssertionError("directional T score did not distinguish the fault")
    if (
        boosted_mid["quadrant_detector"]["area_under_roc"]
        <= boosted_mid["scalar_power_monitor"]["area_under_roc"]
    ):
        raise AssertionError("quadrant baseline did not distinguish the selected fault")

    diagnostics: dict[str, object] = {
        "schema_version": 5,
        "model_revision": "hybrid_directional_burau_t_bank_v3",
        "scope": (
            "reduced-order numerical engineering study of a parameterized "
            "epitaxial-laser source envelope, optical inter-satellite link, "
            "and out-of-path directional Burau T guardian; not a measured laser, "
            "waveform modem model, hardware result, or space qualification"
        ),
        "status": "simulation only",
        "source_model": {
            "class": (
                "parameterized selected-carrier envelope anchored only to a "
                "published epitaxial-laser output-power scale"
            ),
            "coherence_assumption": (
                "single Lorentzian carrier with first-order coherence magnitude "
                "exp(-pi * FWHM linewidth * |arm delay|)"
            ),
            "rin_assumption": (
                "one-sided white RIN in dBc/Hz over the 1/(2T) equivalent "
                "noise bandwidth of a rectangular decision window"
            ),
            "published_device_caveat": (
                "the cited 155-mW InP-on-Si device is a multimode Fabry-Perot "
                "prototype and did not provide the linewidth or RIN defaults "
                "used here"
            ),
            "joint_sweep_caveat": (
                "linewidth, carrier-drift, and RIN curves are component "
                "sensitivities; the default fault-detection Monte Carlo does "
                "not yet span their worst-case combinations"
            ),
        },
        "configuration_assumptions": asdict(config),
        "mode_basis": [
            "nominal spatial/polarization pilot mode",
            "azimuth pointing leakage",
            "elevation pointing leakage",
            "orthogonal polarization leakage",
        ],
        "t_observable": {
            "primary_architecture": (
                "two parallel balanced T cells with a 50:50 x/y photon "
                "fanout and four detectors total"
            ),
            "path_mixer": (
                "exact reduced-Burau word sigma_2^-1 sigma_1 sigma_2^-1 "
                "at omega=pi/4, phase-equivalent to a balanced coupler"
            ),
            "branch_map": (
                "sigma_y on each piston/tangent pair and +I on the unused "
                "modal complement"
            ),
            "ideal_axis_score": (
                "s_j = 2 a_0 b_j + cos(phi_c)(1-a_0^2-b_j^2), phi_c=0"
            ),
            "measured_axis_score": (
                "(I_plus_j-I_minus_j)/(I_plus_j+I_minus_j)"
            ),
            "alarm_statistic": "sqrt(s_x^2+s_y^2)",
            "hardware_note": (
                "the exact three-letter Burau word supplies the path mixer, "
                "but a generic balanced interferometer implements the same "
                "intensity observable"
            ),
            "control_limit": (
                "signed x/y outputs are local tracking errors; the response is "
                "nonlinear outside the characterized range and acquisition "
                "still needs a wide-field PAT sensor"
            ),
        },
        "receiver_baseline_models": {
            "quadrant_detector": {
                "location": "same optical tap, before modeled T-core insertion loss",
                "detectors": 4,
                "statistic": "radial norm of two Gaussian-spot difference/sum axes",
                "gaussian_axis_response": (
                    "erf(sqrt(2) * pointing / qpd_equivalent_spot_radius)"
                ),
                "noise_fairness": (
                    "same quantum efficiency and per-segment dark/read/gain model "
                    "as each T output; four segments therefore incur four read-noise "
                    "and dark-current contributions"
                ),
                "scope_warning": (
                    "ideal gapless infinite detector with an assumed angular spot "
                    "radius; focal length, PSF, gap, saturation, background, and "
                    "defocus must be measured"
                ),
            },
            "loss_matched_mode_sorter": {
                "location": "post-loss, matched to the T core's assumed insertion loss",
                "detectors": 2,
                "statistic": "nominal-mode power minus residual power over their sum",
                "purpose": (
                    "independent conventional modal-sensing baseline with the "
                    "same assumed insertion loss"
                ),
            },
            "generic_directional_balanced_interferometer": {
                "location": "post-loss, with the same two cells and four outputs",
                "detectors": 4,
                "statistic": "the same signed piston/tangent axis scores",
                "purpose": (
                    "matched architectural control for testing whether the "
                    "Burau implementation adds a hardware advantage"
                ),
            },
            "scalar_power_monitor": {
                "location": "same optical tap, before modeled T-core insertion loss",
                "detectors": 1,
                "purpose": "negative control, not the primary practical baseline",
            },
        },
        "exact_identity_checks": exact_checks,
        "link_budget": link_budget,
        "range_assessment": assessed_range_summary,
        "literature_range_context": literature_range_context,
        "technology_trade_study": technology_trade_study,
        "directional_t_design_characterization": directional_t_design,
        "guardian_insertion_loss_sensitivity": guardian_loss_sensitivity,
        "directional_t_complement_phase_ablation": (
            directional_t_complement_phase_ablation
        ),
        "laser_coherence": {
            "linewidths_hz": LINEWIDTHS_HZ.tolist(),
            "arm_delay_mismatch_ps": COHERENCE_DELAYS_PS.tolist(),
            "visibility_rows_by_linewidth": visibility_grid,
        },
        "laser_frequency_detuning": {
            "frequency_drift_from_calibrated_carrier_ghz": detuning_ghz.tolist(),
            "score_gain_by_delay_mismatch_ps": detuning_gain,
            "phase_reference_assumption": (
                "the nominal static arm phase at the calibrated carrier is "
                "assumed corrected"
            ),
        },
        "monte_carlo_fault_detection": {
            "fault": (
                f"{config.receiver_fault_bias_urad:g}-urad mean receiver-AoA "
                "azimuth displacement with unchanged transmitter pointing, "
                "receiver jitter, and polarization "
                "distributions; this is a selected stress case, not a fitted "
                "fault-occurrence distribution"
            ),
            "comparison": (
                "four-output directional Burau T versus (1) a four-segment "
                "quadrant pointing detector on the pre-core tap, (2) a "
                "loss-matched conventional mode sorter, and (3) a one-detector pre-core "
                "scalar-power negative control"
            ),
            "target_false_alarm_probability": (
                config.empirical_false_alarm_probability
            ),
            "rare_event_warning": (
                "threshold calibration, nominal evaluation, and fault evaluation "
                f"use independent {config.monte_carlo_trials_per_class:,}-trial "
                "populations; the 1% operating point would flag about "
                f"{diagnostic_flagged_windows_per_second:,.0f} "
                f"independent {config.decision_window_ns:g}-ns windows per second "
                "and is only a diagnostic ROC point, not an operational alarm "
                "or mission-reliability claim"
            ),
            "results_by_transmit_power": detection,
        },
        "pointing_fault_severity_sweep": severity_sweep,
        "qpd_design_sensitivity": qpd_design_sensitivity,
        "rin_common_mode_test": rin,
        "derived_engineering_scales": derived_scales,
        "literature_anchors": [
            {
                "fact": (
                    "TBIRD used a quad sensor for signed two-axis payload feedback; "
                    "on orbit its overall closed-loop pointing was 20--35 urad RMS "
                    "per axis while supporting 100/200-Gbit/s demonstrations"
                ),
                "url": "https://ntrs.nasa.gov/citations/20230000001",
            },
            {
                "fact": (
                    "SDA OCT v4 treats PAT as part of the physical layer, requires "
                    "2.5-W maximum output capability and a 5,500-km continuous "
                    "irradiance point, defines 20,000-km burst modes, and reports "
                    "fast received power plus frame-sync status"
                ),
                "url": (
                    "https://www.sda.mil/wp-content/uploads/2024/07/"
                    "SDA_OCT_Standard_4.0.0_final-20240701.pdf"
                ),
            },
            {
                "fact": (
                    "a 2024 spaceborne laser-communication study implemented a "
                    "four-quadrant spot-position scheme designed for limited onboard "
                    "compute and memory"
                ),
                "url": "https://doi.org/10.1364/AO.517934",
            },
            {
                "fact": (
                    "calibrated first-order spatial-mode projections can attain "
                    "two-dimensional displacement information beyond direct imaging"
                ),
                "url": "https://doi.org/10.1364/OPTICA.404746",
            },
            {
                "fact": (
                    "ESA's FastSwitching terminal development replaces heritage "
                    "cascaded four-quadrant acquisition/tracking diodes with a pixel "
                    "detector, establishing the pixel sensor as a practical secondary "
                    "PAT baseline"
                ),
                "url": "https://resilience.esa.int/archives/projects/fastswitching",
            },
            {
                "fact": (
                    "experimental free-space mode-diversity reception has been "
                    "demonstrated with non-mode-selective photonic lanterns"
                ),
                "url": "https://doi.org/10.1109/JPHOT.2022.3225337",
            },
            {
                "fact": (
                    "programmable photonic circuits can self-calibrate, but the "
                    "published demonstration explicitly addresses fabrication "
                    "variation, thermal gradients, and thermal crosstalk"
                ),
                "url": "https://doi.org/10.1038/s41566-022-01020-z",
            },
            {
                "fact": (
                    "a 1.55-um InP-on-Si research laser exceeded 155 mW per "
                    "facet and operated CW to 120 C"
                ),
                "url": "https://doi.org/10.1038/s41377-024-01389-2",
            },
        ],
        "omitted_physics": [
            "general off-axis aperture diffraction and truncation integrals",
            "terminal aberrations, stray light, and detector saturation",
            "semiconductor optical-amplifier or EDFA noise",
            "laser rate equations, side modes, mode hops, and optical feedback",
            "frequency-dependent measured T-arm transfer matrices",
            "T-chip thermo-optic path drift",
            "symbol waveform, Doppler/carrier loops, modulation, and FEC",
            "actuator dynamics, network routing, and delivered-goodput events",
            "radiation damage and component aging trajectories",
        ],
        "next_model_gates": [
            (
                "replace assumed source parameters with measured spectrum, "
                "RIN, and L-I data"
            ),
            (
                "replace the modal surrogate with measured telescope/"
                "photonic-lantern fields"
            ),
            (
                "replace ideal quadrant and loss-matched sorter assumptions with "
                "measured transfer functions, gaps, losses, bandwidths, and drift"
            ),
            "run joint laser-impairment, modal-fault, and chip-temperature sweeps",
            (
                "add a waveform-level digital matched filter, pointing-ramp warning "
                "lead time, and modem RSSI/frame-sync/FEC loss-of-lock curves"
            ),
            "add optical feedback, WDM lane failure, redundancy, and network goodput",
            "run importance sampling before claiming rare false-alarm probabilities",
        ],
    }

    _plot_results(
        config,
        link_budget,
        detection,
        rin,
        severity_sweep,
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {FIGURE_PATH.relative_to(Path.cwd())}")
    print(f"Wrote {RESULTS_PATH.relative_to(Path.cwd())}")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return diagnostics


if __name__ == "__main__":
    run_satcom_guardian_study()
