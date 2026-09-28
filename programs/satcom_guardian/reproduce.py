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
QPD_SPOT_RADII_URAD = np.asarray([2.0, 4.0, 8.0])


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
    nominal_pointing_jitter_urad: float = 0.25
    fault_pointing_bias_urad: float = 1.50
    collected_mode_scale_urad: float = 4.0
    qpd_equivalent_spot_radius_urad: float = 4.0
    polarization_jitter_deg: float = 1.0
    monte_carlo_trials_per_class: int = 30_000
    empirical_false_alarm_probability: float = 0.01
    seed: int = 20_260_909


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


def _nominal_mode_power(
    config: GuardianConfig,
    point_x_urad: np.ndarray,
    point_y_urad: np.ndarray,
    polarization_angle_rad: np.ndarray,
) -> np.ndarray:
    """Power in the expected mode of a four-mode collected basis."""
    radial_squared = point_x_urad**2 + point_y_urad**2
    spatial_match = np.exp(
        -2.0 * radial_squared / config.collected_mode_scale_urad**2
    )
    polarization_match = np.cos(polarization_angle_rad) ** 2
    return np.clip(spatial_match * polarization_match, 0.0, 1.0)


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
    fault: bool,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    trials = config.monte_carlo_trials_per_class
    bias = config.fault_pointing_bias_urad if fault else 0.0
    point_x = rng.normal(bias, config.nominal_pointing_jitter_urad, trials)
    point_y = rng.normal(0.0, config.nominal_pointing_jitter_urad, trials)
    polarization = rng.normal(
        0.0, np.deg2rad(config.polarization_jitter_deg), trials
    )
    radial_urad = np.sqrt(point_x**2 + point_y**2)
    expected_mode_power = _nominal_mode_power(
        config, point_x, point_y, polarization
    )
    ideal_score = 2.0 * expected_mode_power - 1.0

    mean_photoelectrons = _guardian_photoelectrons(
        config,
        transmit_power_w,
        range_km,
        radial_urad * 1e-6,
    )
    rin_factor, _ = _rin_common_power(config, rng, trials)
    delay_s = config.t_arm_delay_mismatch_ps * 1e-12
    visibility = _laser_visibility(config.laser_linewidth_hz, delay_s)
    static_phase = (
        2.0
        * np.pi
        * config.laser_frequency_drift_from_calibrated_carrier_ghz
        * 1e9
        * delay_s
    )
    phase = static_phase + rng.normal(
        0.0, config.residual_phase_jitter_rms_rad, trials
    )
    interference_factor = visibility * np.cos(phase)
    observed_ideal_score = interference_factor * ideal_score
    plus_fraction = 0.5 * (1.0 + observed_ideal_score)
    minus_fraction = 1.0 - plus_fraction

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

    plus = rng.poisson(total_signal_mean * plus_fraction + dark_per_port).astype(float)
    minus = rng.poisson(
        total_signal_mean * minus_fraction + dark_per_port
    ).astype(float)
    plus += rng.normal(0.0, config.detector_read_noise_e_rms, trials)
    minus += rng.normal(0.0, config.detector_read_noise_e_rms, trials)
    plus = plus_gain * (plus - dark_per_port)
    minus = minus_gain * (minus - dark_per_port)
    measured_total = plus + minus
    normalized_score = (plus - minus) / np.maximum(measured_total, 1.0)

    # A loss-matched direct mode sorter measures the same nominal-versus-residual
    # observable without a phase-sensitive T interferometer.  Matching its loss,
    # detector count, and readout noise isolates the T core's coherence burden.
    sorter_nominal = rng.poisson(
        total_signal_mean * expected_mode_power + dark_per_port
    ).astype(float)
    sorter_residual = rng.poisson(
        total_signal_mean * (1.0 - expected_mode_power) + dark_per_port
    ).astype(float)
    sorter_nominal += rng.normal(0.0, config.detector_read_noise_e_rms, trials)
    sorter_residual += rng.normal(0.0, config.detector_read_noise_e_rms, trials)
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
    scalar = rng.poisson(pre_core_signal_mean + dark_per_port).astype(float)
    scalar += rng.normal(0.0, config.detector_read_noise_e_rms, trials)
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
    true_qpd_x = erf(np.sqrt(2.0) * point_x / qpd_scale)
    true_qpd_y = erf(np.sqrt(2.0) * point_y / qpd_scale)
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
    qpd = rng.poisson(
        pre_core_signal_mean[:, None] * quadrant_fractions + dark_per_port
    ).astype(float)
    qpd += rng.normal(
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
        "normalized_t_score": normalized_score,
        "normalized_loss_matched_mode_sorter_score": normalized_sorter_score,
        "normalized_scalar_power": normalized_scalar_power,
        "qpd_radial_discriminant": qpd_radial,
        "qpd_x_discriminant": qpd_x,
        "qpd_y_discriminant": qpd_y,
        "ideal_t_score": ideal_score,
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
    false_alarm_count = int(np.count_nonzero(normal_anomaly >= threshold))
    detection_count = int(np.count_nonzero(fault_anomaly >= threshold))
    return {
        "threshold": threshold,
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


def _exact_t_checks() -> dict[str, float]:
    rng = np.random.default_rng(1701)
    dimension = 4
    raw_h = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    h = raw_h / la.norm(raw_h)
    projector = np.outer(h, h.conj())
    identity = np.eye(dimension)
    x = rng.normal(size=(512, dimension)) + 1j * rng.normal(
        size=(512, dimension)
    )
    x /= la.norm(x, axis=1, keepdims=True)
    y = 2.0 * projector - identity
    x_branch = x
    y_branch = x @ y.T
    plus = 0.5 * (x_branch + y_branch)
    minus = 0.5 * (x_branch - y_branch)
    plus_power = np.sum(np.abs(plus) ** 2, axis=1)
    minus_power = np.sum(np.abs(minus) ** 2, axis=1)
    matched_power = np.abs(x @ h.conj()) ** 2
    expected_score = 2.0 * matched_power - 1.0
    measured_score = plus_power - minus_power
    global_phases = np.exp(1j * rng.uniform(-np.pi, np.pi, len(x)))
    phase_shifted = x * global_phases[:, None]
    shifted_matched_power = np.abs(phase_shifted @ h.conj()) ** 2
    checks = {
        "householder_unitarity_residual": float(la.norm(y.conj().T @ y - identity)),
        "maximum_plus_port_projector_error": float(
            np.max(np.abs(plus_power - matched_power))
        ),
        "maximum_minus_port_residual_error": float(
            np.max(np.abs(minus_power - (1.0 - matched_power)))
        ),
        "maximum_energy_checksum_error": float(
            np.max(np.abs(plus_power + minus_power - 1.0))
        ),
        "maximum_quadratic_score_error": float(
            np.max(np.abs(measured_score - expected_score))
        ),
        "maximum_global_phase_error": float(
            np.max(np.abs(shifted_matched_power - matched_power))
        ),
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
    if max(checks.values()) > 2e-12:
        raise AssertionError("ideal T guardian identities failed")
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
        "t_guardian": -population["normalized_t_score"],
        "loss_matched_mode_sorter": -population[
            "normalized_loss_matched_mode_sorter_score"
        ],
        "quadrant_detector": population["qpd_radial_discriminant"],
        "scalar_power_monitor": -population["normalized_scalar_power"],
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
        fault=False,
        rng=np.random.default_rng(seed),
    )
    normal = _simulate_population(
        config,
        range_km=range_km,
        transmit_power_w=transmit_power_w,
        fault=False,
        rng=np.random.default_rng(seed + 1),
    )
    calibration_anomalies = _population_anomalies(calibration)
    normal_anomalies = _population_anomalies(normal)
    rows = []
    for index, bias_urad in enumerate(FAULT_BIASES_URAD):
        sweep_config = replace(config, fault_pointing_bias_urad=float(bias_urad))
        fault = _simulate_population(
            sweep_config,
            range_km=range_km,
            transmit_power_w=transmit_power_w,
            fault=True,
            rng=np.random.default_rng(seed + 100 + index),
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
                    bias_urad / config.nominal_pointing_jitter_urad
                ),
                "bias_over_collected_mode_scale": float(
                    bias_urad / config.collected_mode_scale_urad
                ),
                "mean_true_t_score": float(np.mean(fault["ideal_t_score"])),
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
                fault=False,
                rng=np.random.default_rng(common_seed),
            )
            normal = _simulate_population(
                local_config,
                range_km=range_km,
                transmit_power_w=transmit_power_w,
                fault=False,
                rng=np.random.default_rng(common_seed + 1),
            )
            fault = _simulate_population(
                local_config,
                range_km=range_km,
                transmit_power_w=transmit_power_w,
                fault=True,
                rng=np.random.default_rng(common_seed + 2),
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
        "t_guardian",
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
        "burau_t_guardian": {
            "signed_pointing_control_output": 1,
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
            "t_guardian",
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
            "primary_pointing_baseline": "quadrant_detector_pat",
            "mandatory_system_baseline": (
                "modem_fec_link_telemetry_plus_existing_pat"
            ),
            "matched_observable_control": "conventional_mode_sorter",
            "negative_control_only": "scalar_power_tap",
            "pixel_sensor_role": (
                "secondary baseline when acquisition field of view, multi-spot "
                "disambiguation, or non-Gaussian imagery matters"
            ),
        },
        "current_decision": {
            "burau_specific_advantage_demonstrated": False,
            "reason": (
                "For the modeled pointing fault the T output is an unsigned scalar, "
                "whereas a quadrant detector supplies the two signed control axes. "
                "For general modal residual sensing, a conventional mode sorter "
                "measures the same observable without Burau-specific provenance. "
                "The apparent seed-only QPD deficit at the default spot scale closes "
                "when the assumed spot radius is reduced, showing that it is not an "
                "architecture-independent T advantage."
            ),
            "reconsider_if": (
                "measured hardware shows a loss, bandwidth, stability, fault-coverage, "
                "or SWaP advantage over both the quadrant detector and a conventional "
                "implementation of the same projector"
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
        "t_guardian": ("T guardian", "#7570b3", "o"),
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
        "t_guardian": ("T guardian", "#7570b3", "o", "-"),
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
        config.fault_pointing_bias_urad,
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
        "Phase-0 model: T guardian and practical SATCOM baselines",
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
                fault=False,
                rng=np.random.default_rng(local_seed),
            )
            normal = _simulate_population(
                config,
                range_km=float(range_km),
                transmit_power_w=monitored_carrier_power,
                fault=False,
                rng=np.random.default_rng(local_seed + 1),
            )
            fault = _simulate_population(
                config,
                range_km=float(range_km),
                transmit_power_w=monitored_carrier_power,
                fault=True,
                rng=np.random.default_rng(local_seed + 2),
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
                    "nominal_mean_true_t_score": float(
                        np.mean(normal["ideal_t_score"])
                    ),
                    "fault_mean_true_t_score": float(
                        np.mean(fault["ideal_t_score"])
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
        boosted_mid["t_guardian"]["area_under_roc"]
        <= boosted_mid["scalar_power_monitor"]["area_under_roc"]
    ):
        raise AssertionError("modal score did not distinguish the selected fault")
    if (
        boosted_mid["quadrant_detector"]["area_under_roc"]
        <= boosted_mid["scalar_power_monitor"]["area_under_roc"]
    ):
        raise AssertionError("quadrant baseline did not distinguish the selected fault")

    diagnostics: dict[str, object] = {
        "schema_version": 3,
        "scope": (
            "reduced-order numerical engineering study of a parameterized "
            "epitaxial-laser source envelope, optical inter-satellite link, "
            "and out-of-path T/Householder guardian; not a measured laser, "
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
            "projector": "P0 = |h><h|",
            "branches": "X = I, Y = 2 P0 - I",
            "plus_port": "|<h,x>|^2",
            "minus_port": "||x||^2 - |<h,x>|^2",
            "normalized_score": "(I_plus - I_minus) / (I_plus + I_minus)",
            "hardware_note": (
                "this Householder score is shallow, is exactly the same "
                "observable as an ideal nominal-mode sorter, and does not "
                "require the 132-letter general fixed-omega compiler"
            ),
            "control_limit": (
                "the single score is even in pointing displacement and therefore "
                "does not provide signed azimuth/elevation steering errors"
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
                    "same-observable control that removes T-arm linewidth, delay, "
                    "phase-jitter, and detuning sensitivity"
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
                f"{config.fault_pointing_bias_urad:g}-urad mean azimuth "
                "displacement with unchanged pointing-jitter and polarization "
                "distributions; this is a selected stress case, not a fitted "
                "fault-occurrence distribution"
            ),
            "comparison": (
                "normalized T modal-residual score versus (1) a four-segment "
                "quadrant pointing detector on the pre-core tap, (2) a loss-matched "
                "two-output conventional mode sorter measuring the same projector, "
                "and (3) a one-detector pre-core scalar-power negative control"
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
