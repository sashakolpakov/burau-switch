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
    plus = rng.poisson(total_signal_mean * plus_fraction + dark_per_port).astype(float)
    minus = rng.poisson(
        total_signal_mean * minus_fraction + dark_per_port
    ).astype(float)
    plus += rng.normal(0.0, config.detector_read_noise_e_rms, trials)
    minus += rng.normal(0.0, config.detector_read_noise_e_rms, trials)
    mismatch = config.detector_gain_mismatch_fraction
    plus_gain = 1.0 + 0.5 * mismatch
    minus_gain = 1.0 - 0.5 * mismatch
    plus = plus_gain * (plus - dark_per_port)
    minus = minus_gain * (minus - dark_per_port)
    measured_total = plus + minus
    normalized_score = (plus - minus) / np.maximum(measured_total, 1.0)

    boresight_guardian_photoelectrons = float(
        _guardian_photoelectrons(config, transmit_power_w, range_km, 0.0)
    )
    core_transmission = 10.0 ** (-config.guardian_insertion_loss_db / 10.0)
    scalar_signal_mean = mean_photoelectrons * rin_factor / core_transmission
    scalar = rng.poisson(scalar_signal_mean + dark_per_port).astype(float)
    scalar += rng.normal(0.0, config.detector_read_noise_e_rms, trials)
    scalar -= dark_per_port
    scalar_boresight_photoelectrons = (
        boresight_guardian_photoelectrons / core_transmission
    )
    normalized_scalar_power = scalar / max(scalar_boresight_photoelectrons, 1.0)
    return {
        "normalized_t_score": normalized_score,
        "normalized_scalar_power": normalized_scalar_power,
        "ideal_t_score": ideal_score,
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
    t_calibration = -calibration["normalized_t_score"]
    t_normal = -normal["normalized_t_score"]
    power_calibration = -calibration["normalized_scalar_power"]
    power_normal = -normal["normalized_scalar_power"]
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
        t_metrics = _operating_point(
            t_calibration,
            t_normal,
            -fault["normalized_t_score"],
            config.empirical_false_alarm_probability,
        )
        power_metrics = _operating_point(
            power_calibration,
            power_normal,
            -fault["normalized_scalar_power"],
            config.empirical_false_alarm_probability,
        )
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
                "t_or_ideal_mode_sorter": t_metrics,
                "scalar_power_monitor": power_metrics,
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


def _plot_results(
    config: GuardianConfig,
    link_budget: dict[str, list[float]],
    detection: dict[str, dict[str, list[dict[str, float]]]],
    rin: dict[str, list[float]],
    severity_sweep: dict[str, object],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 8.0))

    axes[0, 0].loglog(
        RANGES_KM,
        link_budget["seed_only_photoelectrons_per_window"],
        marker="o",
        label=(
            f"{config.epitaxial_seed_monitored_carrier_power_w * 1e3:.0f} mW "
            "monitored carrier"
        ),
    )
    axes[0, 0].loglog(
        RANGES_KM,
        link_budget["boosted_photoelectrons_per_window"],
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

    colors = {"seed_only": "#d95f02", "boosted": "#1b9e77"}
    for power_name, linestyle in [("seed_only", "--"), ("boosted", "-")]:
        t_values = [
            item["t_guardian"]["fault_detection_probability"]
            for item in detection[power_name]["by_range"]
        ]
        power_values = [
            item["power_monitor"]["fault_detection_probability"]
            for item in detection[power_name]["by_range"]
        ]
        axes[1, 0].semilogx(
            RANGES_KM,
            t_values,
            marker="o",
            linestyle=linestyle,
            color=colors[power_name],
            label=f"T/mode score, {power_name.replace('_', ' ')}",
        )
        axes[1, 0].semilogx(
            RANGES_KM,
            power_values,
            marker="x",
            linestyle=linestyle,
            color=colors[power_name],
            label=f"power only, {power_name.replace('_', ' ')}",
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
    t_detection = [
        row["t_or_ideal_mode_sorter"]["fault_detection_probability"]
        for row in severity_rows
    ]
    power_detection = [
        row["scalar_power_monitor"]["fault_detection_probability"]
        for row in severity_rows
    ]
    axes[1, 2].plot(biases, t_detection, marker="o", label="T / ideal mode sorter")
    axes[1, 2].plot(biases, power_detection, marker="x", label="scalar power")
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
        "Phase-0 model: parameterized laser, optical ISL, and T guardian",
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
    link_budget = {
        "ranges_km": RANGES_KM.tolist(),
        "seed_only_photoelectrons_per_window": seed_counts,
        "boosted_photoelectrons_per_window": boosted_counts,
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
            t_calibration_anomaly = -calibration["normalized_t_score"]
            t_normal_anomaly = -normal["normalized_t_score"]
            t_fault_anomaly = -fault["normalized_t_score"]
            power_calibration_anomaly = -calibration["normalized_scalar_power"]
            power_normal_anomaly = -normal["normalized_scalar_power"]
            power_fault_anomaly = -fault["normalized_scalar_power"]
            t_metrics = _operating_point(
                t_calibration_anomaly,
                t_normal_anomaly,
                t_fault_anomaly,
                config.empirical_false_alarm_probability,
            )
            power_metrics = _operating_point(
                power_calibration_anomaly,
                power_normal_anomaly,
                power_fault_anomaly,
                config.empirical_false_alarm_probability,
            )
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
                    "mean_fault_total_power_reduction_fraction": float(
                        1.0
                        - np.mean(fault["mean_photoelectrons"])
                        / np.mean(normal["mean_photoelectrons"])
                    ),
                    "t_guardian": t_metrics,
                    "power_monitor": power_metrics,
                }
            )
        detection[power_name] = {
            "monitored_carrier_power_w": monitored_carrier_power,
            "by_range": by_range,
        }

    rin = _rin_sensitivity(config, np.random.default_rng(config.seed + 99_999))
    severity_sweep = _fault_severity_sweep(config)
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
        <= boosted_mid["power_monitor"]["area_under_roc"]
    ):
        raise AssertionError("modal score did not distinguish the selected fault")

    diagnostics: dict[str, object] = {
        "schema_version": 2,
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
        },
        "exact_identity_checks": exact_checks,
        "link_budget": link_budget,
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
                "normalized T modal-residual score versus scalar total-power "
                "alarm on the same optical tap; the scalar baseline is detected "
                "before T-core insertion loss with one detector, while the T "
                "score uses two post-core detectors"
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
        "rin_common_mode_test": rin,
        "derived_engineering_scales": derived_scales,
        "literature_anchors": [
            {
                "fact": (
                    "200-Gbit/s space-to-ground optical transmission was "
                    "demonstrated by NASA TBIRD"
                ),
                "url": (
                    "https://www.nasa.gov/centers-and-facilities/goddard/"
                    "nasa-partners-achieve-fastest-space-to-ground-laser-comms-link/"
                ),
            },
            {
                "fact": (
                    "SDA OCT v4 uses a C-band 100-GHz grid and includes "
                    "2.5-Gbaud interoperable modes"
                ),
                "url": (
                    "https://www.sda.mil/wp-content/uploads/2024/07/"
                    "SDA_OCT_Standard_4.0.0_final-20240701.pdf"
                ),
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
                "compare with an ideal digital matched filter and a "
                "conventional mode sorter"
            ),
            "run joint laser-impairment, modal-fault, and chip-temperature sweeps",
            "add pointing-ramp warning lead time and modem loss-of-lock curves",
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
