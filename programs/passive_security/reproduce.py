"""Receiver-level study for a passive high-dimensional optical security key.

An exact public Burau mixer exposes the sum and difference of two independently
fabricated high-dimensional passive branch maps.  Device-specific structure
and privately phase-randomized weak coherent challenges supply the modeled
anti-emulation gap.  This script evaluates a deliberately narrow
challenge-estimation model; it is not a hardware demonstration or a complete
cryptographic proof.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import numpy.linalg as la
from scipy.stats import binom, poisson

from burau_switch import beta_generator, positive_form, unitarize


HERE = Path(__file__).resolve().parent
FIGURE_PATH = HERE / "figures" / "passive_security.png"
RESULTS_PATH = HERE / "results" / "passive_security.json"

RATIO_SWEEP = (0.08, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 20.48, 32.0, 64.0)
REPETITION_SWEEP = (1, 2, 5, 10, 20, 50)
PLOT_REPETITIONS = np.arange(1, 51)
T_MIXER_OMEGA = float(np.pi / 4.0)
T_MIXER_CODES = (3, 0, 3)
T_MIXER_DECLARED_BRANCH_BIAS = 4.539962982671842
PRIMITIVE_LETTERS = ((1, 1), (1, -1), (2, 1), (2, -1))


@dataclass(frozen=True)
class Receiver:
    """Illustrative photon budget for one fresh challenge round."""

    modes: int
    mean_challenge_photons: float
    true_focused_fraction: float
    key_return_efficiency: float
    detector_efficiency: float
    background_clicks: float

    @property
    def security_parameter(self) -> float:
        return self.modes / self.mean_challenge_photons

    @property
    def focused_signal_clicks(self) -> float:
        return (
            self.mean_challenge_photons
            * self.true_focused_fraction
            * self.key_return_efficiency
            * self.detector_efficiency
        )

    @property
    def detectable_return_click_budget(self) -> float:
        """Clicks for a perfectly matched return before finite focus loss."""

        return (
            self.mean_challenge_photons
            * self.key_return_efficiency
            * self.detector_efficiency
        )

    @property
    def honest_clicks(self) -> float:
        return self.focused_signal_clicks + self.background_clicks


DESIGN = Receiver(
    modes=1024,
    mean_challenge_photons=50.0,
    true_focused_fraction=0.60,
    key_return_efficiency=0.25,
    detector_efficiency=0.70,
    background_clicks=0.05,
)

FOUR_MODE_BRIGHT = Receiver(
    modes=4,
    mean_challenge_photons=50.0,
    true_focused_fraction=DESIGN.true_focused_fraction,
    key_return_efficiency=DESIGN.key_return_efficiency,
    detector_efficiency=DESIGN.detector_efficiency,
    background_clicks=DESIGN.background_clicks,
)

FOUR_MODE_WEAK = Receiver(
    modes=4,
    mean_challenge_photons=3.0,
    true_focused_fraction=DESIGN.true_focused_fraction,
    key_return_efficiency=DESIGN.key_return_efficiency,
    detector_efficiency=DESIGN.detector_efficiency,
    background_clicks=DESIGN.background_clicks,
)


def _validate_receiver(receiver: Receiver) -> None:
    assert receiver.modes >= 2
    assert receiver.mean_challenge_photons > 0.0
    assert 0.0 < receiver.true_focused_fraction <= 1.0
    assert 0.0 < receiver.key_return_efficiency <= 1.0
    assert 0.0 < receiver.detector_efficiency <= 1.0
    assert receiver.background_clicks >= 0.0


def _burau_primitive(code: int, omega: float) -> np.ndarray:
    generator, power = PRIMITIVE_LETTERS[code]
    raw = beta_generator(generator, np.exp(0.5j * omega))
    if power == -1:
        raw = la.inv(raw)
    return unitarize(raw, positive_form(omega))


def _fixed_passive_t_mixer() -> dict[str, object]:
    """Verify the three-letter balanced Burau mixer at omega=pi/4."""

    mixer = np.eye(2, dtype=complex)
    for code in T_MIXER_CODES:
        mixer = _burau_primitive(code, T_MIXER_OMEGA) @ mixer

    balanced = np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    left_phases = np.empty(2, dtype=complex)
    right_phases = np.empty(2, dtype=complex)
    right_phases[0] = 1.0
    left_phases[0] = balanced[0, 0] / mixer[0, 0]
    right_phases[1] = balanced[0, 1] / (
        left_phases[0] * mixer[0, 1]
    )
    left_phases[1] = balanced[1, 0] / mixer[1, 0]
    phase_aligned = np.diag(left_phases) @ mixer @ np.diag(right_phases)

    branch_bias = float(
        np.mod(
            2.0 * (np.angle(mixer[0, 0]) - np.angle(mixer[0, 1])),
            2.0 * np.pi,
        )
    )
    first_mixer_column = mixer[:, 0]
    branch_coefficients = mixer * (
        np.asarray([1.0, np.exp(1j * branch_bias)]) * first_mixer_column
    )[None, :]
    sum_difference_target = np.asarray(
        [[0.5, 0.5], [0.5, -0.5]], dtype=complex
    )
    output_phases = np.asarray(
        [
            sum_difference_target[row, 0] / branch_coefficients[row, 0]
            for row in range(2)
        ]
    )
    aligned_coefficients = np.diag(output_phases) @ branch_coefficients

    unitarity_residual = float(la.norm(mixer.conj().T @ mixer - np.eye(2)))
    magnitude_residual = float(
        np.max(np.abs(np.abs(mixer) - 1.0 / np.sqrt(2.0)))
    )
    symmetry_residual = float(la.norm(mixer - mixer.T))
    coupler_equivalence_residual = float(la.norm(phase_aligned - balanced))
    sum_difference_residual = float(
        la.norm(aligned_coefficients - sum_difference_target)
    )

    test_a = np.diag([np.exp(-0.35j), np.exp(0.35j)])
    test_b = np.asarray(
        [
            [np.cos(0.55), -1j * np.sin(0.55)],
            [-1j * np.sin(0.55), np.cos(0.55)],
        ]
    )
    test_state = np.asarray([1.0, 1j]) / np.sqrt(2.0)
    order_ba = test_b @ test_a
    order_ab = test_a @ test_b
    plus_probability = float(
        0.25 * la.norm((order_ba + order_ab) @ test_state) ** 2
    )
    minus_probability = float(
        0.25 * la.norm((order_ba - order_ab) @ test_state) ** 2
    )
    group_commutator = test_a.conj().T @ test_b.conj().T @ test_a @ test_b
    minus_probability_identity = float(
        0.5
        * (
            1.0
            - np.real(
                test_state.conj().T @ group_commutator @ test_state
            )
        )
    )
    commutator_probability_residual = abs(
        minus_probability - minus_probability_identity
    )
    probability_normalization_residual = abs(
        plus_probability + minus_probability - 1.0
    )

    assert unitarity_residual < 2e-14
    assert magnitude_residual < 2e-14
    assert symmetry_residual < 2e-14
    assert coupler_equivalence_residual < 2e-14
    assert abs(branch_bias - T_MIXER_DECLARED_BRANCH_BIAS) < 2e-14
    assert sum_difference_residual < 2e-14
    assert commutator_probability_residual < 2e-14
    assert probability_normalization_residual < 2e-14

    return {
        "omega": T_MIXER_OMEGA,
        "omega_exact": "pi/4",
        "application_order_letter_codes": list(T_MIXER_CODES),
        "application_order_braid_word": "sigma_2^-1 sigma_1 sigma_2^-1",
        "primitive_codebook": {
            "0": "sigma_1",
            "1": "sigma_1^-1",
            "2": "sigma_2",
            "3": "sigma_2^-1",
        },
        "matrix": {
            "real": np.real(mixer).tolist(),
            "imaginary": np.imag(mixer).tolist(),
        },
        "unitarity_residual": unitarity_residual,
        "maximum_entry_magnitude_residual_from_one_over_sqrt_two": (
            magnitude_residual
        ),
        "symmetry_residual": symmetry_residual,
        "phase_equivalence_residual_to_standard_50_50_coupler": (
            coupler_equivalence_residual
        ),
        "branch_bias_radians": branch_bias,
        "sum_difference_coefficient_residual": sum_difference_residual,
        "generic_branch_action": (
            "With an equal path superposition and this fixed branch bias, "
            "the two output ports are, up to fixed port phases, "
            "(X+Y)x/2 and (X-Y)x/2 for arbitrary branch maps X,Y."
        ),
        "controlled_order_specialization": (
            "Setting X=BA and Y=AB makes the difference port [B,A]x/2."
        ),
        "commutator_port_identity": (
            "For unitary A and B and normalized x, P_minus = "
            "||[B,A]x||^2/4 = "
            "[1-Re x^dagger A^dagger B^dagger A B x]/2, and "
            "P_plus + P_minus = 1."
        ),
        "commutator_probability_identity_residual": (
            commutator_probability_residual
        ),
        "plus_minus_probability_normalization_residual": (
            probability_normalization_residual
        ),
        "deployable_branch_topology": (
            "Independent high-dimensional branches X and Y can be fabricated "
            "and enrolled directly. Exact reuse of A and B in opposite orders "
            "is required only for the optional commutator specialization."
        ),
        "baseline": (
            "A conventional 50:50 directional coupler or MZI implements the "
            "same balanced recombination up to port phases and is the required "
            "cost, loss, and stability baseline."
        ),
        "security_scope": (
            "This exact passive identity supplies no secrecy or unclonability "
            "and does not improve the state-estimation bound by itself."
        ),
    }


def _isometry_and_defect_check() -> dict[str, float | int]:
    """Numerically verify the stacked-map and measured-defect inequalities."""

    rng = np.random.default_rng(20260909)
    dimension = 7

    def haar_unitary() -> np.ndarray:
        matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
            size=(dimension, dimension)
        )
        q_matrix, r_matrix = la.qr(matrix)
        diagonal = np.diag(r_matrix)
        phases = diagonal / np.abs(diagonal)
        return q_matrix @ np.diag(np.conj(phases))

    branch_x = haar_unitary()
    branch_y = haar_unitary()
    stacked = 0.5 * np.vstack(
        (branch_x + branch_y, branch_x - branch_y)
    )
    isometry_residual = float(
        la.norm(stacked.conj().T @ stacked - np.eye(dimension), ord=2)
    )

    overlap_residual = 0.0
    for _ in range(128):
        state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        estimate = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        state /= la.norm(state)
        estimate /= la.norm(estimate)
        overlap_residual = max(
            overlap_residual,
            abs(
                np.vdot(stacked @ state, stacked @ estimate)
                - np.vdot(state, estimate)
            ),
        )

    declared_epsilon = 0.03
    eigenvectors = haar_unitary()
    defect_eigenvalues = np.linspace(
        -declared_epsilon, declared_epsilon, dimension
    )
    gram = (
        eigenvectors
        @ np.diag(1.0 + defect_eigenvalues)
        @ eigenvectors.conj().T
    )
    gram = 0.5 * (gram + gram.conj().T)
    gram_eigenvalues, gram_eigenvectors = la.eigh(gram)
    square_root_gram = (
        gram_eigenvectors
        @ np.diag(np.sqrt(gram_eigenvalues))
        @ gram_eigenvectors.conj().T
    )
    measured_map = stacked @ square_root_gram
    measured_epsilon = float(
        la.norm(measured_map.conj().T @ measured_map - np.eye(dimension), ord=2)
    )

    maximum_pointwise_bound_violation = 0.0
    minimum_bound_slack = math.inf
    trials = 512
    for _ in range(trials):
        state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        estimate = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        state /= la.norm(state)
        estimate /= la.norm(estimate)
        response = measured_map @ state
        estimated_response = measured_map @ estimate
        actual = abs(
            np.vdot(response, estimated_response)
            / (la.norm(response) * la.norm(estimated_response))
        ) ** 2
        upper = (
            (abs(np.vdot(state, estimate)) + measured_epsilon)
            / (1.0 - measured_epsilon)
        ) ** 2
        minimum_bound_slack = min(minimum_bound_slack, float(upper - actual))
        maximum_pointwise_bound_violation = max(
            maximum_pointwise_bound_violation, float(actual - upper)
        )

    assert isometry_residual < 2e-14
    assert overlap_residual < 2e-14
    assert abs(measured_epsilon - declared_epsilon) < 2e-14
    assert maximum_pointwise_bound_violation < 2e-14

    return {
        "dimension": dimension,
        "overlap_trials": 128,
        "defect_bound_trials": trials,
        "stacked_map_isometry_residual": isometry_residual,
        "maximum_overlap_preservation_residual": overlap_residual,
        "declared_isometry_defect": declared_epsilon,
        "measured_isometry_defect": measured_epsilon,
        "minimum_sampled_defect_bound_slack": minimum_bound_slack,
        "maximum_sampled_defect_bound_violation": (
            maximum_pointwise_bound_violation
        ),
    }


def coherent_state_factor_bound(modes: float, mean_photons: float) -> float:
    """Jensen bound (nbar + 1) / (nbar + K) for the coherent ensemble.

    The fixed-photon-number result allows arbitrary POVMs.  For weak coherent
    light, this average applies to a privately phase-randomized (hence
    photon-number-diagonal) challenge ensemble.  The model separately assumes
    that the adversary cannot directly implement a loss-matched physical
    realization of the enrolled K-input-mode transfer map H (or an isometric
    dilation of it).
    """

    return (mean_photons + 1.0) / (mean_photons + modes)


def coherent_state_factor(modes: int, mean_photons: float) -> tuple[float, float]:
    """Sum E[(N+1)/(N+K)] for a phase-randomized coherent pulse.

    The returned tail mass bounds the truncation error because the summand is
    in [0, 1].
    """

    probability = math.exp(-mean_photons)
    total_probability = probability
    factor = probability / modes
    photon_number = 0
    while photon_number < 100_000:
        photon_number += 1
        probability *= mean_photons / photon_number
        total_probability += probability
        factor += probability * (photon_number + 1.0) / (photon_number + modes)
        if photon_number > mean_photons and probability < 1e-17:
            break
    else:  # pragma: no cover - guards accidental use far outside this study
        raise RuntimeError("coherent-state sum did not converge")

    tail_mass = max(0.0, 1.0 - total_probability)
    return factor, tail_mass


def click_weighted_coherent_factor(
    modes: int, mean_photons: float
) -> tuple[float, float]:
    """Sum the per-click factor E[(M+2)/(M+K+1)], M~Poisson(nbar).

    Size bias appears when an attacker returns photon number proportional to
    the intercepted Poisson sector.  The returned tail mass bounds numerical
    truncation error because the summand lies in [0, 1].
    """

    probability = math.exp(-mean_photons)
    total_probability = probability
    factor = probability * 2.0 / (modes + 1.0)
    photon_number = 0
    while photon_number < 100_000:
        photon_number += 1
        probability *= mean_photons / photon_number
        total_probability += probability
        factor += probability * (photon_number + 2.0) / (
            photon_number + modes + 1.0
        )
        if photon_number > mean_photons and probability < 1e-17:
            break
    else:  # pragma: no cover
        raise RuntimeError("click-weighted coherent-state sum did not converge")

    tail_mass = max(0.0, 1.0 - total_probability)
    return factor, tail_mass


def click_weighted_factor_bound(modes: float, mean_photons: float) -> float:
    """Jensen upper bound (nbar+2)/(nbar+K+1) on the per-click factor."""

    return (mean_photons + 2.0) / (mean_photons + modes + 1.0)


def goorden_asymptotic_factor(modes: float, mean_photons: float) -> float:
    """Simplified focused-signal ratio 1/(1 + K/nbar)."""

    return mean_photons / (mean_photons + modes)


def quadrature_focused_factor(receiver: Receiver) -> float:
    """Finite-focus form of Skoric et al. Eq. (20), clipped at one.

    This is an asymptotic focused-detector expression for the best equal-split
    quadrature challenge-estimation attack, not a bound on arbitrary attacks.
    It assumes K > nbar and K >> 1.
    """

    numerator = 1.0 + 1.0 / (
        receiver.mean_challenge_photons * receiver.true_focused_fraction
    )
    return min(1.0, numerator / (1.0 + receiver.security_parameter))


def one_sided_projection_factor_bound(projection_upper_bound: float) -> float:
    """Return the adversarial factor used by a one-sided count test.

    Acceptance is monotone in the matched count, so its FAR calculation must
    use the full available upper factor.  Attenuation to the honest mean would
    instead be relevant to a two-sided acceptance window.
    """

    return min(1.0, projection_upper_bound)


def _decision_at_threshold(
    honest_mean: float,
    attacker_mean: float,
    repetitions: int,
    threshold: int,
) -> dict[str, float | int]:
    honest_total = repetitions * honest_mean
    attacker_total = repetitions * attacker_mean
    false_reject = float(poisson.cdf(threshold - 1, honest_total))
    false_accept = float(poisson.sf(threshold - 1, attacker_total))
    return {
        "repetitions": repetitions,
        "threshold_total_clicks": threshold,
        "honest_total_mean": honest_total,
        "attacker_total_mean": attacker_total,
        "false_accept_probability": false_accept,
        "false_reject_probability": false_reject,
        "balanced_error_probability": 0.5 * (false_accept + false_reject),
        "worst_error_probability": max(false_accept, false_reject),
    }


def optimize_total_count_decision(
    receiver: Receiver,
    attacker_absolute_projection_factor: float,
    repetitions: int,
) -> dict[str, float | int]:
    """Choose the integer total-count threshold minimizing worst error."""

    assert 0.0 <= attacker_absolute_projection_factor <= 1.0
    assert repetitions >= 1
    attacker_mean = (
        receiver.background_clicks
        + attacker_absolute_projection_factor
        * receiver.detectable_return_click_budget
    )
    search_limit = math.ceil(
        repetitions * receiver.honest_clicks
        + 12.0 * math.sqrt(repetitions * receiver.honest_clicks + 1.0)
        + 20.0
    )
    candidates = [
        _decision_at_threshold(
            receiver.honest_clicks,
            attacker_mean,
            repetitions,
            threshold,
        )
        for threshold in range(search_limit + 1)
    ]
    return min(
        candidates,
        key=lambda item: (
            item["worst_error_probability"],
            item["balanced_error_probability"],
            abs(
                item["false_accept_probability"]
                - item["false_reject_probability"]
            ),
            item["threshold_total_clicks"],
        ),
    )


def _binary_round_decision_at_threshold(
    honest_mean: float,
    attacker_conditional_mean_bound: float,
    repetitions: int,
    threshold: int,
) -> dict[str, float | int]:
    """Conservative click/no-click decision from a conditional mean bound."""

    honest_click_probability = 1.0 - math.exp(-honest_mean)
    attacker_click_probability_bound = min(
        1.0, attacker_conditional_mean_bound
    )
    false_reject = float(
        binom.cdf(threshold - 1, repetitions, honest_click_probability)
    )
    false_accept_bound = float(
        binom.sf(
            threshold - 1,
            repetitions,
            attacker_click_probability_bound,
        )
    )
    return {
        "repetitions": repetitions,
        "threshold_rounds_with_one_or_more_accepted_clicks": threshold,
        "honest_click_probability": honest_click_probability,
        "attacker_click_probability_upper_bound": (
            attacker_click_probability_bound
        ),
        "false_accept_probability_upper_bound": false_accept_bound,
        "false_reject_probability": false_reject,
        "worst_error_probability_bound": max(
            false_accept_bound, false_reject
        ),
    }


def optimize_binary_round_decision(
    receiver: Receiver,
    attacker_absolute_projection_factor: float,
    repetitions: int,
) -> dict[str, float | int]:
    """Optimize a binarized-round threshold valid beyond Poisson tails.

    If every conditional attacker count mean is bounded by ``m_A``, Markov's
    inequality bounds its conditional click probability by ``min(1,m_A)``.
    Adaptive click totals are then upper-tail dominated by Binomial(R,p_A).
    """

    attacker_mean_bound = (
        receiver.background_clicks
        + attacker_absolute_projection_factor
        * receiver.detectable_return_click_budget
    )
    candidates = [
        _binary_round_decision_at_threshold(
            receiver.honest_clicks,
            attacker_mean_bound,
            repetitions,
            threshold,
        )
        for threshold in range(repetitions + 2)
    ]
    return min(
        candidates,
        key=lambda item: (
            item["worst_error_probability_bound"],
            abs(
                item["false_accept_probability_upper_bound"]
                - item["false_reject_probability"]
            ),
            item["threshold_rounds_with_one_or_more_accepted_clicks"],
        ),
    )


def _factor_record(receiver: Receiver) -> dict[str, float]:
    coherent, tail_mass = coherent_state_factor(
        receiver.modes, receiver.mean_challenge_photons
    )
    click_weighted, click_tail_mass = click_weighted_coherent_factor(
        receiver.modes, receiver.mean_challenge_photons
    )
    click_bound = click_weighted_factor_bound(
        receiver.modes, receiver.mean_challenge_photons
    )
    coherent_bound = coherent_state_factor_bound(
        receiver.modes, receiver.mean_challenge_photons
    )
    return {
        "phase_randomized_coherent_state_jensen_bound": coherent_bound,
        "phase_randomized_coherent_state_exact": coherent,
        "phase_randomized_coherent_sum_omitted_tail_mass": tail_mass,
        "sector_proportional_resend_click_weighted_exact": click_weighted,
        "sector_proportional_resend_click_weighted_jensen_bound": click_bound,
        "sector_proportional_resend_sum_omitted_tail_mass": click_tail_mass,
        "main_bound_to_honest_signal_ratio": (
            coherent_bound / receiver.true_focused_fraction
        ),
        "goorden_simplified_factor": goorden_asymptotic_factor(
            receiver.modes, receiver.mean_challenge_photons
        ),
        "quadrature_finite_focus_factor": quadrature_focused_factor(receiver),
    }


def _decision_series(
    receiver: Receiver,
    attacker_factor: float,
    repetitions: Iterable[int],
) -> list[dict[str, float | int]]:
    return [
        optimize_total_count_decision(receiver, attacker_factor, repetition)
        for repetition in repetitions
    ]


def _sweep_record() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ratio in RATIO_SWEEP:
        modes = int(round(DESIGN.mean_challenge_photons * ratio))
        receiver = Receiver(
            modes=max(2, modes),
            mean_challenge_photons=DESIGN.mean_challenge_photons,
            true_focused_fraction=DESIGN.true_focused_fraction,
            key_return_efficiency=DESIGN.key_return_efficiency,
            detector_efficiency=DESIGN.detector_efficiency,
            background_clicks=DESIGN.background_clicks,
        )
        actual_ratio = receiver.security_parameter
        factor = coherent_state_factor_bound(
            receiver.modes, receiver.mean_challenge_photons
        )
        rows.append(
            {
                "modes": receiver.modes,
                "K_over_n": actual_ratio,
                "inside_K_greater_than_n_regime": receiver.modes
                > receiver.mean_challenge_photons,
                "conditional_energy_absolute_projection_bound": factor,
                "one_sided_projection_factor_bound": (
                    one_sided_projection_factor_bound(factor)
                ),
                "decisions": _decision_series(
                    receiver,
                    one_sided_projection_factor_bound(factor),
                    REPETITION_SWEEP,
                ),
            }
        )
    return rows


def _make_figure(
    ratio_sweep: list[dict[str, object]],
    design_decisions: list[dict[str, float | int]],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)

    ratio_axis = axes[0, 0]
    ratios = np.logspace(-1.3, 2.0, 400)
    state_bound = (DESIGN.mean_challenge_photons + 1.0) / (
        DESIGN.mean_challenge_photons * (1.0 + ratios)
    )
    click_bound = (DESIGN.mean_challenge_photons + 2.0) / (
        DESIGN.mean_challenge_photons * (1.0 + ratios) + 1.0
    )
    ratio_axis.loglog(
        ratios,
        state_bound,
        label=r"main conditional-energy bound $(\bar n+1)/(\bar n+K)$",
    )
    ratio_axis.loglog(
        ratios,
        click_bound,
        "--",
        label=r"sector-proportional resend diagnostic $(\bar n+2)/(\bar n+K+1)$",
    )
    ratio_axis.axvspan(ratios.min(), 1.0, color="#d9d9d9", alpha=0.55)
    ratio_axis.axvline(DESIGN.security_parameter, color="#087e8b", linewidth=1.2)
    ratio_axis.scatter(
        [FOUR_MODE_BRIGHT.security_parameter, DESIGN.security_parameter],
        [
            coherent_state_factor_bound(
                FOUR_MODE_BRIGHT.modes,
                FOUR_MODE_BRIGHT.mean_challenge_photons,
            ),
            coherent_state_factor_bound(
                DESIGN.modes, DESIGN.mean_challenge_photons
            ),
        ],
        color=["#c43b3b", "#087e8b"],
        zorder=5,
    )
    ratio_axis.set(
        xlabel=r"security parameter $S=K/\bar n$",
        ylabel="absolute matched-mode probability bound",
        title="A. State-estimation separation",
        ylim=(8e-3, 1.2),
    )
    ratio_axis.text(
        0.065, 0.014, "bare 4-mode\n" + r"$\bar n=50$", color="#a52a2a"
    )
    ratio_axis.text(22.5, 0.075, "hybrid\n$K=1024$", color="#08616a")
    ratio_axis.legend(fontsize=8)

    heat_axis = axes[0, 1]
    x_values = np.asarray([row["K_over_n"] for row in ratio_sweep])
    heat_values = np.asarray(
        [
            [
                max(float(decision["worst_error_probability"]), 1e-16)
                for decision in row["decisions"]
            ]
            for row in ratio_sweep
        ]
    ).T
    log_x = np.log10(x_values)
    log_x_edges = np.concatenate(
        (
            [log_x[0] - 0.5 * (log_x[1] - log_x[0])],
            0.5 * (log_x[:-1] + log_x[1:]),
            [log_x[-1] + 0.5 * (log_x[-1] - log_x[-2])],
        )
    )
    repetition_edges = np.arange(len(REPETITION_SWEEP) + 1) - 0.5
    image = heat_axis.pcolormesh(
        log_x_edges,
        repetition_edges,
        np.log10(heat_values),
        cmap="viridis_r",
        vmin=-12,
        vmax=0,
        shading="flat",
    )
    heat_axis.axvline(0.0, color="white", linestyle="--", linewidth=1.0)
    heat_axis.set_xticks(np.log10([0.1, 1.0, 10.0, 64.0]))
    heat_axis.set_xticklabels(["0.1", "1", "10", "64"])
    heat_axis.set_yticks(range(len(REPETITION_SWEEP)))
    heat_axis.set_yticklabels(REPETITION_SWEEP)
    heat_axis.set(
        xlabel=r"$K/\bar n$ (log scale; left of dashed line outside $K>\bar n$)",
        ylabel="fresh challenge rounds",
        title="B. Exact Poisson worst decision error",
    )
    colorbar = figure.colorbar(image, ax=heat_axis, pad=0.01)
    colorbar.set_label(r"$\log_{10}\max(\mathrm{FAR},\mathrm{FRR})$")

    pmf_axis = axes[1, 0]
    design_factor = coherent_state_factor_bound(
        DESIGN.modes, DESIGN.mean_challenge_photons
    )
    design_factor = one_sided_projection_factor_bound(design_factor)
    attacker_mean = (
        DESIGN.background_clicks
        + design_factor * DESIGN.detectable_return_click_budget
    )
    threshold = int(design_decisions[0]["threshold_total_clicks"])
    click_values = np.arange(0, 16)
    pmf_axis.stem(
        click_values - 0.08,
        poisson.pmf(click_values, DESIGN.honest_clicks),
        linefmt="#087e8b",
        markerfmt="o",
        basefmt=" ",
        label=f"true key, mean {DESIGN.honest_clicks:.2f}",
    )
    pmf_axis.stem(
        click_values + 0.08,
        poisson.pmf(click_values, attacker_mean),
        linefmt="#c43b3b",
        markerfmt="s",
        basefmt=" ",
        label=f"emulator bound, mean {attacker_mean:.2f}",
    )
    pmf_axis.axvline(threshold - 0.5, color="black", linestyle="--")
    pmf_axis.set(
        xlabel="focused-detector clicks in one round",
        ylabel="probability",
        title=rf"C. Illustrative hybrid point; accept at $X\geq {threshold}$",
        xlim=(-0.5, 13.5),
    )
    pmf_axis.legend(fontsize=8)

    repetition_axis = axes[1, 1]
    design_plot = _decision_series(DESIGN, design_factor, PLOT_REPETITIONS)
    four_factor = coherent_state_factor_bound(
        FOUR_MODE_BRIGHT.modes, FOUR_MODE_BRIGHT.mean_challenge_photons
    )
    four_factor = one_sided_projection_factor_bound(four_factor)
    four_plot = _decision_series(
        FOUR_MODE_BRIGHT, four_factor, PLOT_REPETITIONS
    )
    repetition_axis.semilogy(
        PLOT_REPETITIONS,
        [item["worst_error_probability"] for item in design_plot],
        label="hybrid $K=1024$ receiver model",
        color="#087e8b",
    )
    repetition_axis.semilogy(
        PLOT_REPETITIONS,
        [item["worst_error_probability"] for item in four_plot],
        label="4-mode state-estimation extrapolation",
        color="#d98b2b",
    )
    direct_emulator_plot = _decision_series(
        FOUR_MODE_BRIGHT,
        FOUR_MODE_BRIGHT.true_focused_fraction,
        PLOT_REPETITIONS,
    )
    repetition_axis.plot(
        PLOT_REPETITIONS,
        [item["worst_error_probability"] for item in direct_emulator_plot],
        "--",
        label="4-mode direct emulator (deterministic-threshold minimax)",
        color="#c43b3b",
    )
    repetition_axis.set(
        xlabel="fresh challenge rounds",
        ylabel=r"optimized $\max(\mathrm{FAR},\mathrm{FRR})$",
        title="D. Repetition helps only when distributions differ",
        ylim=(1e-14, 1.0),
    )
    repetition_axis.legend(fontsize=8)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def run_passive_security_study() -> dict[str, object]:
    """Run the checked receiver study, write artifacts, and return its record."""
    for receiver in (DESIGN, FOUR_MODE_BRIGHT, FOUR_MODE_WEAK):
        _validate_receiver(receiver)

    design_factors = _factor_record(DESIGN)
    four_bright_factors = _factor_record(FOUR_MODE_BRIGHT)
    four_weak_factors = _factor_record(FOUR_MODE_WEAK)

    for factors in (design_factors, four_bright_factors, four_weak_factors):
        assert factors["phase_randomized_coherent_state_exact"] <= (
            factors["phase_randomized_coherent_state_jensen_bound"] + 1e-12
        )
        assert factors[
            "sector_proportional_resend_click_weighted_exact"
        ] <= (
            factors[
                "sector_proportional_resend_click_weighted_jensen_bound"
            ]
            + 1e-12
        )
        assert factors["goorden_simplified_factor"] <= (
            factors["phase_randomized_coherent_state_jensen_bound"] + 1e-12
        )
        assert (
            factors["phase_randomized_coherent_sum_omitted_tail_mass"]
            < 1e-12
        )
        assert (
            factors["sector_proportional_resend_sum_omitted_tail_mass"]
            < 1e-12
        )

    main_factor = one_sided_projection_factor_bound(
        design_factors["phase_randomized_coherent_state_jensen_bound"]
    )
    design_decisions = _decision_series(
        DESIGN, main_factor, REPETITION_SWEEP
    )
    design_binary_decisions = [
        optimize_binary_round_decision(DESIGN, main_factor, repetition)
        for repetition in REPETITION_SWEEP
    ]
    four_bright_decisions = _decision_series(
        FOUR_MODE_BRIGHT,
        one_sided_projection_factor_bound(
            four_bright_factors[
                "phase_randomized_coherent_state_jensen_bound"
            ]
        ),
        REPETITION_SWEEP,
    )
    four_weak_decisions = _decision_series(
        FOUR_MODE_WEAK,
        one_sided_projection_factor_bound(
            four_weak_factors[
                "phase_randomized_coherent_state_jensen_bound"
            ]
        ),
        REPETITION_SWEEP,
    )
    ratio_sweep = _sweep_record()
    passive_t_mixer = _fixed_passive_t_mixer()
    map_checks = _isometry_and_defect_check()

    one_signal_click_photons = 1.0 / (
        FOUR_MODE_BRIGHT.true_focused_fraction
        * FOUR_MODE_BRIGHT.key_return_efficiency
        * FOUR_MODE_BRIGHT.detector_efficiency
    )
    direct_emulator = {
        "attacker_signal_factor": 1.0,
        "total_variation_distance_from_honest": 0.0,
        "equal_prior_bayes_error": 0.5,
        "explanation": (
            "A direct emulator of the public four-mode transfer map produces "
            "the honest Poisson distribution; every deterministic threshold "
            "has (FAR + FRR)/2 = 1/2."
        ),
    }

    assert DESIGN.modes > DESIGN.mean_challenge_photons
    assert DESIGN.security_parameter == 20.48
    assert main_factor < 0.05
    assert float(design_decisions[2]["worst_error_probability"]) < 1e-3
    assert float(design_decisions[3]["worst_error_probability"]) < 1e-6
    assert float(
        design_binary_decisions[5]["worst_error_probability_bound"]
    ) < 1e-8
    assert FOUR_MODE_BRIGHT.modes < FOUR_MODE_BRIGHT.mean_challenge_photons
    assert one_signal_click_photons > FOUR_MODE_BRIGHT.modes
    assert direct_emulator["equal_prior_bayes_error"] == 0.5

    results = {
        "study": "passive high-dimensional optical-key receiver model",
        "status": "deterministic analytical study; no hardware demonstration",
        "model": {
            "honest_per_round_click_mean": (
                "background + nbar * focused_fraction * return_efficiency "
                "* detector_efficiency"
            ),
            "attacker_per_round_click_mean": (
                "background + absolute_projection_factor * Lambda * "
                "detector_efficiency, with illustrative "
                "Lambda = return_efficiency * nbar"
            ),
            "decision": (
                "accept when the sum of focused-detector clicks over fresh "
                "rounds reaches an integer threshold chosen to minimize the "
                "larger of exact Poisson FAR and FRR"
            ),
            "main_attacker_factor": "(nbar + 1) / (nbar + K)",
            "main_bound_scope": (
                "Jensen bound for arbitrary-POVM estimation of private, "
                "uniformly phase-randomized, freshly sampled Haar "
                "K-dimensional coherent states. Every refined attacker "
                "outcome is assumed to obey the accepted-response conditional "
                "energy cap. The model excludes a fast, sufficiently low-loss "
                "physical realization of the enrolled K-input-mode transfer "
                "map H directly (or an isometric dilation thereof)."
            ),
            "poisson_tail_scope": (
                "exact for the stipulated independent Poisson click model; "
                "a mean fidelity bound alone does not prove Poisson attacker "
                "tails against correlated or adaptive strategies"
            ),
            "adaptive_binary_tail_scope": (
                "if every fresh round retains the conditional attacker mean "
                "bound, Markov bounds its click probability and a binomial "
                "upper tail remains valid for adaptive histories"
            ),
            "assumptions": [
                "trusted, calibrated challenge preparation and analyzer",
                "tamper-resistant verifier/reader and authenticated decision path",
                "fresh independent uniformly random K-mode challenge each round",
                "weak coherent pulses with calibrated mean photon number and privately randomized global phase",
                "full complex transfer operator enrolled; no finite reusable challenge list",
                "stable passive key and independent stationary Poisson background",
                "an accepted-response energy gate certifies, for every refined attacker outcome, a conditional mean return no larger than the calibrated authentic return budget over every spatial/modal, spectral, polarization, and timing degree of freedom capable of causing an accepted detector event",
                "attacker performs challenge estimation and digital emulation",
                "attacker cannot substitute a fast, sufficiently low-loss physical realization of H or an isometric dilation",
            ],
            "not_covered": [
                "physical cloning or substitution of the scattering key",
                "phishing, relay, replay, database poisoning, or verifier compromise",
                "side channels, Trojan-horse light, detector blinding, or denial of service",
                "full count-valued adaptive or correlated tails outside the Poisson surrogate; only the reported binary-round bound covers adaptive histories",
                "imperfect global-phase randomization or a finite challenge set",
                "evasion or miscalibration of the conditional accepted-response energy cap",
            ],
        },
        "t_chip_role": {
            "security_statement": (
                "The public deterministic four-mode Burau/T-chip transform is "
                "not the secret and provides no unclonability by itself."
            ),
            "positive_role": (
                "The exact mixer exposes sum and difference ports of two "
                "independently fabricated high-dimensional passive branch maps "
                "X and Y; X=BA and Y=AB is an optional commutator reference. "
                "No advantage over a generic balanced coupler has been demonstrated."
            ),
            "dimensionality_warning": (
                "A fixed expansion of four controllable amplitudes into many "
                "pixels still has challenge dimension at most four. The "
                "K=1024 point requires 1024 independently controlled challenge "
                "modes; the T-chip does not manufacture those dimensions."
            ),
            "verified_fixed_passive_mixer": passive_t_mixer,
            "verified_stacked_map_and_defect_bound": map_checks,
        },
        "illustrative_hybrid_design": {
            "architecture": (
                "trusted phase-randomized K-mode weak-pulse shaper -> exact "
                "passive T-mixer -> sealed high-dimensional branch maps X,Y -> "
                "sum/difference ports -> computed matched analyzer -> power tap "
                "and photon counter"
            ),
            "parameters": asdict(DESIGN),
            "derived": {
                "K_over_n": DESIGN.security_parameter,
                "detectable_return_click_budget_before_focus": (
                    DESIGN.detectable_return_click_budget
                ),
                "conditional_return_photon_cap_at_analyzer_input": (
                    DESIGN.key_return_efficiency
                    * DESIGN.mean_challenge_photons
                ),
                "conditional_energy_cap_multiplier_of_nominal_authentic_return": 1.0,
                "focused_signal_clicks_per_round": DESIGN.focused_signal_clicks,
                "honest_click_mean_per_round": DESIGN.honest_clicks,
                "attacker_click_mean_per_round_main_bound": (
                    DESIGN.background_clicks
                    + main_factor * DESIGN.detectable_return_click_budget
                ),
                "factors": design_factors,
            },
            "optimized_total_count_decisions": design_decisions,
            "adaptive_safe_binary_round_decisions": design_binary_decisions,
        },
        "four_mode_t_chip_diagnostic": {
            "bright_usable_count_case": {
                "parameters": asdict(FOUR_MODE_BRIGHT),
                "derived": {
                    "K_over_n": FOUR_MODE_BRIGHT.security_parameter,
                    "inside_n_less_than_K_theorem_setup": False,
                    "factors_for_diagnostic_only": four_bright_factors,
                    "minimum_nbar_for_one_honest_signal_click_per_round": (
                        one_signal_click_photons
                    ),
                },
                "state_estimation_only_extrapolation": four_bright_decisions,
                "direct_public_transfer_emulator": direct_emulator,
            },
            "n_less_than_K_case": {
                "parameters": asdict(FOUR_MODE_WEAK),
                "derived": {
                    "K_over_n": FOUR_MODE_WEAK.security_parameter,
                    "focused_signal_clicks_per_round": (
                        FOUR_MODE_WEAK.focused_signal_clicks
                    ),
                    "factors": four_weak_factors,
                },
                "state_estimation_only_decisions": four_weak_decisions,
                "fatal_scope_failure": (
                    "Because the enrolled transform is public and only 4x4, "
                    "the no-fast-lossless-emulator premise is not credible; a "
                    "direct emulator has factor one regardless of nbar."
                ),
            },
        },
        "K_over_n_and_repetition_sweep": ratio_sweep,
        "interpretation": [
            "The positive target is possession authentication or anti-counterfeit readout of a passive disorder-bearing token.",
            "At the illustrative K=1024, nbar=50 point, the receiver model predicts strong count separation after a small number of fresh challenges.",
            "This calculation is a design gate for an experiment, not evidence that a fabricated token is unclonable or that a protocol is secure.",
            "The bare T-chip remains useful as a braid/control experiment, but it is not the physical key.",
        ],
    }

    _make_figure(ratio_sweep, design_decisions)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"wrote {RESULTS_PATH.relative_to(HERE.parent.parent)}")
    print(f"wrote {FIGURE_PATH.relative_to(HERE.parent.parent)}")
    print(
        "hybrid point: "
        f"S={DESIGN.security_parameter:.2f}, "
        f"honest mean={DESIGN.honest_clicks:.4f}, "
        "attacker mean="
        f"{DESIGN.background_clicks + main_factor * DESIGN.detectable_return_click_budget:.4f}"
    )
    for decision in design_decisions:
        print(
            f"R={decision['repetitions']:>2}: "
            f"T={decision['threshold_total_clicks']:>3}, "
            f"FAR={decision['false_accept_probability']:.3e}, "
            f"FRR={decision['false_reject_probability']:.3e}"
        )

    return results


def main() -> None:
    run_passive_security_study()


if __name__ == "__main__":
    main()
