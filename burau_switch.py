"""Numerical checks for the Burau--Squier controlled-order Gedankenexperiment.

The script verifies the two definite Squier intervals, the exact braid-order
Helstrom witness, and the secondary response of the braid-dressed switch.
It intentionally makes no causal-nonseparability claim from a closed-unitary
discrimination score.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la


TWO_PI = 2.0 * np.pi
INNER_LEFT = 2.0 * np.pi / 3.0
INNER_RIGHT = 4.0 * np.pi / 3.0


def squier_form(omega: float) -> np.ndarray:
    """The specialized 2-by-2 Squier form J(omega)."""
    diagonal = 2.0 * np.cos(omega / 2.0)
    return np.array([[diagonal, -1.0], [-1.0, diagonal]], dtype=complex)


def beta_generator(index: int, s: complex) -> np.ndarray:
    """Squier's modified reduced Burau generators for B_3."""
    if index == 1:
        return np.array([[-s**2, s], [0.0, 1.0]], dtype=complex)
    if index == 2:
        return np.array([[1.0, 0.0], [s, -s**2]], dtype=complex)
    raise ValueError("index must be 1 or 2")


def beta_word(word: list[tuple[int, int]], s: complex) -> np.ndarray:
    """Evaluate a word [(generator, integer power), ...] from left to right."""
    result = np.eye(2, dtype=complex)
    for index, power in word:
        generator = beta_generator(index, s)
        if power < 0:
            generator = la.inv(generator)
        for _ in range(abs(power)):
            result = result @ generator
    return result


def definiteness_sign(omega: float) -> int | None:
    """Return epsilon with H=epsilon*J positive, or None outside Omega_+."""
    if 0.0 < omega < INNER_LEFT:
        return 1
    if INNER_RIGHT < omega < TWO_PI:
        return -1
    return None


def positive_form(omega: float) -> np.ndarray:
    """The sign-normalized positive form H on Omega_+."""
    sign = definiteness_sign(omega)
    if sign is None:
        raise ValueError("omega is outside the Squier definiteness region")
    return sign * squier_form(omega)


def positive_square_root(matrix: np.ndarray, tolerance: float = 1e-13) -> np.ndarray:
    """Return the positive Hermitian square root of a positive matrix."""
    eigenvalues, eigenvectors = la.eigh(matrix)
    if np.min(eigenvalues) <= tolerance:
        raise ValueError("matrix is not numerically positive definite")
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.conj().T


def unitarize(matrix: np.ndarray, form: np.ndarray) -> np.ndarray:
    """Conjugate an H-unitary matrix into an ordinary Euclidean unitary."""
    root = positive_square_root(form)
    return root @ matrix @ la.inv(root)


def helstrom_unitary_success(
    first: np.ndarray, second: np.ndarray, tolerance: float = 1e-12
) -> float:
    """Optimal equal-prior, single-use discrimination of two unitary channels."""
    relative = first.conj().T @ second
    angles = np.sort(np.mod(np.angle(la.eigvals(relative)), TWO_PI))
    gaps = np.concatenate((np.diff(angles), [TWO_PI - angles[-1] + angles[0]]))
    covering_arc = float(np.clip(TWO_PI - np.max(gaps), 0.0, TWO_PI))
    capped_arc = min(covering_arc, np.pi)
    if capped_arc > np.pi - tolerance:
        return 1.0
    return 0.5 * (1.0 + np.sin(capped_arc / 2.0))


def exact_order_witness(omega: float) -> float:
    """Closed form for p*(U1 U2, U2 U1) - 1/2."""
    radicand = 1.0 - (0.5 - np.cos(omega)) ** 2
    return 0.5 * np.sqrt(max(0.0, radicand))


def rx(angle: float) -> np.ndarray:
    cosine = np.cos(angle / 2.0)
    sine = -1j * np.sin(angle / 2.0)
    return np.array([[cosine, sine], [sine, cosine]], dtype=complex)


def rz(angle: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def switch_unitary(first: np.ndarray, second: np.ndarray, phase: float) -> np.ndarray:
    """Closed switch S=|0><0| tensor BA + exp(i phase)|1><1| tensor AB."""
    projector_zero = np.diag([1.0, 0.0])
    projector_one = np.diag([0.0, 1.0])
    return np.kron(projector_zero, second @ first) + np.exp(
        1j * phase
    ) * np.kron(projector_one, first @ second)


def fixed_order_simulator(
    first: np.ndarray, second: np.ndarray, phase: float
) -> np.ndarray:
    """Fixed target order AB with the same observable control-path phase."""
    control_phase = np.diag([1.0, np.exp(1j * phase)])
    return np.kron(control_phase, first @ second)


def _format_phase_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0.0, TWO_PI)
    axis.set_xticks([0.0, INNER_LEFT, INNER_RIGHT, TWO_PI])
    axis.set_xticklabels(
        [r"$0$", r"$2\pi/3$", r"$4\pi/3$", r"$2\pi$"]
    )
    axis.set_xlabel(r"$\omega$")
    axis.axvspan(0.0, INNER_LEFT, color="#2ca02c", alpha=0.07)
    axis.axvspan(INNER_RIGHT, TWO_PI, color="#2ca02c", alpha=0.07)
    for boundary in (INNER_LEFT, INNER_RIGHT):
        axis.axvline(boundary, color="0.55", linestyle=":", linewidth=0.9)


def run_verification(
    *,
    grid_points: int = 5001,
    boundary_margin: float = 1e-4,
    save_figure: bool = True,
    show_figure: bool = True,
    figure_path: str | Path = "figures/witness_gap_summary.png",
) -> dict[str, float]:
    """Run all checks, plot the results, and return numerical diagnostics."""
    omegas = np.linspace(boundary_margin, TWO_PI - boundary_margin, grid_points)
    valid = np.array([definiteness_sign(value) is not None for value in omegas])

    lambda_minus = 2.0 * np.cos(omegas / 2.0) - 1.0
    lambda_plus = 2.0 * np.cos(omegas / 2.0) + 1.0
    form_error = np.full(grid_points, np.nan)
    braid_error = np.full(grid_points, np.nan)
    unitary_error = np.full(grid_points, np.nan)
    numerical_witness = np.full(grid_points, np.nan)
    closed_witness = np.full(grid_points, np.nan)
    switch_contrast = np.full(grid_points, np.nan)
    abelian_phase_contrast = np.full(grid_points, np.nan)
    fixed_simulator_error = np.full(grid_points, np.nan)
    commutator_trace_error = np.full(grid_points, np.nan)
    commutator_determinant_error = np.full(grid_points, np.nan)
    minimum_h_eigenvalue = np.inf

    first_target = rx(1.5)
    second_target = rz(0.75)
    identity_two = np.eye(2, dtype=complex)
    identity_four = np.eye(4, dtype=complex)
    abelian_mixer_phase = -0.2
    abelian_mixer = np.diag(
        [
            np.exp(-0.5j * abelian_mixer_phase),
            np.exp(0.5j * abelian_mixer_phase),
        ]
    )

    for position, omega in enumerate(omegas):
        if not valid[position]:
            continue

        s = np.exp(0.5j * omega)
        form = positive_form(float(omega))
        minimum_h_eigenvalue = min(
            minimum_h_eigenvalue, float(np.min(la.eigvalsh(form)))
        )
        beta_one = beta_generator(1, s)
        beta_two = beta_generator(2, s)
        form_error[position] = max(
            la.norm(beta_one.conj().T @ form @ beta_one - form),
            la.norm(beta_two.conj().T @ form @ beta_two - form),
        )

        unitary_one = unitarize(beta_one, form)
        unitary_two = unitarize(beta_two, form)
        word_unitary = unitarize(beta_word([(1, 3)], s), form)
        unitary_error[position] = max(
            la.norm(unitary_one.conj().T @ unitary_one - identity_two),
            la.norm(unitary_two.conj().T @ unitary_two - identity_two),
            la.norm(word_unitary.conj().T @ word_unitary - identity_two),
        )
        braid_error[position] = la.norm(
            unitary_one @ unitary_two @ unitary_one
            - unitary_two @ unitary_one @ unitary_two
        )

        order_twelve = unitary_one @ unitary_two
        order_twenty_one = unitary_two @ unitary_one
        group_commutator = order_twelve.conj().T @ order_twenty_one
        commutator_trace_error[position] = abs(
            np.trace(group_commutator) - (1.0 - 2.0 * np.cos(omega))
        )
        commutator_determinant_error[position] = abs(
            la.det(group_commutator) - 1.0
        )
        numerical_witness[position] = (
            helstrom_unitary_success(order_twelve, order_twenty_one) - 0.5
        )
        closed_witness[position] = exact_order_witness(float(omega))

        bare_switch = switch_unitary(first_target, second_target, float(omega))
        dressed_switch = (
            np.kron(word_unitary, identity_two)
            @ bare_switch
            @ np.kron(word_unitary, identity_two)
        )
        switch_contrast[position] = helstrom_unitary_success(
            identity_four, dressed_switch
        ) - helstrom_unitary_success(identity_four, bare_switch)
        abelian_dressed_switch = (
            np.kron(abelian_mixer, identity_two)
            @ bare_switch
            @ np.kron(abelian_mixer, identity_two)
        )
        abelian_phase_contrast[position] = helstrom_unitary_success(
            identity_four, abelian_dressed_switch
        ) - helstrom_unitary_success(identity_four, bare_switch)

        fixed = fixed_order_simulator(first_target, second_target, float(omega))
        fixed_simulator_error[position] = abs(
            helstrom_unitary_success(identity_four, bare_switch)
            - helstrom_unitary_success(identity_four, fixed)
        )

    diagnostics = {
        "det_J_at_0": float(la.det(squier_form(0.0)).real),
        "det_J_at_2pi_over_3": float(la.det(squier_form(INNER_LEFT)).real),
        "det_J_at_4pi_over_3": float(la.det(squier_form(INNER_RIGHT)).real),
        "det_J_at_2pi": float(la.det(squier_form(TWO_PI)).real),
        "max_form_error": float(np.nanmax(form_error)),
        "max_braid_error": float(np.nanmax(braid_error)),
        "max_unitary_error": float(np.nanmax(unitary_error)),
        "max_closed_form_error": float(
            np.nanmax(np.abs(numerical_witness - closed_witness))
        ),
        "max_commutator_trace_error": float(np.nanmax(commutator_trace_error)),
        "max_commutator_determinant_error": float(
            np.nanmax(commutator_determinant_error)
        ),
        "max_fixed_simulator_error": float(np.nanmax(fixed_simulator_error)),
        "minimum_sampled_H_eigenvalue": float(minimum_h_eigenvalue),
        "minimum_switch_contrast": float(np.nanmin(switch_contrast)),
        "maximum_switch_contrast": float(np.nanmax(switch_contrast)),
        "minimum_abelian_phase_contrast": float(
            np.nanmin(abelian_phase_contrast)
        ),
        "maximum_abelian_phase_contrast": float(
            np.nanmax(abelian_phase_contrast)
        ),
        "maximum_order_witness": float(np.nanmax(numerical_witness)),
        "order_witness_at_pi_over_3": exact_order_witness(np.pi / 3.0),
        "order_witness_at_5pi_over_3": exact_order_witness(5.0 * np.pi / 3.0),
        "identity_counterexample": helstrom_unitary_success(
            identity_four, switch_unitary(identity_two, identity_two, np.pi / 2.0)
        ),
    }

    assert diagnostics["det_J_at_0"] > 0.0
    assert diagnostics["det_J_at_2pi"] > 0.0
    assert abs(diagnostics["det_J_at_2pi_over_3"]) < 1e-12
    assert abs(diagnostics["det_J_at_4pi_over_3"]) < 1e-12
    assert diagnostics["minimum_sampled_H_eigenvalue"] > 0.0
    assert diagnostics["max_form_error"] < 1e-10
    assert diagnostics["max_braid_error"] < 1e-10
    assert diagnostics["max_unitary_error"] < 1e-9
    assert diagnostics["max_closed_form_error"] < 1e-10
    assert diagnostics["max_commutator_trace_error"] < 1e-10
    assert diagnostics["max_commutator_determinant_error"] < 1e-10
    assert diagnostics["max_fixed_simulator_error"] < 1e-10
    assert diagnostics["minimum_switch_contrast"] < 0.0
    assert diagnostics["maximum_switch_contrast"] > 0.0
    assert diagnostics["minimum_abelian_phase_contrast"] < 0.0
    assert diagnostics["maximum_abelian_phase_contrast"] > 0.0
    assert abs(diagnostics["order_witness_at_pi_over_3"] - 0.5) < 1e-12
    assert abs(diagnostics["order_witness_at_5pi_over_3"] - 0.5) < 1e-12
    assert abs(
        diagnostics["identity_counterexample"] - (2.0 + np.sqrt(2.0)) / 4.0
    ) < 1e-12

    figure, axes_grid = plt.subplots(2, 2, figsize=(11.5, 7.2))
    axes = axes_grid.ravel()

    axes[0].plot(omegas, lambda_minus, label=r"$2\cos(\omega/2)-1$")
    axes[0].plot(omegas, lambda_plus, label=r"$2\cos(\omega/2)+1$")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("eigenvalue")
    axes[0].set_title(r"(a) Signature of $J(\omega)$")
    axes[0].legend(fontsize=8)

    error_floor = 1e-16
    axes[1].plot(
        omegas, np.maximum(form_error, error_floor), label=r"$H$ invariance"
    )
    axes[1].plot(
        omegas, np.maximum(braid_error, error_floor), label="braid relation"
    )
    axes[1].plot(
        omegas, np.maximum(unitary_error, error_floor), label="unitarity"
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-16, 1e-8)
    axes[1].set_title("(b) Numerical residuals")
    axes[1].legend(fontsize=8)

    axes[2].plot(omegas, numerical_witness, linewidth=2.2, label="numerical")
    axes[2].plot(
        omegas,
        closed_witness,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="closed form",
    )
    axes[2].set_ylim(bottom=0.0)
    axes[2].set_ylabel(r"$\mathcal{W}_{\rm NA}$")
    axes[2].set_title("(c) Two-generator order witness")
    axes[2].legend(fontsize=8)

    axes[3].plot(
        omegas,
        switch_contrast,
        color="#7f3c8d",
        linewidth=2.0,
        label=r"braid word $\sigma_1^3$",
    )
    axes[3].plot(
        omegas,
        abelian_phase_contrast,
        color="#d95f02",
        linestyle="--",
        linewidth=1.4,
        label=r"phase only, $\phi=-0.2$",
    )
    axes[3].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[3].set_ylabel(r"$p_{\rm test}-p_{\rm switch}$")
    axes[3].set_title("(d) Switch-response contrasts")
    axes[3].legend(fontsize=8)

    for axis in axes:
        _format_phase_axis(axis)
        axis.grid(alpha=0.18)

    figure.tight_layout()
    if save_figure:
        output_path = Path(figure_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=240, bbox_inches="tight")
    if show_figure:
        plt.show()
    else:
        plt.close(figure)

    return diagnostics


if __name__ == "__main__":
    results = run_verification()
    for name, value in results.items():
        print(f"{name}: {value:.12g}")
