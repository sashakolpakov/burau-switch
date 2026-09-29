"""Numerical checks for the Burau--Squier braid-order experiment.

The two elementary unitarized Burau generators occupy the operation slots of
a controlled-order circuit.  The two branches apply U1 U2 and U2 U1.
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
    """Return the specialized 2-by-2 Squier form J(omega)."""
    diagonal = 2.0 * np.cos(omega / 2.0)
    return np.array([[diagonal, -1.0], [-1.0, diagonal]], dtype=complex)


def beta_generator(index: int, s: complex) -> np.ndarray:
    """Return Squier's modified reduced Burau generator beta_i."""
    if index == 1:
        return np.array([[-s**2, s], [0.0, 1.0]], dtype=complex)
    if index == 2:
        return np.array([[1.0, 0.0], [s, -s**2]], dtype=complex)
    raise ValueError("index must be 1 or 2")


def beta_word(word: list[tuple[int, int]], s: complex) -> np.ndarray:
    """Evaluate a braid word [(generator, integer power), ...]."""
    result = np.eye(2, dtype=complex)
    for index, power in word:
        generator = beta_generator(index, s)
        if power < 0:
            generator = la.inv(generator)
        for _ in range(abs(power)):
            result = result @ generator
    return result


def definiteness_sign(omega: float) -> int | None:
    """Return epsilon for which H=epsilon*J is positive definite."""
    if 0.0 < omega < INNER_LEFT:
        return 1
    if INNER_RIGHT < omega < TWO_PI:
        return -1
    return None


def positive_form(omega: float) -> np.ndarray:
    """Return the sign-normalized positive form H on Omega_+."""
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


def burau_unitary(word: list[tuple[int, int]], omega: float) -> np.ndarray:
    """Return U_omega(word)=H^(1/2) beta_omega(word) H^(-1/2)."""
    s = np.exp(0.5j * omega)
    return unitarize(beta_word(word, s), positive_form(omega))


def elementary_unitaries(omega: float) -> tuple[np.ndarray, np.ndarray]:
    """Return U1(omega) and U2(omega)."""
    return burau_unitary([(1, 1)], omega), burau_unitary([(2, 1)], omega)


def ordered_products(omega: float) -> tuple[np.ndarray, np.ndarray]:
    """Return V12=U1 U2 and V21=U2 U1."""
    unitary_one, unitary_two = elementary_unitaries(omega)
    return unitary_one @ unitary_two, unitary_two @ unitary_one


def controlled_order_unitary(omega: float, phase: float = 0.0) -> np.ndarray:
    """Return |0><0| tensor V12 + exp(i phase)|1><1| tensor V21."""
    order_twelve, order_twenty_one = ordered_products(omega)
    projector_zero = np.diag([1.0, 0.0])
    projector_one = np.diag([0.0, 1.0])
    return np.kron(projector_zero, order_twelve) + np.exp(
        1j * phase
    ) * np.kron(projector_one, order_twenty_one)


def _covering_arc(unitary: np.ndarray) -> float:
    """Length of the shortest closed arc containing a unitary's spectrum."""
    angles = np.sort(np.mod(np.angle(la.eigvals(unitary)), TWO_PI))
    gaps = np.concatenate((np.diff(angles), [TWO_PI - angles[-1] + angles[0]]))
    return float(np.clip(TWO_PI - np.max(gaps), 0.0, TWO_PI))


def numerical_range_distance(unitary: np.ndarray) -> float:
    """Distance from zero to the numerical range of a normal unitary."""
    covering_arc = _covering_arc(unitary)
    if covering_arc >= np.pi:
        return 0.0
    return float(np.cos(covering_arc / 2.0))


def helstrom_unitary_success(first: np.ndarray, second: np.ndarray) -> float:
    """Optimal equal-prior, single-use discrimination of two unitary channels."""
    relative = first.conj().T @ second
    capped_arc = min(_covering_arc(relative), np.pi)
    return 0.5 * (1.0 + np.sin(capped_arc / 2.0))


def exact_order_witness(omega: float) -> float:
    """Closed form for p*(U1 U2,U2 U1)-1/2."""
    radicand = 1.0 - (0.5 - np.cos(omega)) ** 2
    return 0.5 * np.sqrt(max(0.0, radicand))


def exact_minimum_visibility(omega: float) -> float:
    """Closed form for min_psi |<psi|V12^dagger V21|psi>|."""
    return abs(0.5 - np.cos(omega))


def _format_phase_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0.0, TWO_PI)
    axis.set_xticks([0.0, INNER_LEFT, INNER_RIGHT, TWO_PI])
    axis.set_xticklabels([r"$0$", r"$2\pi/3$", r"$4\pi/3$", r"$2\pi$"])
    axis.set_xlabel(r"$\omega$")
    axis.axvspan(0.0, INNER_LEFT, color="#2ca02c", alpha=0.07)
    axis.axvspan(INNER_RIGHT, TWO_PI, color="#2ca02c", alpha=0.07)
    for boundary in (INNER_LEFT, INNER_RIGHT):
        axis.axvline(boundary, color="0.55", linestyle=":", linewidth=0.9)


def run_verification(
    *,
    grid_points: int = 6001,
    boundary_margin: float = 1e-4,
    save_figure: bool = True,
    show_figure: bool = True,
    figure_path: str | Path = "figures/witness_gap_summary.png",
) -> dict[str, float]:
    """Run all manuscript checks, produce the figure, and return diagnostics."""
    omegas = np.linspace(boundary_margin, TWO_PI - boundary_margin, grid_points)
    valid = np.array(
        [definiteness_sign(float(value)) is not None for value in omegas]
    )

    lambda_minus = 2.0 * np.cos(omegas / 2.0) - 1.0
    lambda_plus = 2.0 * np.cos(omegas / 2.0) + 1.0
    form_error = np.full(grid_points, np.nan)
    braid_error = np.full(grid_points, np.nan)
    unitary_error = np.full(grid_points, np.nan)
    controlled_order_error = np.full(grid_points, np.nan)
    numerical_witness = np.full(grid_points, np.nan)
    closed_witness = np.full(grid_points, np.nan)
    numerical_visibility = np.full(grid_points, np.nan)
    closed_visibility = np.full(grid_points, np.nan)
    commutator_trace_error = np.full(grid_points, np.nan)
    commutator_determinant_error = np.full(grid_points, np.nan)
    minimum_h_eigenvalue = np.inf

    identity_two = np.eye(2, dtype=complex)
    identity_four = np.eye(4, dtype=complex)

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
        unitary_error[position] = max(
            la.norm(unitary_one.conj().T @ unitary_one - identity_two),
            la.norm(unitary_two.conj().T @ unitary_two - identity_two),
        )
        braid_error[position] = la.norm(
            unitary_one @ unitary_two @ unitary_one
            - unitary_two @ unitary_one @ unitary_two
        )

        order_twelve = unitary_one @ unitary_two
        order_twenty_one = unitary_two @ unitary_one
        relative = order_twelve.conj().T @ order_twenty_one
        controlled = controlled_order_unitary(float(omega))
        controlled_order_error[position] = la.norm(
            controlled.conj().T @ controlled - identity_four
        )

        commutator_trace_error[position] = abs(
            np.trace(relative) - (1.0 - 2.0 * np.cos(omega))
        )
        commutator_determinant_error[position] = abs(la.det(relative) - 1.0)
        numerical_witness[position] = (
            helstrom_unitary_success(order_twelve, order_twenty_one) - 0.5
        )
        closed_witness[position] = exact_order_witness(float(omega))
        numerical_visibility[position] = numerical_range_distance(relative)
        closed_visibility[position] = exact_minimum_visibility(float(omega))

    diagnostics = {
        "max_form_error": float(np.nanmax(form_error)),
        "max_braid_error": float(np.nanmax(braid_error)),
        "max_unitary_error": float(np.nanmax(unitary_error)),
        "max_controlled_order_unitary_error": float(
            np.nanmax(controlled_order_error)
        ),
        "max_closed_form_error": float(
            np.nanmax(np.abs(numerical_witness - closed_witness))
        ),
        "max_visibility_closed_form_error": float(
            np.nanmax(np.abs(numerical_visibility - closed_visibility))
        ),
        "max_commutator_trace_error": float(np.nanmax(commutator_trace_error)),
        "max_commutator_determinant_error": float(
            np.nanmax(commutator_determinant_error)
        ),
        "minimum_sampled_H_eigenvalue": float(minimum_h_eigenvalue),
        "maximum_order_witness": float(np.nanmax(numerical_witness)),
        "minimum_order_visibility": float(np.nanmin(numerical_visibility)),
        "order_witness_at_pi_over_3": exact_order_witness(np.pi / 3.0),
        "order_witness_at_5pi_over_3": exact_order_witness(5.0 * np.pi / 3.0),
        "visibility_at_pi_over_3": exact_minimum_visibility(np.pi / 3.0),
        "visibility_at_5pi_over_3": exact_minimum_visibility(5.0 * np.pi / 3.0),
    }

    assert diagnostics["minimum_sampled_H_eigenvalue"] > 0.0
    assert diagnostics["max_form_error"] < 1e-10
    assert diagnostics["max_braid_error"] < 1e-10
    assert diagnostics["max_unitary_error"] < 1e-9
    assert diagnostics["max_controlled_order_unitary_error"] < 1e-9
    assert diagnostics["max_closed_form_error"] < 1e-10
    assert diagnostics["max_visibility_closed_form_error"] < 1e-10
    assert diagnostics["max_commutator_trace_error"] < 1e-10
    assert diagnostics["max_commutator_determinant_error"] < 1e-10
    assert abs(diagnostics["order_witness_at_pi_over_3"] - 0.5) < 1e-12
    assert abs(diagnostics["order_witness_at_5pi_over_3"] - 0.5) < 1e-12
    assert diagnostics["visibility_at_pi_over_3"] < 1e-12
    assert diagnostics["visibility_at_5pi_over_3"] < 1e-12

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))

    axes[0].plot(omegas, lambda_minus, label=r"$2\cos(\omega/2)-1$")
    axes[0].plot(omegas, lambda_plus, label=r"$2\cos(\omega/2)+1$")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("eigenvalue")
    axes[0].set_title(r"(a) Signature of $J(\omega)$")
    axes[0].legend(fontsize=8, frameon=False)

    error_floor = 1e-16
    axes[1].plot(
        omegas, np.maximum(form_error, error_floor), label=r"$H$ preservation"
    )
    axes[1].plot(
        omegas, np.maximum(braid_error, error_floor), label="braid relation"
    )
    axes[1].plot(
        omegas, np.maximum(unitary_error, error_floor), label="unitarity"
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-16, 1e-8)
    axes[1].set_ylabel("matrix-norm residual")
    axes[1].set_title("(b) Numerical checks")
    axes[1].legend(fontsize=8, frameon=False)

    axes[2].plot(
        omegas,
        closed_witness,
        color="#3b6fb6",
        linewidth=2.2,
        label=r"$\mathcal{W}_{\rm NA}$",
    )
    axes[2].plot(
        omegas,
        closed_visibility,
        color="#c54e55",
        linewidth=2.0,
        label=r"$\mathcal{V}_{\min}$",
    )
    marker_stride = max(1, grid_points // 24)
    axes[2].plot(
        omegas[::marker_stride],
        numerical_witness[::marker_stride],
        linestyle="none",
        marker="o",
        markersize=3.0,
        color="#214a80",
    )
    axes[2].plot(
        omegas[::marker_stride],
        numerical_visibility[::marker_stride],
        linestyle="none",
        marker="s",
        markersize=2.8,
        color="#8e3037",
    )
    axes[2].set_ylim(0.0, 1.04)
    axes[2].set_ylabel("dimensionless response")
    axes[2].set_title("(c) Direct braid-order response")
    axes[2].legend(fontsize=8, frameon=False)

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
