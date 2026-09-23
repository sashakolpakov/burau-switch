"""Numerical checks for the Burau--Squier controlled-order experiment.

A single braid word supplies the control mixer. The target switch retains
the two orders BA and AB, and no omega-dependent path phase is inserted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la


TWO_PI = 2.0 * np.pi
INNER_LEFT = 2.0 * np.pi / 3.0
INNER_RIGHT = 4.0 * np.pi / 3.0
MIXER_WORD = [(1, 1), (2, 1), (1, -1), (2, -1)]


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
    """Return epsilon for which H=epsilon*J is positive definite."""
    if 0.0 <= omega < INNER_LEFT:
        return 1
    if INNER_RIGHT < omega < TWO_PI:
        return -1
    return None


def positive_form(omega: float) -> np.ndarray:
    """The sign-normalized positive form H on Omega_def."""
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


def rx(angle: float) -> np.ndarray:
    cosine = np.cos(angle / 2.0)
    sine = -1j * np.sin(angle / 2.0)
    return np.array([[cosine, sine], [sine, cosine]], dtype=complex)


def rz(angle: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def switch_unitary(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The fixed switch S=|0><0| tensor BA + |1><1| tensor AB."""
    projector_zero = np.diag([1.0, 0.0])
    projector_one = np.diag([0.0, 1.0])
    return np.kron(projector_zero, second @ first) + np.kron(
        projector_one, first @ second
    )


def mixer_unitary(omega: float) -> np.ndarray:
    """Unitarize the one-word mixer [sigma_1,sigma_2]."""
    s = np.exp(0.5j * omega)
    return unitarize(beta_word(MIXER_WORD, s), positive_form(omega))


def _dressed_response(
    omega: float, first_target: np.ndarray, second_target: np.ndarray
) -> tuple[float, float, float]:
    identity_two = np.eye(2, dtype=complex)
    identity_four = np.eye(4, dtype=complex)
    bare_switch = switch_unitary(first_target, second_target)
    mixer = mixer_unitary(omega)
    dressed_switch = (
        np.kron(mixer, identity_two)
        @ bare_switch
        @ np.kron(mixer, identity_two)
    )
    p_switch = helstrom_unitary_success(identity_four, bare_switch)
    p_test = helstrom_unitary_success(identity_four, dressed_switch)
    return p_switch, p_test, p_test - p_switch


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
    grid_points: int = 6000,
    boundary_margin: float = 1e-4,
    save_figure: bool = True,
    show_figure: bool = True,
    figure_path: str | Path = "figures/witness_gap_summary.png",
) -> dict[str, float]:
    """Run the manuscript checks, plot them, and return diagnostics."""
    omegas = np.linspace(0.0, TWO_PI, grid_points, endpoint=False)
    valid = np.array(
        [
            definiteness_sign(float(value)) is not None
            and abs(value - INNER_LEFT) > boundary_margin
            and abs(value - INNER_RIGHT) > boundary_margin
            for value in omegas
        ]
    )

    lambda_minus = 2.0 * np.cos(omegas / 2.0) - 1.0
    lambda_plus = 2.0 * np.cos(omegas / 2.0) + 1.0
    form_error = np.full(grid_points, np.nan)
    braid_error = np.full(grid_points, np.nan)
    unitary_error = np.full(grid_points, np.nan)
    response_contrast = np.full(grid_points, np.nan)
    abelian_null_contrast = np.full(grid_points, np.nan)
    abelian_word_error = np.full(grid_points, np.nan)
    minimum_h_eigenvalue = np.inf

    first_target = rx(1.5)
    second_target = rz(0.75)
    identity_two = np.eye(2, dtype=complex)
    identity_four = np.eye(4, dtype=complex)
    bare_switch = switch_unitary(first_target, second_target)
    p_switch = helstrom_unitary_success(identity_four, bare_switch)
    p_fixed = max(
        helstrom_unitary_success(identity_two, first_target @ second_target),
        helstrom_unitary_success(identity_two, second_target @ first_target),
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
        mixer = unitarize(beta_word(MIXER_WORD, s), form)
        unitary_error[position] = la.norm(
            mixer.conj().T @ mixer - identity_two
        )
        braid_error[position] = la.norm(
            unitary_one @ unitary_two @ unitary_one
            - unitary_two @ unitary_one @ unitary_two
        )

        dressed_switch = (
            np.kron(mixer, identity_two)
            @ bare_switch
            @ np.kron(mixer, identity_two)
        )
        response_contrast[position] = (
            helstrom_unitary_success(identity_four, dressed_switch) - p_switch
        )

        # A general pair of diagonal generator images commutes, so its
        # commutator word is the identity even when the two images differ.
        diagonal_one = np.diag(
            [np.exp(1j * omega), np.exp(-1j * omega)]
        )
        diagonal_two = np.diag(
            [np.exp(0.37j * omega), np.exp(-0.37j * omega)]
        )
        abelian_word = (
            diagonal_one
            @ diagonal_two
            @ diagonal_one.conj().T
            @ diagonal_two.conj().T
        )
        abelian_word_error[position] = la.norm(abelian_word - identity_two)
        abelian_dressed = (
            np.kron(abelian_word, identity_two)
            @ bare_switch
            @ np.kron(abelian_word, identity_two)
        )
        abelian_null_contrast[position] = (
            helstrom_unitary_success(identity_four, abelian_dressed) - p_switch
        )

    _, p_test_at_pi_over_3, contrast_at_pi_over_3 = _dressed_response(
        np.pi / 3.0, first_target, second_target
    )
    _, p_test_at_5pi_over_3, contrast_at_5pi_over_3 = _dressed_response(
        5.0 * np.pi / 3.0, first_target, second_target
    )

    diagnostics = {
        "det_J_at_0": float(la.det(squier_form(0.0)).real),
        "det_J_at_2pi_over_3": float(la.det(squier_form(INNER_LEFT)).real),
        "det_J_at_4pi_over_3": float(la.det(squier_form(INNER_RIGHT)).real),
        "det_J_at_2pi": float(la.det(squier_form(TWO_PI)).real),
        "max_form_error": float(np.nanmax(form_error)),
        "max_braid_error": float(np.nanmax(braid_error)),
        "max_unitary_error": float(np.nanmax(unitary_error)),
        "minimum_sampled_H_eigenvalue": float(minimum_h_eigenvalue),
        "p_fixed": float(p_fixed),
        "p_switch": float(p_switch),
        "fixed_switch_difference": float(abs(p_fixed - p_switch)),
        "minimum_response_contrast": float(np.nanmin(response_contrast)),
        "maximum_response_contrast": float(np.nanmax(response_contrast)),
        "contrast_at_pi_over_3": float(contrast_at_pi_over_3),
        "contrast_at_5pi_over_3": float(contrast_at_5pi_over_3),
        "p_test_at_pi_over_3": float(p_test_at_pi_over_3),
        "p_test_at_5pi_over_3": float(p_test_at_5pi_over_3),
        "max_abelian_word_error": float(np.nanmax(abelian_word_error)),
        "max_abelian_null_contrast": float(
            np.nanmax(np.abs(abelian_null_contrast))
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
    assert diagnostics["fixed_switch_difference"] < 1e-12
    assert diagnostics["minimum_response_contrast"] > -1e-10
    assert diagnostics["maximum_response_contrast"] > 0.1
    assert abs(diagnostics["contrast_at_pi_over_3"]) < 1e-10
    assert abs(diagnostics["contrast_at_5pi_over_3"]) < 1e-10
    assert diagnostics["max_abelian_word_error"] < 1e-12
    assert diagnostics["max_abelian_null_contrast"] < 1e-12

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
        omegas, np.maximum(unitary_error, error_floor), label=r"$M$ unitarity"
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-16, 1e-8)
    axes[1].set_ylabel("matrix-norm residual")
    axes[1].set_title("(b) Numerical checks")
    axes[1].legend(fontsize=8, frameon=False)

    axes[2].plot(
        omegas,
        response_contrast,
        color="#7f3c8d",
        linewidth=2.0,
        label=r"$w=[\sigma_1,\sigma_2]$",
    )
    axes[2].plot(
        omegas,
        abelian_null_contrast,
        color="#d95f02",
        linestyle="--",
        linewidth=1.3,
        label="commuting-generator null",
    )
    axes[2].axhline(0.0, color="black", linewidth=0.7)
    axes[2].set_ylim(-0.008, 0.145)
    axes[2].set_ylabel(r"$\Delta_{\rm int}=p_{\rm test}-p_{\rm switch}$")
    axes[2].set_title("(c) One-word mixer response")
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
