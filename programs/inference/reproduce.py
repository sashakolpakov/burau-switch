"""Expressivity probe for passive Burau-block neural feature maps.

A passive interferometer followed by square-law detection produces Hermitian
quadratic features.  This experiment tests whether banks of four-mode meshes
built from two-mode Burau--Squier blocks span that entire quadratic space and
compares them with phase-only and generic Haar-unitary baselines.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import numpy.linalg as la

from burau_switch import beta_word, positive_form, unitarize


HERE = Path(__file__).resolve().parent
FIGURE_PATH = HERE / "figures" / "inference_probe.png"
RESULTS_PATH = HERE / "results" / "inference_probe.json"
DEFAULT_MESH_DEPTH = 12


def _random_omega(rng: np.random.Generator, margin: float = 0.08) -> float:
    if rng.random() < 0.5:
        return float(rng.uniform(margin, 2.0 * np.pi / 3.0 - margin))
    return float(rng.uniform(4.0 * np.pi / 3.0 + margin, 2.0 * np.pi - margin))


def _random_word(
    rng: np.random.Generator, minimum_length: int = 2, maximum_length: int = 7
) -> list[tuple[int, int]]:
    length = int(rng.integers(minimum_length, maximum_length + 1))
    letters = rng.choice(np.array([-2, -1, 1, 2]), size=length)
    return [(abs(int(letter)), 1 if letter > 0 else -1) for letter in letters]


def _burau_block(rng: np.random.Generator) -> np.ndarray:
    omega = _random_omega(rng)
    s = np.exp(0.5j * omega)
    return unitarize(beta_word(_random_word(rng), s), positive_form(omega))


def _embed_two_mode(
    block: np.ndarray, number_of_modes: int, first: int, second: int
) -> np.ndarray:
    embedded = np.eye(number_of_modes, dtype=complex)
    indices = np.array([first, second])
    embedded[np.ix_(indices, indices)] = block
    return embedded


def _burau_mesh(
    rng: np.random.Generator,
    number_of_modes: int,
    depth: int = DEFAULT_MESH_DEPTH,
) -> np.ndarray:
    if number_of_modes != 4:
        raise ValueError("the present probe uses a four-mode mesh")
    pair_cycle = ((0, 1), (2, 3), (1, 2), (0, 3))
    result = np.eye(number_of_modes, dtype=complex)
    for layer in range(depth):
        first, second = pair_cycle[layer % len(pair_cycle)]
        result = _embed_two_mode(
            _burau_block(rng), number_of_modes, first, second
        ) @ result
    return result


def _haar_unitary(
    rng: np.random.Generator, number_of_modes: int
) -> np.ndarray:
    matrix = rng.normal(size=(number_of_modes, number_of_modes)) + 1j * rng.normal(
        size=(number_of_modes, number_of_modes)
    )
    unitary, triangular = la.qr(matrix)
    diagonal = np.diag(triangular)
    phases = diagonal / np.where(np.abs(diagonal) > 0.0, np.abs(diagonal), 1.0)
    return unitary @ np.diag(phases.conj())


def _phase_only_unitary(
    rng: np.random.Generator, number_of_modes: int
) -> np.ndarray:
    phases = rng.uniform(0.0, 2.0 * np.pi, size=number_of_modes)
    return np.diag(np.exp(1j * phases))


def _hermitian_coordinates(matrix: np.ndarray) -> np.ndarray:
    """Real Hilbert--Schmidt coordinates for a Hermitian matrix."""
    number_of_modes = matrix.shape[0]
    coordinates: list[float] = [
        float(np.real(matrix[index, index])) for index in range(number_of_modes)
    ]
    for row in range(number_of_modes):
        for column in range(row + 1, number_of_modes):
            coordinates.append(float(np.sqrt(2.0) * np.real(matrix[row, column])))
            coordinates.append(float(np.sqrt(2.0) * np.imag(matrix[row, column])))
    return np.asarray(coordinates)


def _measurement_coordinates(unitaries: list[np.ndarray]) -> np.ndarray:
    rows = []
    for unitary in unitaries:
        for output in range(unitary.shape[0]):
            row = unitary[output]
            projector = np.outer(row.conj(), row)
            rows.append(_hermitian_coordinates(projector))
    return np.asarray(rows)


def _prefix_ranks(unitaries: list[np.ndarray]) -> list[int]:
    return [
        int(la.matrix_rank(_measurement_coordinates(unitaries[:count]), tol=1e-10))
        for count in range(1, len(unitaries) + 1)
    ]


def _rank_upper_bounds(
    number_of_modes: int, number_of_settings: int
) -> list[int]:
    """Maximum rank supplied by complete orthonormal measurement bases.

    The first setting contributes at most d projectors. Every later setting
    contributes at most d - 1 new directions because its projectors sum to
    the identity, which is already in the span.
    """
    maximum_dimension = number_of_modes**2
    return [
        min(
            maximum_dimension,
            number_of_modes + (count - 1) * (number_of_modes - 1),
        )
        for count in range(1, number_of_settings + 1)
    ]


def _frame_condition_number(unitaries: list[np.ndarray]) -> float:
    singular_values = la.svd(_measurement_coordinates(unitaries), compute_uv=False)
    nonzero_singular_values = singular_values[singular_values > 1e-10]
    return float(nonzero_singular_values[0] / nonzero_singular_values[-1])


def _random_states(
    rng: np.random.Generator, samples: int, number_of_modes: int
) -> np.ndarray:
    states = rng.normal(size=(samples, number_of_modes)) + 1j * rng.normal(
        size=(samples, number_of_modes)
    )
    return states / la.norm(states, axis=1, keepdims=True)


def _intensity_features(
    states: np.ndarray, unitaries: list[np.ndarray]
) -> np.ndarray:
    return np.concatenate(
        [np.abs(states @ unitary.T) ** 2 for unitary in unitaries], axis=1
    )


def _random_hermitian(
    rng: np.random.Generator, number_of_modes: int
) -> np.ndarray:
    matrix = rng.normal(size=(number_of_modes, number_of_modes)) + 1j * rng.normal(
        size=(number_of_modes, number_of_modes)
    )
    hermitian = 0.5 * (matrix + matrix.conj().T)
    return hermitian / la.norm(hermitian)


def _expectation(states: np.ndarray, observable: np.ndarray) -> np.ndarray:
    return np.real(np.einsum("bi,ij,bj->b", states.conj(), observable, states))


def _normalized_regression_error(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    weights = la.lstsq(train_features, train_targets, rcond=1e-12)[0]
    predictions = test_features @ weights
    error = float(np.sqrt(np.mean((predictions - test_targets) ** 2)))
    normalized_error = error / float(np.std(test_targets))
    return normalized_error, predictions


def run_inference_probe(
    *,
    number_of_modes: int = 4,
    number_of_settings: int = 8,
    train_samples: int = 3000,
    test_samples: int = 1500,
    robustness_trials: int = 32,
    mesh_depth: int = DEFAULT_MESH_DEPTH,
    seed: int = 2718,
) -> dict[str, object]:
    """Measure quadratic span and regression capability of each feature bank."""
    if number_of_modes != 4:
        raise ValueError("the present probe uses four-mode Burau meshes")
    minimum_full_span_settings = number_of_modes + 1
    if number_of_settings < minimum_full_span_settings:
        raise ValueError(
            "number_of_settings must be at least d + 1 for a full quadratic span"
        )
    if robustness_trials < 1:
        raise ValueError("robustness_trials must be positive")
    if mesh_depth < 1:
        raise ValueError("mesh_depth must be positive")

    rng = np.random.default_rng(seed)
    burau = [
        _burau_mesh(rng, number_of_modes, mesh_depth)
        for _ in range(number_of_settings)
    ]
    phase_only = [
        _phase_only_unitary(rng, number_of_modes)
        for _ in range(number_of_settings)
    ]
    haar = [
        _haar_unitary(rng, number_of_modes) for _ in range(number_of_settings)
    ]
    banks = {"Burau mesh": burau, "phase only": phase_only, "Haar baseline": haar}
    ranks = {name: _prefix_ranks(bank) for name, bank in banks.items()}
    rank_upper_bounds = _rank_upper_bounds(number_of_modes, number_of_settings)

    coordinates = _measurement_coordinates(burau)
    singular_values = la.svd(coordinates, compute_uv=False)
    condition_number = _frame_condition_number(burau)
    identity = np.eye(number_of_modes, dtype=complex)
    maximum_burau_unitarity_error = float(
        max(la.norm(unitary.conj().T @ unitary - identity) for unitary in burau)
    )

    robustness_rng = np.random.default_rng(seed + 10_000)
    robustness_ranks: dict[str, list[int]] = {
        "Burau mesh": [],
        "Haar baseline": [],
    }
    robustness_condition_numbers: dict[str, list[float]] = {
        "Burau mesh": [],
        "Haar baseline": [],
    }
    for _ in range(robustness_trials):
        trial_banks = {
            "Burau mesh": [
                _burau_mesh(robustness_rng, number_of_modes, mesh_depth)
                for _ in range(minimum_full_span_settings)
            ],
            "Haar baseline": [
                _haar_unitary(robustness_rng, number_of_modes)
                for _ in range(minimum_full_span_settings)
            ],
        }
        for bank_name, bank in trial_banks.items():
            rank = _prefix_ranks(bank)[-1]
            robustness_ranks[bank_name].append(rank)
            if rank == number_of_modes**2:
                robustness_condition_numbers[bank_name].append(
                    _frame_condition_number(bank)
                )

    for bank_name, condition_numbers in robustness_condition_numbers.items():
        if not condition_numbers:
            raise AssertionError(
                f"no full-rank minimal {bank_name} bank in robustness sample"
            )

    train_states = _random_states(rng, train_samples, number_of_modes)
    test_states = _random_states(rng, test_samples, number_of_modes)
    first_observable = _random_hermitian(rng, number_of_modes)
    second_observable = _random_hermitian(rng, number_of_modes)
    train_first = _expectation(train_states, first_observable)
    test_first = _expectation(test_states, first_observable)
    train_second = _expectation(train_states, second_observable)
    test_second = _expectation(test_states, second_observable)
    targets = {
        "quadratic": (train_first, test_first),
        "quartic": (train_first * train_second, test_first * test_second),
    }

    errors: dict[str, dict[str, float]] = {}
    quadratic_predictions: dict[str, np.ndarray] = {}
    for bank_name, bank in banks.items():
        train_features = _intensity_features(train_states, bank)
        test_features = _intensity_features(test_states, bank)
        errors[bank_name] = {}
        for target_name, (train_target, test_target) in targets.items():
            error, predictions = _normalized_regression_error(
                train_features, train_target, test_features, test_target
            )
            errors[bank_name][target_name] = error
            if target_name == "quadratic":
                quadratic_predictions[bank_name] = predictions

    maximum_dimension = number_of_modes**2
    diagnostics: dict[str, object] = {
        "number_of_modes": number_of_modes,
        "number_of_settings": number_of_settings,
        "burau_blocks_per_mesh": mesh_depth,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "seed": seed,
        "minimum_settings_for_full_quadratic_span": minimum_full_span_settings,
        "maximum_hermitian_feature_dimension": maximum_dimension,
        "orthonormal_basis_rank_upper_bounds": rank_upper_bounds,
        "feature_ranks_by_setting": ranks,
        "burau_bank_condition_number": condition_number,
        "maximum_burau_mesh_unitarity_error": maximum_burau_unitarity_error,
        "minimal_bank_robustness": {
            bank_name: {
                "trials": robustness_trials,
                "full_rank_fraction": float(
                    np.mean(
                        np.asarray(robustness_ranks[bank_name])
                        == maximum_dimension
                    )
                ),
                "minimum_rank": int(min(robustness_ranks[bank_name])),
                "median_condition_number_among_full_rank_banks": float(
                    np.median(robustness_condition_numbers[bank_name])
                ),
                "maximum_condition_number_among_full_rank_banks": float(
                    np.max(robustness_condition_numbers[bank_name])
                ),
            }
            for bank_name in robustness_ranks
        },
        "normalized_test_rmse": errors,
        "interpretation": (
            "A passive unitary bank plus intensity detection and a linear "
            "readout spans Hermitian quadratic scores, not arbitrary deep "
            "neural functions."
        ),
    }

    if ranks["Burau mesh"] != rank_upper_bounds:
        raise AssertionError("Burau feature bank did not saturate the rank bound")
    if ranks["Burau mesh"][-1] != maximum_dimension:
        raise AssertionError("Burau feature bank did not reach full quadratic rank")
    if min(robustness_ranks["Burau mesh"]) != maximum_dimension:
        raise AssertionError("a sampled minimal Burau bank was rank deficient")
    if maximum_burau_unitarity_error > 1e-10:
        raise AssertionError("a Burau mesh is not numerically unitary")
    if errors["Burau mesh"]["quadratic"] > 1e-10:
        raise AssertionError("full-rank Burau bank did not recover a quadratic target")
    if errors["Burau mesh"]["quartic"] < 0.05:
        raise AssertionError(
            "quartic target did not expose the passive quadratic ceiling"
        )

    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.5))
    flat_axes = axes.ravel()
    settings = np.arange(1, number_of_settings + 1)
    colors = {
        "Burau mesh": "#7f3c8d",
        "phase only": "#d95f02",
        "Haar baseline": "#1b9e77",
    }
    plot_order = ("phase only", "Haar baseline", "Burau mesh")
    line_styles = {
        "Burau mesh": "-",
        "phase only": ":",
        "Haar baseline": "--",
    }
    markers = {"Burau mesh": "o", "phase only": "s", "Haar baseline": "x"}
    for name in plot_order:
        values = ranks[name]
        flat_axes[0].plot(
            settings,
            values,
            marker=markers[name],
            linestyle=line_styles[name],
            linewidth=2.5 if name == "Haar baseline" else 1.8,
            markersize=8 if name == "Burau mesh" else 6,
            markerfacecolor="white" if name == "Burau mesh" else colors[name],
            zorder=3 if name == "Burau mesh" else 2,
            label=name,
            color=colors[name],
        )
    flat_axes[0].axhline(
        maximum_dimension, color="black", linestyle="--", linewidth=0.9
    )
    flat_axes[0].plot(
        settings,
        rank_upper_bounds,
        color="0.35",
        linestyle=(0, (1, 2)),
        linewidth=1.2,
        label="basis-count upper bound",
    )
    flat_axes[0].set_xlabel("measurement settings")
    flat_axes[0].set_ylabel("Hermitian feature rank")
    flat_axes[0].set_title("(a) Quadratic feature-space coverage")
    flat_axes[0].text(
        5.15,
        14.7,
        "Burau = Haar rank",
        color=colors["Burau mesh"],
        fontsize=8,
    )
    flat_axes[0].legend(fontsize=8)

    flat_axes[1].semilogy(
        np.arange(1, len(singular_values) + 1),
        singular_values / singular_values[0],
        marker="o",
        color=colors["Burau mesh"],
    )
    flat_axes[1].set_xlabel("singular-value index")
    flat_axes[1].set_ylabel("normalized singular value")
    flat_axes[1].set_title("(b) Burau measurement frame")

    flat_axes[2].scatter(
        test_first,
        quadratic_predictions["Burau mesh"],
        s=7,
        alpha=0.35,
        color=colors["Burau mesh"],
    )
    limits = [float(np.min(test_first)), float(np.max(test_first))]
    flat_axes[2].plot(limits, limits, color="black", linestyle="--")
    flat_axes[2].set_xlabel("true quadratic score")
    flat_axes[2].set_ylabel("linear readout prediction")
    flat_axes[2].set_title("(c) Exact quadratic inference")

    names = list(banks)
    positions = np.arange(len(names))
    width = 0.34
    flat_axes[3].bar(
        positions - width / 2,
        [errors[name]["quadratic"] for name in names],
        width,
        label="quadratic target",
        color="#4c78a8",
    )
    flat_axes[3].bar(
        positions + width / 2,
        [errors[name]["quartic"] for name in names],
        width,
        label="quartic target",
        color="#f58518",
    )
    flat_axes[3].set_yscale("log")
    flat_axes[3].set_xticks(positions)
    flat_axes[3].set_xticklabels(names, rotation=12)
    flat_axes[3].set_ylabel("normalized test RMSE")
    flat_axes[3].set_title("(d) Capability and ceiling")
    flat_axes[3].legend(fontsize=8)

    for axis in flat_axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Passive Burau meshes form a complete quadratic feature bank",
        fontsize=12,
    )
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)

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
    run_inference_probe()
