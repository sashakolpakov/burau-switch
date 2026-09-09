"""Conditioning and detector-noise scaling for quadratic inference banks.

This experiment uses generic unitary spectral and mutually unbiased bases.  It
does not compile those bases into the restricted Burau-derived block library.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import numpy.linalg as la


HERE = Path(__file__).resolve().parent
FIGURE_PATH = HERE / "figures" / "inference_scaling.png"
RESULTS_PATH = HERE / "results" / "inference_scaling.json"

DEFAULT_DIMENSIONS = (3, 4, 5, 7, 11, 13)
DEFAULT_CONDITION_TRIALS = {3: 80, 4: 80, 5: 60, 7: 40, 11: 20, 13: 12}
DEFAULT_NOISE_TRIALS = {3: 20, 4: 20, 5: 20, 7: 16, 11: 10, 13: 8}
DEFAULT_SAMPLES = 1200
DEFAULT_TOTAL_DETECTED_PHOTOELECTRONS = 1_000_000.0
DEFAULT_READ_NOISE_ELECTRONS_RMS = 5.0
DEFAULT_CONDITION_SEED = 20260908
DEFAULT_NOISE_SEED = 260908
SINGULAR_VALUE_TOLERANCE = 1e-10


def _haar_unitary(rng: np.random.Generator, dimension: int) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    unitary, triangular = la.qr(matrix)
    diagonal = np.diag(triangular)
    phases = diagonal / np.where(np.abs(diagonal) > 0.0, np.abs(diagonal), 1.0)
    return unitary @ np.diag(phases.conj())


def _odd_prime_mub(dimension: int) -> list[np.ndarray]:
    """Return the computational basis and odd-prime chirp MUBs.

    Row ``b`` of basis ``a`` has entry ``n`` equal to
    ``d**(-1/2) exp(2 pi i (a n**2 + b n) / d)``.  This implementation uses
    that formula only for the declared odd-prime dimensions.
    """
    coordinate = np.arange(dimension)
    bank = [np.eye(dimension, dtype=complex)]
    for basis_index in range(dimension):
        bank.append(
            np.asarray(
                [
                    np.exp(
                        2j
                        * np.pi
                        * (
                            basis_index * coordinate**2
                            + row_index * coordinate
                        )
                        / dimension
                    )
                    / np.sqrt(dimension)
                    for row_index in range(dimension)
                ]
            )
        )
    return bank


def _pauli_mub_dimension_four() -> list[np.ndarray]:
    """Return five two-qubit stabilizer MUBs in dimension four."""
    identity = np.eye(2, dtype=complex)
    pauli_x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    pauli_y = np.asarray([[0.0, -1j], [1j, 0.0]], dtype=complex)
    pauli_z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    paulis = {
        "I": identity,
        "X": pauli_x,
        "Y": pauli_y,
        "Z": pauli_z,
    }

    def pauli_product(label: str) -> np.ndarray:
        return np.kron(paulis[label[0]], paulis[label[1]])

    commuting_classes = (
        ("ZI", "IZ"),
        ("XI", "IX"),
        ("YI", "IY"),
        ("XY", "YZ"),
        ("XZ", "ZY"),
    )
    bank = []
    for first, second in commuting_classes:
        _, eigenvectors = la.eigh(
            pauli_product(first) + 2.0 * pauli_product(second)
        )
        bank.append(eigenvectors.conj().T)
    return bank


def _complete_mub(dimension: int) -> list[np.ndarray]:
    if dimension == 4:
        return _pauli_mub_dimension_four()
    if dimension in (3, 5, 7, 11, 13):
        return _odd_prime_mub(dimension)
    raise ValueError(
        "the checked study implements d=4 and odd primes 3, 5, 7, 11, 13"
    )


def _hermitian_coordinates(matrix: np.ndarray) -> np.ndarray:
    """Return real coordinates in an orthonormal Hermitian matrix basis."""
    dimension = matrix.shape[0]
    coordinates: list[float] = [
        float(np.real(matrix[index, index])) for index in range(dimension)
    ]
    for row in range(dimension):
        for column in range(row + 1, dimension):
            coordinates.extend(
                (
                    float(np.sqrt(2.0) * np.real(matrix[row, column])),
                    float(np.sqrt(2.0) * np.imag(matrix[row, column])),
                )
            )
    return np.asarray(coordinates)


def _measurement_matrix(bank: Sequence[np.ndarray]) -> np.ndarray:
    rows = []
    for unitary in bank:
        for output_row in unitary:
            projector = np.outer(output_row.conj(), output_row)
            rows.append(_hermitian_coordinates(projector))
    return np.asarray(rows)


def _measurement_frame_condition(bank: Sequence[np.ndarray]) -> float:
    dimension = bank[0].shape[0]
    singular_values = la.svd(_measurement_matrix(bank), compute_uv=False)
    nonzero = singular_values[singular_values > SINGULAR_VALUE_TOLERANCE]
    if len(nonzero) != dimension**2:
        return float("inf")
    return float(nonzero[0] / nonzero[-1])


def _centered_frame_condition(bank: Sequence[np.ndarray]) -> float:
    """Condition number after removing the exactly known identity direction."""
    dimension = bank[0].shape[0]
    identity_coordinate = _hermitian_coordinates(
        np.eye(dimension, dtype=complex) / dimension
    )
    centered = _measurement_matrix(bank) - identity_coordinate[None, :]
    singular_values = la.svd(centered, compute_uv=False)
    nonzero = singular_values[singular_values > SINGULAR_VALUE_TOLERANCE]
    if len(nonzero) != dimension**2 - 1:
        return float("inf")
    return float(nonzero[0] / nonzero[-1])


def _validate_mub(bank: Sequence[np.ndarray]) -> dict[str, float]:
    dimension = bank[0].shape[0]
    identity = np.eye(dimension)
    unitarity_error = max(
        float(la.norm(unitary @ unitary.conj().T - identity))
        for unitary in bank
    )
    unbiasedness_error = 0.0
    for left in range(len(bank)):
        for right in range(left + 1, len(bank)):
            overlaps = np.abs(bank[left] @ bank[right].conj().T) ** 2
            unbiasedness_error = max(
                unbiasedness_error,
                float(np.max(np.abs(overlaps - 1.0 / dimension))),
            )
    return {
        "maximum_unbiasedness_error": unbiasedness_error,
        "maximum_unitarity_error": unitarity_error,
    }


def _random_states(
    rng: np.random.Generator, samples: int, dimension: int
) -> np.ndarray:
    states = rng.normal(size=(samples, dimension)) + 1j * rng.normal(
        size=(samples, dimension)
    )
    return states / la.norm(states, axis=1, keepdims=True)


def _random_traceless_observable(
    rng: np.random.Generator, dimension: int
) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    observable = 0.5 * (matrix + matrix.conj().T)
    observable -= np.trace(observable) * np.eye(dimension) / dimension
    return observable / la.norm(observable)


def _intensity_features(
    states: np.ndarray, bank: Sequence[np.ndarray]
) -> np.ndarray:
    return np.concatenate(
        [np.abs(states @ unitary.T) ** 2 for unitary in bank], axis=1
    )


def _observable_weights(
    bank: Sequence[np.ndarray], observable: np.ndarray
) -> np.ndarray:
    return la.lstsq(
        _measurement_matrix(bank).T,
        _hermitian_coordinates(observable),
        rcond=1e-12,
    )[0]


def _noisy_features(
    rng: np.random.Generator,
    exact_features: np.ndarray,
    dimension: int,
    settings: int,
    total_detected_photoelectrons: float,
    read_noise_electrons_rms: float,
) -> np.ndarray:
    expected = (
        total_detected_photoelectrons / settings
    ) * exact_features.reshape(-1, settings, dimension)
    counts = rng.poisson(expected).astype(float)
    counts += rng.normal(scale=read_noise_electrons_rms, size=counts.shape)
    counts = np.clip(counts, 0.0, None)
    counts /= np.maximum(np.sum(counts, axis=2, keepdims=True), 1.0)
    return counts.reshape(len(exact_features), -1)


def _normalized_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean((prediction - target) ** 2)) / np.std(target)
    )


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "maximum": float(np.max(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _plot_results(diagnostics: dict[str, object]) -> None:
    dimensions = np.asarray(diagnostics["dimensions_tested"], dtype=int)
    results = diagnostics["results_by_dimension"]

    mub_conditions = np.asarray(
        [results[str(d)]["frame_condition_number"]["complete_mub"] for d in dimensions]
    )
    minimum_condition_medians = np.asarray(
        [
            results[str(d)]["frame_condition_number"]["haar_minimum_d_plus_1"][
                "median"
            ]
            for d in dimensions
        ]
    )
    minimum_condition_p10 = np.asarray(
        [
            results[str(d)]["frame_condition_number"]["haar_minimum_d_plus_1"][
                "p10"
            ]
            for d in dimensions
        ]
    )
    minimum_condition_p90 = np.asarray(
        [
            results[str(d)]["frame_condition_number"]["haar_minimum_d_plus_1"][
                "p90"
            ]
            for d in dimensions
        ]
    )
    overcomplete_condition_medians = np.asarray(
        [
            results[str(d)]["frame_condition_number"]["haar_overcomplete_2d"][
                "median"
            ]
            for d in dimensions
        ]
    )

    noise_architectures = (
        ("spectral_one_setting", "spectral, one setting", "#0072B2", "o"),
        (
            "complete_mub_d_plus_1_settings",
            "complete MUB, d+1 settings",
            "#009E73",
            "s",
        ),
        (
            "haar_minimum_d_plus_1_settings",
            "Haar, d+1 settings",
            "#D55E00",
            "^",
        ),
        ("haar_overcomplete_2d_settings", "Haar, 2d settings", "#CC79A7", "D"),
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].semilogy(
        dimensions,
        mub_conditions,
        marker="s",
        color="#009E73",
        linewidth=2.0,
        label=r"complete MUB: $\sqrt{d+1}$",
    )
    axes[0].semilogy(
        dimensions,
        minimum_condition_medians,
        marker="^",
        color="#D55E00",
        linewidth=2.0,
        label="Haar, d+1 (median)",
    )
    axes[0].fill_between(
        dimensions,
        minimum_condition_p10,
        minimum_condition_p90,
        color="#D55E00",
        alpha=0.18,
        linewidth=0,
        label="Haar, d+1 (10--90%)",
    )
    axes[0].semilogy(
        dimensions,
        overcomplete_condition_medians,
        marker="D",
        color="#CC79A7",
        linewidth=1.8,
        label="Haar, 2d (median)",
    )
    axes[0].set_xlabel("mode dimension d")
    axes[0].set_ylabel(r"projector-frame condition $\kappa_2$")
    axes[0].set_title("(a) Full rank is not robust conditioning")
    axes[0].legend(fontsize=7.6, loc="upper left")

    for key, label, color, marker in noise_architectures:
        medians = [
            results[str(d)]["normalized_score_rmse"][key]["median"]
            for d in dimensions
        ]
        axes[1].semilogy(
            dimensions,
            medians,
            marker=marker,
            color=color,
            linewidth=1.9,
            label=label,
        )
    axes[1].axhline(
        1e-2,
        color="0.25",
        linestyle="--",
        linewidth=1.0,
        label="1% score-error guide",
    )
    axes[1].set_xlabel("mode dimension d")
    axes[1].set_ylabel("median normalized score RMSE")
    axes[1].set_title("(b) Fixed total detected-light budget")
    axes[1].legend(fontsize=7.3, loc="upper left")

    for axis in axes:
        axis.set_xticks(dimensions)
        axis.grid(alpha=0.22, which="both")
    figure.suptitle(
        "Generic-unitary quadratic inference scaling (not Burau compilation)",
        fontsize=11.5,
    )
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=240, bbox_inches="tight")
    plt.close(figure)


def run_inference_scaling(
    *,
    dimensions: Sequence[int] = DEFAULT_DIMENSIONS,
    condition_trials: dict[int, int] | None = None,
    noise_trials: dict[int, int] | None = None,
    samples: int = DEFAULT_SAMPLES,
    total_detected_photoelectrons: float = (
        DEFAULT_TOTAL_DETECTED_PHOTOELECTRONS
    ),
    read_noise_electrons_rms: float = DEFAULT_READ_NOISE_ELECTRONS_RMS,
    condition_seed: int = DEFAULT_CONDITION_SEED,
    noise_seed: int = DEFAULT_NOISE_SEED,
) -> dict[str, object]:
    """Run deterministic conditioning and detector-noise scaling sweeps."""
    start = time.perf_counter()
    dimensions = tuple(int(dimension) for dimension in dimensions)
    condition_trials = (
        DEFAULT_CONDITION_TRIALS.copy()
        if condition_trials is None
        else condition_trials.copy()
    )
    noise_trials = (
        DEFAULT_NOISE_TRIALS.copy()
        if noise_trials is None
        else noise_trials.copy()
    )
    if samples < 2:
        raise ValueError("samples must be at least two")
    if total_detected_photoelectrons <= 0.0:
        raise ValueError("total_detected_photoelectrons must be positive")
    if read_noise_electrons_rms < 0.0:
        raise ValueError("read_noise_electrons_rms cannot be negative")
    for dimension in dimensions:
        if condition_trials.get(dimension, 0) < 1:
            raise ValueError(f"missing positive condition trial count for d={dimension}")
        if noise_trials.get(dimension, 0) < 1:
            raise ValueError(f"missing positive noise trial count for d={dimension}")

    condition_rng = np.random.default_rng(condition_seed)
    noise_rng = np.random.default_rng(noise_seed)
    results_by_dimension: dict[str, object] = {}

    for dimension in dimensions:
        mub = _complete_mub(dimension)
        mub_validation = _validate_mub(mub)
        if len(mub) != dimension + 1:
            raise AssertionError("complete MUB does not contain d+1 bases")
        if mub_validation["maximum_unitarity_error"] > 1e-10:
            raise AssertionError(f"MUB unitarity validation failed at d={dimension}")
        if mub_validation["maximum_unbiasedness_error"] > 1e-10:
            raise AssertionError(f"MUB unbiasedness validation failed at d={dimension}")

        minimum_conditions = []
        overcomplete_conditions = []
        for _ in range(condition_trials[dimension]):
            minimum_conditions.append(
                _measurement_frame_condition(
                    [
                        _haar_unitary(condition_rng, dimension)
                        for _ in range(dimension + 1)
                    ]
                )
            )
            overcomplete_conditions.append(
                _measurement_frame_condition(
                    [
                        _haar_unitary(condition_rng, dimension)
                        for _ in range(2 * dimension)
                    ]
                )
            )

        errors: dict[str, list[float]] = {
            "complete_mub_d_plus_1_settings": [],
            "haar_minimum_d_plus_1_settings": [],
            "haar_overcomplete_2d_settings": [],
            "spectral_one_setting": [],
        }
        for _ in range(noise_trials[dimension]):
            states = _random_states(noise_rng, samples, dimension)
            observable = _random_traceless_observable(noise_rng, dimension)
            targets = np.real(
                np.einsum("bi,ij,bj->b", states.conj(), observable, states)
            )
            eigenvalues, eigenvectors = la.eigh(observable)
            banks = {
                "spectral_one_setting": [eigenvectors.conj().T],
                "complete_mub_d_plus_1_settings": mub,
                "haar_minimum_d_plus_1_settings": [
                    _haar_unitary(noise_rng, dimension)
                    for _ in range(dimension + 1)
                ],
                "haar_overcomplete_2d_settings": [
                    _haar_unitary(noise_rng, dimension)
                    for _ in range(2 * dimension)
                ],
            }
            for name, bank in banks.items():
                exact_features = _intensity_features(states, bank)
                readout = (
                    eigenvalues
                    if name == "spectral_one_setting"
                    else _observable_weights(bank, observable)
                )
                noisy = _noisy_features(
                    noise_rng,
                    exact_features,
                    dimension,
                    len(bank),
                    total_detected_photoelectrons,
                    read_noise_electrons_rms,
                )
                errors[name].append(
                    _normalized_rmse(noisy @ readout, targets)
                )

        mub_condition = _measurement_frame_condition(mub)
        centered_mub_condition = _centered_frame_condition(mub)
        expected_condition = float(np.sqrt(dimension + 1))
        if abs(mub_condition - expected_condition) > 1e-9:
            raise AssertionError(f"MUB frame spectrum failed at d={dimension}")
        if abs(centered_mub_condition - 1.0) > 1e-9:
            raise AssertionError(f"centered MUB is not tight at d={dimension}")
        if not all(np.isfinite(minimum_conditions + overcomplete_conditions)):
            raise AssertionError(f"a sampled Haar bank was rank deficient at d={dimension}")

        results_by_dimension[str(dimension)] = {
            "condition_trials_per_random_architecture": condition_trials[dimension],
            "frame_condition_number": {
                "complete_mub": mub_condition,
                "complete_mub_centered_traceless_subspace": (
                    centered_mub_condition
                ),
                "complete_mub_closed_form_including_identity": (
                    expected_condition
                ),
                "haar_minimum_d_plus_1": _summary(minimum_conditions),
                "haar_overcomplete_2d": _summary(overcomplete_conditions),
                "sampled_haar_full_rank_fraction": {
                    "d_plus_1_settings": 1.0,
                    "2d_settings": 1.0,
                },
            },
            "mub_validation": mub_validation,
            "noise_trials_per_architecture": noise_trials[dimension],
            "normalized_score_rmse": {
                name: _summary(values) for name, values in errors.items()
            },
        }

    diagnostics: dict[str, object] = {
        "schema_version": 1,
        "scope": (
            "generic-unitary scaling audit only; the complete MUB bases are not "
            "compiled into, or claimed reachable by, the fixed Burau-derived "
            "two-mode block library"
        ),
        "dimensions_tested": list(dimensions),
        "condition_number_convention": (
            "sigma_max/sigma_min of the real Hilbert--Schmidt coordinate matrix "
            "whose rows are rank-one output projectors; the reported primary "
            "condition includes the identity direction and uses singular-value "
            "rank threshold 1e-10"
        ),
        "mub_construction": {
            "dimension_4": (
                "two-qubit stabilizer bases from commuting Pauli classes "
                "{ZI,IZ}, {XI,IX}, {YI,IY}, {XY,YZ}, {XZ,ZY}"
            ),
            "odd_prime_dimensions": [
                dimension for dimension in dimensions if dimension != 4
            ],
            "odd_prime_formula": (
                "computational basis plus rows d^(-1/2) exp[2 pi i "
                "(a n^2 + b n)/d], with a,b,n in Z_d"
            ),
        },
        "condition_seed": condition_seed,
        "noise_seed": noise_seed,
        "noise_model": {
            "input_states": (
                "normalized complex-Gaussian vectors (Haar-distributed pure "
                "states)"
            ),
            "observables": (
                "independent random traceless Hermitian matrices with unit "
                "Frobenius norm"
            ),
            "read_noise_electrons_rms_per_output": read_noise_electrons_rms,
            "samples_per_trial": samples,
            "total_detected_photoelectrons_per_inference": (
                total_detected_photoelectrons
            ),
            "detector_model": (
                "Poisson counts plus additive Gaussian read noise; total detected "
                "budget divided equally over settings; each setting normalized "
                "to unit summed intensity after clipping"
            ),
            "budget_scope": (
                "post-fanout, post-coupling, post-mesh detected electrons; source "
                "power and component insertion loss are not modeled"
            ),
        },
        "results_by_dimension": results_by_dimension,
        "resource_scaling": {
            "task_specific_spectral": (
                "one generic d-mode unitary, d detectors, d(d-1)/2 generic "
                "two-mode cells, and O(d) path depth"
            ),
            "parallel_generic_minimum_bank": (
                "d+1 settings and d(d+1) detectors; d+1 independently "
                "instantiated universal meshes contain (d+1)d(d-1)/2 generic "
                "two-mode cells, while specializing the computational-basis "
                "branch as a direct detector path reduces this to d^2(d-1)/2"
            ),
            "time_multiplexed_generic_minimum_bank": (
                "one reusable O(d^2)-cell universal mesh and d detectors, at the "
                "cost of d+1 sequential settings and mesh reprogramming"
            ),
            "structured_mub_opportunity": (
                "the odd-prime chirp bases factor into a Fourier transform and "
                "basis-dependent diagonal phases, so a specialized implementation "
                "need not pay the independent-generic-mesh cell count"
            ),
        },
        "interpretation": (
            "Complete MUBs are tight on the traceless Hermitian subspace and "
            "remain well conditioned as d grows. Random banks at the rank-bound "
            "minimum are generically full rank in this sample but develop severe, "
            "heavy-tailed noise amplification. Under the declared fixed detected-"
            "light model, the spectral scorer follows approximately sqrt(d/N) "
            "normalized error and the complete MUB bank approximately d/sqrt(N)."
        ),
        "limitations": [
            "No Burau-derived compilation is performed or implied.",
            "Complete mutually unbiased bases are guaranteed in prime-power "
            "dimensions, not in every dimension; only the listed constructions "
            "are implemented here.",
            "The detector budget is fixed after optical losses and therefore is "
            "not an end-to-end source-energy comparison.",
            "The model omits component loss, detector gain mismatch, quantization, "
            "laser noise, thermal crosstalk, and transfer-matrix drift.",
            "The random-bank tails are finite deterministic Monte Carlo samples, "
            "not an asymptotic theorem.",
        ],
    }

    _plot_results(diagnostics)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime = time.perf_counter() - start
    print(f"Wrote {FIGURE_PATH.relative_to(Path.cwd())}")
    print(f"Wrote {RESULTS_PATH.relative_to(Path.cwd())}")
    print(f"Inference scaling runtime: {runtime:.3f} seconds")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return diagnostics


if __name__ == "__main__":
    run_inference_scaling()
