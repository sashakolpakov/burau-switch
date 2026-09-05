"""Digital twin for a bright-light passive Burau feature processor.

The physical model is a classical coherent field transformed by lossless
multiport matrices and measured by ordinary square-law photodiodes.  Qiskit,
when installed, is used only as an independent statevector implementation of
the same normalized modal amplitudes; no single-photon hardware is assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import numpy.linalg as la

from programs.inference.reproduce import (
    DEFAULT_MESH_DEPTH,
    _burau_mesh,
    _frame_condition_number,
    _hermitian_coordinates,
    _intensity_features,
    _measurement_coordinates,
    _random_hermitian,
    _random_states,
)


HERE = Path(__file__).resolve().parent
FIGURE_PATH = HERE / "figures" / "digital_twin.png"
RESULTS_PATH = HERE / "results" / "digital_twin.json"
MODEL_PATH = HERE / "results" / "digital_twin_model.json"


def _qiskit_probabilities(
    input_amplitudes: np.ndarray, unitary: np.ndarray
) -> tuple[np.ndarray | None, str | None]:
    """Return Qiskit statevector probabilities, or None if Qiskit is absent."""
    try:
        import qiskit
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except ModuleNotFoundError as error:
        if error.name is not None and error.name.startswith("qiskit"):
            return None, None
        raise

    dimension = len(input_amplitudes)
    number_of_qubits = int(round(np.log2(dimension)))
    if 2**number_of_qubits != dimension:
        raise ValueError("Qiskit amplitude encoding requires a power-of-two mode count")

    circuit = QuantumCircuit(number_of_qubits)
    circuit.initialize(input_amplitudes, circuit.qubits)
    circuit.unitary(unitary, circuit.qubits, label="Burau mesh")
    probabilities = np.asarray(
        Statevector.from_instruction(circuit).probabilities(), dtype=float
    )
    return probabilities, qiskit.__version__


def _observable_weights(
    bank: list[np.ndarray], observable: np.ndarray
) -> np.ndarray:
    measurement_coordinates = _measurement_coordinates(bank)
    observable_coordinates = _hermitian_coordinates(observable)
    return la.lstsq(
        measurement_coordinates.T, observable_coordinates, rcond=1e-12
    )[0]


def _normalized_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean((predictions - targets) ** 2)) / np.std(targets)
    )


def _noisy_detector_features(
    exact_features: np.ndarray,
    number_of_modes: int,
    number_of_settings: int,
    total_photoelectrons: float,
    read_noise_electrons: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate parallel direct detection with a fixed total light budget.

    The total detected photoelectron budget is divided equally among settings.
    Each branch is power-normalized after Poisson shot noise and additive
    Gaussian read noise.  The chosen count levels describe bright coherent
    operation, not a single-photon experiment.
    """
    expected = (
        total_photoelectrons / number_of_settings
    ) * exact_features.reshape(-1, number_of_settings, number_of_modes)
    counts = rng.poisson(expected).astype(float)
    counts += rng.normal(scale=read_noise_electrons, size=counts.shape)
    counts = np.clip(counts, 0.0, None)
    totals = np.sum(counts, axis=2, keepdims=True)
    normalized = counts / np.maximum(totals, 1.0)
    return normalized.reshape(len(exact_features), -1)


def _unitary_perturbation(
    rng: np.random.Generator, dimension: int, scale: float
) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    generator = 0.5 * (matrix + matrix.conj().T)
    generator /= la.norm(generator, 2)
    eigenvalues, eigenvectors = la.eigh(generator)
    return (eigenvectors * np.exp(1j * scale * eigenvalues)) @ eigenvectors.conj().T


def _drifted_score_error(
    *,
    bank: list[np.ndarray],
    weights: np.ndarray,
    states: np.ndarray,
    targets: np.ndarray,
    perturbation_scale: float,
    trials: int,
    rng: np.random.Generator,
) -> float:
    errors = []
    dimension = bank[0].shape[0]
    for _ in range(trials):
        drifted_bank = [
            _unitary_perturbation(rng, dimension, perturbation_scale) @ unitary
            for unitary in bank
        ]
        predictions = _intensity_features(states, drifted_bank) @ weights
        errors.append(_normalized_rmse(predictions, targets))
    return float(np.mean(errors))


def run_digital_twin(
    *,
    number_of_modes: int = 4,
    maximum_settings: int = 8,
    samples: int = 2000,
    drift_trials: int = 24,
    read_noise_electrons: float = 5.0,
    mesh_depth: int = DEFAULT_MESH_DEPTH,
    seed: int = 314159,
) -> dict[str, object]:
    """Run ideal, detector-noise, transfer-drift, and Qiskit cross-checks."""
    if number_of_modes != 4:
        raise ValueError("the prototype currently targets a four-mode device")
    if maximum_settings < number_of_modes + 1:
        raise ValueError("at least d + 1 settings are needed")
    if mesh_depth < 1:
        raise ValueError("mesh_depth must be positive")

    rng = np.random.default_rng(seed)
    bank = [
        _burau_mesh(rng, number_of_modes, mesh_depth)
        for _ in range(maximum_settings)
    ]
    representative_input = _random_states(rng, 1, number_of_modes)[0]
    classical_probabilities = np.abs(bank[0] @ representative_input) ** 2
    qiskit_probabilities, qiskit_version = _qiskit_probabilities(
        representative_input, bank[0]
    )
    qiskit_maximum_error = (
        None
        if qiskit_probabilities is None
        else float(np.max(np.abs(qiskit_probabilities - classical_probabilities)))
    )

    states = _random_states(rng, samples, number_of_modes)
    observable = _random_hermitian(rng, number_of_modes)
    targets = np.real(
        np.einsum("bi,ij,bj->b", states.conj(), observable, states)
    )
    spectral_weights, spectral_vectors = la.eigh(observable)
    task_compiled_unitary = spectral_vectors.conj().T
    task_compiled_bank = [task_compiled_unitary]
    task_compiled_predictions = (
        _intensity_features(states, task_compiled_bank) @ spectral_weights
    )
    task_compiled_ideal_error = _normalized_rmse(
        task_compiled_predictions, targets
    )
    task_compiled_reconstruction_error = float(
        la.norm(
            _measurement_coordinates(task_compiled_bank).T @ spectral_weights
            - _hermitian_coordinates(observable)
        )
    )

    setting_counts = list(range(number_of_modes + 1, maximum_settings + 1))
    conditions: dict[str, float] = {}
    weights_by_count: dict[int, np.ndarray] = {}
    ideal_errors: dict[str, float] = {}
    observable_reconstruction_errors: dict[str, float] = {}
    for count in setting_counts:
        selected_bank = bank[:count]
        weights = _observable_weights(selected_bank, observable)
        weights_by_count[count] = weights
        conditions[str(count)] = _frame_condition_number(selected_bank)
        observable_reconstruction_errors[str(count)] = float(
            la.norm(
                _measurement_coordinates(selected_bank).T @ weights
                - _hermitian_coordinates(observable)
            )
        )
        predictions = _intensity_features(states, selected_bank) @ weights
        ideal_errors[str(count)] = _normalized_rmse(predictions, targets)

    compared_setting_counts = sorted({number_of_modes + 1, maximum_settings})
    total_photoelectron_levels = [1e4, 1e5, 1e6, 1e7]
    detector_errors: dict[str, list[float]] = {}
    for count in compared_setting_counts:
        exact_features = _intensity_features(states, bank[:count])
        detector_errors[str(count)] = []
        for total_photoelectrons in total_photoelectron_levels:
            noisy_features = _noisy_detector_features(
                exact_features,
                number_of_modes,
                count,
                total_photoelectrons,
                read_noise_electrons,
                rng,
            )
            predictions = noisy_features @ weights_by_count[count]
            detector_errors[str(count)].append(
                _normalized_rmse(predictions, targets)
            )
    task_compiled_exact_features = _intensity_features(
        states, task_compiled_bank
    )
    task_compiled_detector_errors = []
    for total_photoelectrons in total_photoelectron_levels:
        noisy_features = _noisy_detector_features(
            task_compiled_exact_features,
            number_of_modes,
            1,
            total_photoelectrons,
            read_noise_electrons,
            rng,
        )
        task_compiled_detector_errors.append(
            _normalized_rmse(noisy_features @ spectral_weights, targets)
        )

    perturbation_scales = [0.0, 1e-4, 1e-3, 1e-2, 5e-2]
    drift_errors: dict[str, list[float]] = {}
    drift_states = states[: min(500, samples)]
    drift_targets = targets[: len(drift_states)]
    for count in compared_setting_counts:
        drift_errors[str(count)] = [
            _drifted_score_error(
                bank=bank[:count],
                weights=weights_by_count[count],
                states=drift_states,
                targets=drift_targets,
                perturbation_scale=scale,
                trials=drift_trials,
                rng=rng,
            )
            for scale in perturbation_scales
        ]
    task_compiled_drift_errors = [
        _drifted_score_error(
            bank=task_compiled_bank,
            weights=spectral_weights,
            states=drift_states,
            targets=drift_targets,
            perturbation_scale=scale,
            trials=drift_trials,
            rng=rng,
        )
        for scale in perturbation_scales
    ]

    diagnostics: dict[str, object] = {
        "physical_regime": (
            "bright classical coherent fields, passive unitary meshes, and "
            "ordinary square-law photodiodes"
        ),
        "single_photon_components_required": False,
        "qiskit_role": (
            "independent statevector check of normalized modal probabilities; "
            "not a requirement for the physical device"
        ),
        "qiskit_available": qiskit_probabilities is not None,
        "qiskit_version": qiskit_version,
        "qiskit_maximum_probability_error": qiskit_maximum_error,
        "number_of_modes": number_of_modes,
        "maximum_settings": maximum_settings,
        "burau_blocks_per_mesh": mesh_depth,
        "samples": samples,
        "seed": seed,
        "parallel_output_photodiodes": maximum_settings * number_of_modes,
        "ideal_equal_fanout_loss_db": float(10.0 * np.log10(maximum_settings)),
        "task_compiled_spectral_design": {
            "settings": 1,
            "output_photodiodes": number_of_modes,
            "ideal_equal_fanout_loss_db": 0.0,
            "ideal_normalized_score_rmse": task_compiled_ideal_error,
            "observable_reconstruction_error": (
                task_compiled_reconstruction_error
            ),
            "burau_compilation_status": (
                "generic spectral unitary baseline; exact compilation into "
                "the restricted Burau-block family is not yet established"
            ),
        },
        "condition_number_by_setting_count": conditions,
        "observable_reconstruction_error_by_setting_count": (
            observable_reconstruction_errors
        ),
        "ideal_normalized_score_rmse_by_setting_count": ideal_errors,
        "detector_noise_model": {
            "fixed_total_photoelectrons_per_inference": total_photoelectron_levels,
            "read_noise_electrons_rms_per_output": read_noise_electrons,
            "normalized_score_rmse_by_setting_count": detector_errors,
            "task_compiled_one_mesh_normalized_score_rmse": (
                task_compiled_detector_errors
            ),
        },
        "coherent_transfer_drift_model": {
            "effective_unitary_perturbation_scales": perturbation_scales,
            "trials_per_scale": drift_trials,
            "normalized_score_rmse_by_setting_count": drift_errors,
            "task_compiled_one_mesh_normalized_score_rmse": (
                task_compiled_drift_errors
            ),
        },
    }

    if ideal_errors[str(maximum_settings)] > 1e-10:
        raise AssertionError("ideal digital twin failed to recover its quadratic score")
    if max(observable_reconstruction_errors.values()) > 1e-10:
        raise AssertionError("exported readout weights do not reconstruct the observable")
    if task_compiled_reconstruction_error > 1e-10:
        raise AssertionError("spectral one-mesh design does not reconstruct the score")
    if qiskit_maximum_error is not None and qiskit_maximum_error > 1e-12:
        raise AssertionError("Qiskit and coherent-field probabilities disagree")

    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.5))
    output_modes = np.arange(number_of_modes)
    width = 0.36
    axes[0, 0].bar(
        output_modes - width / 2,
        classical_probabilities,
        width,
        label="coherent-field model",
        color="#7f3c8d",
    )
    comparison = (
        classical_probabilities
        if qiskit_probabilities is None
        else qiskit_probabilities
    )
    comparison_label = (
        "NumPy exact cross-check"
        if qiskit_probabilities is None
        else f"Qiskit {qiskit_version} statevector"
    )
    axes[0, 0].bar(
        output_modes + width / 2,
        comparison,
        width,
        label=comparison_label,
        color="#1b9e77",
    )
    axes[0, 0].set_xticks(output_modes)
    axes[0, 0].set_xlabel("output mode")
    axes[0, 0].set_ylabel("normalized intensity")
    axes[0, 0].set_title("(a) Independent ideal-model agreement")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].semilogy(
        setting_counts,
        [conditions[str(count)] for count in setting_counts],
        marker="o",
        color="#7f3c8d",
    )
    axes[0, 1].set_xlabel("parallel mesh settings")
    axes[0, 1].set_ylabel("measurement-frame condition number")
    axes[0, 1].set_title("(b) Redundancy improves calibration")

    axes[1, 0].loglog(
        total_photoelectron_levels,
        task_compiled_detector_errors,
        marker="o",
        label="spectral 1-mesh baseline",
        color="#4c78a8",
    )
    for count, color in zip(compared_setting_counts, ("#d95f02", "#1b9e77")):
        axes[1, 0].loglog(
            total_photoelectron_levels,
            detector_errors[str(count)],
            marker="o",
            label=f"fixed {count}-mesh bank",
            color=color,
        )
    axes[1, 0].set_xlabel("total detected photoelectrons / inference")
    axes[1, 0].set_ylabel("normalized score RMSE")
    axes[1, 0].set_title("(c) Bright-light detector budget")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].loglog(
        perturbation_scales[1:],
        task_compiled_drift_errors[1:],
        marker="o",
        label="spectral 1-mesh baseline",
        color="#4c78a8",
    )
    for count, color in zip(compared_setting_counts, ("#d95f02", "#1b9e77")):
        axes[1, 1].loglog(
            perturbation_scales[1:],
            drift_errors[str(count)][1:],
            marker="o",
            label=f"fixed {count}-mesh bank",
            color=color,
        )
    axes[1, 1].set_xlabel("effective coherent perturbation scale")
    axes[1, 1].set_ylabel("normalized score RMSE")
    axes[1, 1].set_title("(d) Unrecalibrated transfer drift")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Digital twin of a classical passive Burau feature processor",
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
    hardware_model = {
        "schema_version": 1,
        "mode_basis_order": list(range(number_of_modes)),
        "transfer_convention": "output_amplitudes = unitary @ input_amplitudes",
        "input_normalization": "unit Euclidean norm; carry norm squared separately",
        "burau_mesh_unitaries": [
            {
                "real": np.real(unitary).tolist(),
                "imaginary": np.imag(unitary).tolist(),
            }
            for unitary in bank
        ],
        "representative_input": {
            "real": np.real(representative_input).tolist(),
            "imaginary": np.imag(representative_input).tolist(),
        },
        "quadratic_observable": {
            "real": np.real(observable).tolist(),
            "imaginary": np.imag(observable).tolist(),
        },
        "electronic_readout_weights_by_setting_count": {
            str(count): weights_by_count[count].tolist()
            for count in setting_counts
        },
        "task_compiled_spectral_unitary": {
            "real": np.real(task_compiled_unitary).tolist(),
            "imaginary": np.imag(task_compiled_unitary).tolist(),
        },
        "task_compiled_spectral_readout_weights": spectral_weights.tolist(),
    }
    MODEL_PATH.write_text(
        json.dumps(hardware_model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {FIGURE_PATH.relative_to(Path.cwd())}")
    print(f"Wrote {RESULTS_PATH.relative_to(Path.cwd())}")
    print(f"Wrote {MODEL_PATH.relative_to(Path.cwd())}")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return diagnostics


if __name__ == "__main__":
    run_digital_twin()
