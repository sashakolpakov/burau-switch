"""Compile four-mode detector bases with restricted Burau--Squier blocks.

The primary short-depth compiler uses one independently tunable omega for every
primitive Burau generator.  That is an optimistic engineering ansatz, not a
single fixed gate library.  A constructive compiler separately synthesizes the
six cells of an exact Givens mesh from exponent-neutral words in one fixed 2x2
B3 Burau-derived block library.  Those blocks are embedded on successive mode
pairs; this is not a global four-mode Burau representation or one B_n word.  A
direct random shared-omega search is retained as a search-budget control, and
the analytic Givens mesh is the hardware-neutral baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import platform
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import numpy.linalg as la
import scipy
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial import cKDTree

from burau_switch import beta_generator, beta_word, positive_form, unitarize


HERE = Path(__file__).resolve().parent
TARGET_MODEL_PATH = HERE.parent / "digital_twin" / "results" / "digital_twin_model.json"
FIGURE_PATH = HERE / "figures" / "compiler.png"
RESULTS_PATH = HERE / "results" / "compiler.json"
MODEL_PATH = HERE / "results" / "compiled_mesh.json"

NUMBER_OF_MODES = 4
NEAREST_NEIGHBOR_PAIR_CYCLE = ((0, 1), (2, 3), (1, 2))
PRIMITIVE_LETTERS = ((1, 1), (1, -1), (2, 1), (2, -1))
APPLICATION_ORDER_CONVENTION = (
    "entries are listed first-to-last in temporal application order on a "
    "column vector, so the total matrix is B_last @ ... @ B_first; reverse a "
    "cell's application-order generator list before passing it to beta_word"
)
INDEPENDENT_DEPTHS = (8, 12, 16, 20)
SHARED_DEPTHS = (8, 12, 16, 24, 32)
OMEGA_MARGIN = 0.12
TARGET_SUCCESS_THRESHOLD = 1e-8
CONSTRUCTIVE_SHARED_OMEGA = float(np.sqrt(2.0))
CONSTRUCTIVE_HALF_WORD_DEPTH = 12
CONSTRUCTIVE_CANDIDATES_PER_CELL = 160
CONSTRUCTIVE_BEAM_WIDTH = 1000
CONSTRUCTIVE_QUERY_NEIGHBORS = 8
CONSTRUCTIVE_BASIS_SUCCESS_THRESHOLD = 1e-2
CONSTRUCTIVE_SCORE_SUCCESS_THRESHOLD = 1e-2
MITM_MATRIX_DEDUPLICATION_DECIMALS = 11


@dataclass(frozen=True)
class Layer:
    """One embedded primitive Burau letter."""

    pair: tuple[int, int]
    generator: int
    power: int = 1


def _complex_matrix(payload: dict[str, object]) -> np.ndarray:
    return np.asarray(payload["real"], dtype=float) + 1j * np.asarray(
        payload["imaginary"], dtype=float
    )


def _complex_payload(matrix: np.ndarray) -> dict[str, object]:
    return {
        "real": np.real(matrix).tolist(),
        "imaginary": np.imag(matrix).tolist(),
    }


def _load_target_model() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not TARGET_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"missing {TARGET_MODEL_PATH}; run "
            "python -m programs.digital_twin.reproduce first"
        )
    payload = json.loads(TARGET_MODEL_PATH.read_text(encoding="utf-8"))
    target = _complex_matrix(payload["task_compiled_spectral_unitary"])
    observable = _complex_matrix(payload["quadratic_observable"])
    weights = np.asarray(
        payload["task_compiled_spectral_readout_weights"], dtype=float
    )
    return target, observable, weights


def _architecture(depth: int) -> tuple[Layer, ...]:
    """Return the fixed nearest-neighbor architecture used by the compiler."""
    return tuple(
        Layer(
            pair=NEAREST_NEIGHBOR_PAIR_CYCLE[index % 3],
            generator=1 + index % 2,
        )
        for index in range(depth)
    )


def _burau_letter(generator: int, power: int, omega: float) -> np.ndarray:
    raw = beta_generator(generator, np.exp(0.5j * omega))
    if power == -1:
        raw = la.inv(raw)
    elif power != 1:
        raise ValueError("the compiler supports primitive powers +1 and -1")
    return unitarize(raw, positive_form(omega))


def _embed(block: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
    embedded = np.eye(NUMBER_OF_MODES, dtype=complex)
    indices = np.asarray(pair)
    embedded[np.ix_(indices, indices)] = block
    return embedded


def _evaluate_mesh(
    omegas: Sequence[float], architecture: Sequence[Layer]
) -> np.ndarray:
    if len(omegas) != len(architecture):
        raise ValueError("one omega is required for every independently tuned layer")
    result = np.eye(NUMBER_OF_MODES, dtype=complex)
    for omega, layer in zip(omegas, architecture):
        result = _embed(
            _burau_letter(layer.generator, layer.power, float(omega)),
            layer.pair,
        ) @ result
    return result


def _evaluate_shared_mesh(omega: float, codes: Sequence[int]) -> np.ndarray:
    """Evaluate a fixed-omega word; codes enumerate sigma_1/2 and inverses."""
    blocks = [
        _burau_letter(generator, power, omega)
        for generator, power in PRIMITIVE_LETTERS
    ]
    result = np.eye(NUMBER_OF_MODES, dtype=complex)
    for index, code in enumerate(codes):
        pair = NEAREST_NEIGHBOR_PAIR_CYCLE[index % 3]
        result = _embed(blocks[int(code)], pair) @ result
    return result


def _fixed_omega_blocks(omega: float) -> np.ndarray:
    """Return sigma_1, sigma_1^-1, sigma_2, sigma_2^-1 at one omega."""
    return np.asarray(
        [
            _burau_letter(generator, power, omega)
            for generator, power in PRIMITIVE_LETTERS
        ]
    )


def _enumerate_reduced_half_words(
    omega: float, maximum_depth: int
) -> dict[str, np.ndarray]:
    """Enumerate words through ``maximum_depth`` without adjacent inverses.

    ``matrices[index]`` follows the same left-action convention as the mesh:
    if its code row is ``(c_1, ..., c_k)``, the matrix is
    ``B[c_k] ... B[c_1]``.  Removing adjacent inverse pairs is exact and only
    avoids redundant representatives; the empty word is retained.
    """
    if maximum_depth < 1:
        raise ValueError("maximum_depth must be positive")

    number_of_words = 1 + 2 * (3**maximum_depth - 1)
    matrices = np.empty((number_of_words, 2, 2), dtype=complex)
    codes = np.full((number_of_words, maximum_depth), 255, dtype=np.uint8)
    lengths = np.zeros(number_of_words, dtype=np.uint8)
    exponent_sums = np.zeros(number_of_words, dtype=np.int8)
    last_codes = np.full(number_of_words, -1, dtype=np.int8)
    matrices[0] = np.eye(2, dtype=complex)

    blocks = _fixed_omega_blocks(omega)
    inverse_codes = np.asarray((1, 0, 3, 2), dtype=np.int8)
    exponent_steps = np.asarray((1, -1, 1, -1), dtype=np.int8)
    frontier = np.asarray((0,), dtype=np.int64)
    cursor = 1
    for depth in range(1, maximum_depth + 1):
        children: list[np.ndarray] = []
        for code in range(4):
            parents = frontier[last_codes[frontier] != inverse_codes[code]]
            count = len(parents)
            destination = np.arange(cursor, cursor + count, dtype=np.int64)
            matrices[destination] = blocks[code] @ matrices[parents]
            if depth > 1:
                codes[destination, : depth - 1] = codes[parents, : depth - 1]
            codes[destination, depth - 1] = code
            lengths[destination] = depth
            exponent_sums[destination] = (
                exponent_sums[parents] + exponent_steps[code]
            )
            last_codes[destination] = code
            children.append(destination)
            cursor += count
        frontier = np.concatenate(children)

    if cursor != number_of_words:
        raise AssertionError("reduced-word enumeration count is inconsistent")
    return {
        "matrices": matrices,
        "codes": codes,
        "lengths": lengths,
        "exponent_sums": exponent_sums,
    }


def _batch_matrix_coordinates(matrices: np.ndarray) -> np.ndarray:
    """Euclidean coordinates whose distance is the Frobenius distance."""
    return np.concatenate(
        (
            matrices.real.reshape(len(matrices), -1),
            matrices.imag.reshape(len(matrices), -1),
        ),
        axis=1,
    )


def _cell_word_candidates(
    target: np.ndarray,
    library: dict[str, np.ndarray],
    grouped_indices: dict[int, np.ndarray],
    grouped_trees: dict[int, cKDTree],
    *,
    number_of_candidates: int,
    query_neighbors: int,
) -> list[dict[str, object]]:
    """Meet in the middle for determinant-one approximants of one SU(2) cell."""
    matrices = library["matrices"]
    codes = library["codes"]
    lengths = library["lengths"]
    exponent_sums = library["exponent_sums"]
    candidate_errors: list[np.ndarray] = []
    candidate_prefixes: list[np.ndarray] = []
    candidate_suffixes: list[np.ndarray] = []

    for exponent_sum, prefix_indices in grouped_indices.items():
        suffix_indices = grouped_indices.get(-exponent_sum)
        suffix_tree = grouped_trees.get(-exponent_sum)
        if suffix_indices is None or suffix_tree is None:
            continue
        prefix_matrices = matrices[prefix_indices]
        desired_suffixes = target[None, :, :] @ np.swapaxes(
            prefix_matrices.conj(), 1, 2
        )
        neighbors = min(query_neighbors, len(suffix_indices))
        _, local_neighbors = suffix_tree.query(
            _batch_matrix_coordinates(desired_suffixes),
            k=neighbors,
            workers=1,
        )
        if neighbors == 1:
            local_neighbors = np.asarray(local_neighbors)[:, None]
        prefix_grid = np.broadcast_to(
            prefix_indices[:, None], local_neighbors.shape
        )
        suffix_grid = suffix_indices[local_neighbors]
        combined = matrices[suffix_grid] @ matrices[prefix_grid]
        errors = la.norm(combined - target, axis=(2, 3)) / np.sqrt(2.0)
        flat_errors = errors.ravel()
        # Keep an oversampled pool here: distinct braid words can represent the
        # same matrix through exact braid relations, and the beam needs matrix
        # diversity rather than many algebraically equivalent factorizations.
        keep = min(4 * number_of_candidates, len(flat_errors))
        if keep < len(flat_errors):
            selected = np.argpartition(flat_errors, keep - 1)[:keep]
        else:
            selected = np.arange(keep)
        candidate_errors.append(flat_errors[selected])
        candidate_prefixes.append(prefix_grid.ravel()[selected])
        candidate_suffixes.append(suffix_grid.ravel()[selected])

    errors = np.concatenate(candidate_errors)
    prefixes = np.concatenate(candidate_prefixes)
    suffixes = np.concatenate(candidate_suffixes)
    distinct_candidates: dict[bytes, dict[str, object]] = {}
    for index in range(len(errors)):
        prefix = int(prefixes[index])
        suffix = int(suffixes[index])
        matrix = matrices[suffix] @ matrices[prefix]
        matrix_key = np.round(
            _batch_matrix_coordinates(matrix[None]),
            MITM_MATRIX_DEDUPLICATION_DECIMALS,
        ).tobytes()
        prefix_length = int(lengths[prefix])
        suffix_length = int(lengths[suffix])
        word_codes = np.concatenate(
            (codes[prefix, :prefix_length], codes[suffix, :suffix_length])
        ).astype(int)
        exponent_sum = int(exponent_sums[prefix] + exponent_sums[suffix])
        candidate: dict[str, object] = {
            "matrix": matrix,
            "letter_codes": word_codes,
            "exponent_sum": exponent_sum,
            "normalized_cell_frobenius_error": float(errors[index]),
        }
        previous = distinct_candidates.get(matrix_key)
        if previous is None or len(word_codes) < len(previous["letter_codes"]):
            distinct_candidates[matrix_key] = candidate
        elif (
            len(word_codes) == len(previous["letter_codes"])
            and float(errors[index])
            < float(previous["normalized_cell_frobenius_error"])
        ):
            distinct_candidates[matrix_key] = candidate

    candidates = sorted(
        distinct_candidates.values(),
        key=lambda candidate: float(candidate["normalized_cell_frobenius_error"]),
    )[:number_of_candidates]
    if len(candidates) < number_of_candidates:
        raise AssertionError(
            "meet-in-the-middle search retained too few distinct cells"
        )
    return candidates


def _freely_reduce_codes(codes: Sequence[int]) -> np.ndarray:
    """Cancel adjacent generator/inverse pairs without changing the word."""
    inverse_codes = (1, 0, 3, 2)
    reduced: list[int] = []
    for code_value in codes:
        code = int(code_value)
        if reduced and reduced[-1] == inverse_codes[code]:
            reduced.pop()
        else:
            reduced.append(code)
    return np.asarray(reduced, dtype=int)


def _batch_basis_error(unitaries: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Vectorized ordered detector-basis error for a matrix batch."""
    overlaps = np.sum(unitaries.conj() * target, axis=-1)
    squared = 1.0 - np.mean(np.abs(overlaps) ** 2, axis=-1)
    return np.sqrt(np.maximum(0.0, squared))


def _constructive_shared_omega_compile(
    target: np.ndarray,
    generic: dict[str, object],
    *,
    omega: float,
    half_word_depth: int,
    candidates_per_cell: int,
    beam_width: int,
    query_neighbors: int,
) -> dict[str, object]:
    """Compile six SU(2) Givens cells into words at one shared omega.

    Each meet-in-the-middle word has total Artin exponent zero.  Since all four
    primitive blocks have the same determinant phase up to inversion, this
    enforces determinant one and lets the word approximate an SU(2) Givens
    cell without borrowing an unmodeled per-cell phase.
    """
    if not 0.0 < omega < 2.0 * np.pi / 3.0:
        raise ValueError("constructive omega lies outside the definite branch")
    library = _enumerate_reduced_half_words(omega, half_word_depth)
    exponent_sums = library["exponent_sums"]
    library_coordinates = np.round(
        _batch_matrix_coordinates(library["matrices"]),
        MITM_MATRIX_DEDUPLICATION_DECIMALS,
    )
    grouped_indices = {}
    for value in np.unique(exponent_sums):
        raw_indices = np.flatnonzero(exponent_sums == value)
        _, first_occurrences = np.unique(
            library_coordinates[raw_indices], axis=0, return_index=True
        )
        grouped_indices[int(value)] = raw_indices[np.sort(first_occurrences)]
    grouped_trees = {
        exponent_sum: cKDTree(
            _batch_matrix_coordinates(library["matrices"][indices]),
            balanced_tree=True,
            compact_nodes=True,
        )
        for exponent_sum, indices in grouped_indices.items()
    }

    ideal_cells = generic["triangular_givens_cells"]
    candidates_by_cell = []
    for cell in ideal_cells:
        block = _complex_matrix(cell["unitary"])
        determinant_error = abs(la.det(block) - 1.0)
        if determinant_error > 1e-10:
            raise AssertionError("the constructive target cell is not in SU(2)")
        candidates_by_cell.append(
            _cell_word_candidates(
                block,
                library,
                grouped_indices,
                grouped_trees,
                number_of_candidates=candidates_per_cell,
                query_neighbors=query_neighbors,
            )
        )

    beam_meshes = np.eye(NUMBER_OF_MODES, dtype=complex)[None, :, :]
    beam_paths = np.empty((1, 0), dtype=np.int16)
    ideal_prefix = np.eye(NUMBER_OF_MODES, dtype=complex)
    for cell_index, (cell, candidates) in enumerate(
        zip(ideal_cells, candidates_by_cell)
    ):
        pair = tuple(int(value) for value in cell["pair"])
        ideal_prefix = _embed(_complex_matrix(cell["unitary"]), pair) @ ideal_prefix
        candidate_matrices = np.asarray(
            [candidate["matrix"] for candidate in candidates]
        )
        embedded = np.broadcast_to(
            np.eye(NUMBER_OF_MODES, dtype=complex),
            (len(candidates), NUMBER_OF_MODES, NUMBER_OF_MODES),
        ).copy()
        embedded[:, pair[0], pair[0]] = candidate_matrices[:, 0, 0]
        embedded[:, pair[0], pair[1]] = candidate_matrices[:, 0, 1]
        embedded[:, pair[1], pair[0]] = candidate_matrices[:, 1, 0]
        embedded[:, pair[1], pair[1]] = candidate_matrices[:, 1, 1]
        expanded = embedded[:, None, :, :] @ beam_meshes[None, :, :, :]
        errors = _batch_basis_error(expanded, ideal_prefix).ravel()
        keep = min(beam_width, len(errors))
        if keep < len(errors):
            selected = np.argpartition(errors, keep - 1)[:keep]
        else:
            selected = np.arange(keep)
        selected = selected[np.argsort(errors[selected], kind="stable")]
        parent_indices = selected % len(beam_meshes)
        candidate_indices = selected // len(beam_meshes)
        beam_meshes = expanded.reshape(-1, NUMBER_OF_MODES, NUMBER_OF_MODES)[
            selected
        ]
        beam_paths = np.column_stack(
            (beam_paths[parent_indices], candidate_indices.astype(np.int16))
        )
        if beam_paths.shape[1] != cell_index + 1:
            raise AssertionError("constructive beam path bookkeeping failed")

    final_errors = np.asarray([_basis_error(mesh, target) for mesh in beam_meshes])
    best_index = int(np.argmin(final_errors))
    mesh = beam_meshes[best_index]
    selected_path = beam_paths[best_index]
    serialized_cells = []
    total_letters = 0
    maximum_cell_unitarity_error = 0.0
    maximum_serialized_word_replay_error = 0.0
    maximum_beta_word_algebraic_replay_error = 0.0
    replayed_mesh = np.eye(NUMBER_OF_MODES, dtype=complex)
    primitive_blocks = _fixed_omega_blocks(omega)
    for cell, candidates, candidate_index in zip(
        ideal_cells, candidates_by_cell, selected_path
    ):
        candidate = candidates[int(candidate_index)]
        word_codes = _freely_reduce_codes(candidate["letter_codes"])
        reduced_exponent_sum = int(
            sum(1 if code % 2 == 0 else -1 for code in word_codes)
        )
        if reduced_exponent_sum != int(candidate["exponent_sum"]):
            raise AssertionError("free reduction changed the Artin exponent sum")
        total_letters += len(word_codes)
        cell_matrix = np.asarray(candidate["matrix"])
        maximum_cell_unitarity_error = max(
            maximum_cell_unitarity_error,
            float(la.norm(cell_matrix.conj().T @ cell_matrix - np.eye(2))),
        )
        replayed_cell = np.eye(2, dtype=complex)
        for code in word_codes:
            replayed_cell = primitive_blocks[int(code)] @ replayed_cell
        replay_error = float(la.norm(replayed_cell - cell_matrix))
        maximum_serialized_word_replay_error = max(
            maximum_serialized_word_replay_error, replay_error
        )
        pair = tuple(int(value) for value in cell["pair"])
        replayed_mesh = _embed(replayed_cell, pair) @ replayed_mesh
        application_order_word = [
            [int(PRIMITIVE_LETTERS[code][0]), int(PRIMITIVE_LETTERS[code][1])]
            for code in word_codes
        ]
        algebraic_word = list(reversed(application_order_word))
        algebraic_replay = unitarize(
            beta_word(
                [(int(generator), int(power)) for generator, power in algebraic_word],
                np.exp(0.5j * omega),
            ),
            positive_form(omega),
        )
        algebraic_replay_error = float(la.norm(algebraic_replay - cell_matrix))
        maximum_beta_word_algebraic_replay_error = max(
            maximum_beta_word_algebraic_replay_error, algebraic_replay_error
        )
        serialized_cells.append(
            {
                "pair": [int(value) for value in cell["pair"]],
                "braid_letters": len(word_codes),
                "application_order_letter_codes": word_codes.tolist(),
                "application_order_local_generators": application_order_word,
                "beta_word_compatible_algebraic_word": algebraic_word,
                "word_evaluation_convention": APPLICATION_ORDER_CONVENTION,
                "artin_exponent_sum": reduced_exponent_sum,
                "serialized_word_replay_error": replay_error,
                "beta_word_algebraic_replay_error": algebraic_replay_error,
                "normalized_cell_frobenius_error": float(
                    candidate["normalized_cell_frobenius_error"]
                ),
                "ideal_givens_unitary": cell["unitary"],
                "compiled_word_unitary": _complex_payload(cell_matrix),
            }
        )

    return {
        "mesh": mesh,
        "omega": omega,
        "omega_branch": "0 < omega < 2*pi/3",
        "squier_form_condition_number": float(la.cond(positive_form(omega))),
        "half_word_depth": half_word_depth,
        "maximum_cell_word_length": 2 * half_word_depth,
        "reduced_half_word_library_size": int(len(library["matrices"])),
        "within_exponent_distinct_half_word_matrices": int(
            sum(len(indices) for indices in grouped_indices.values())
        ),
        "query_neighbors_per_prefix": query_neighbors,
        "candidates_retained_per_givens_cell": candidates_per_cell,
        "beam_width": beam_width,
        "matrix_deduplication_decimal_places": (
            MITM_MATRIX_DEDUPLICATION_DECIMALS
        ),
        "logical_givens_cells": len(serialized_cells),
        "total_braid_letters": total_letters,
        "conservative_sequential_pair_layers": total_letters,
        "maximum_cell_unitarity_error": maximum_cell_unitarity_error,
        "maximum_serialized_word_replay_error": (
            maximum_serialized_word_replay_error
        ),
        "maximum_beta_word_algebraic_replay_error": (
            maximum_beta_word_algebraic_replay_error
        ),
        "full_mesh_serialized_word_replay_error": float(
            la.norm(replayed_mesh - mesh)
        ),
        "cells": serialized_cells,
    }


def _replay_constructive_cell_words(
    cells: Sequence[dict[str, object]], omega: float
) -> np.ndarray:
    """Replay serialized cell words using one fixed two-mode block library."""
    primitive_blocks = _fixed_omega_blocks(omega)
    mesh = np.eye(NUMBER_OF_MODES, dtype=complex)
    for cell in cells:
        block = np.eye(2, dtype=complex)
        for code in cell["application_order_letter_codes"]:
            block = primitive_blocks[int(code)] @ block
        mesh = _embed(block, tuple(int(value) for value in cell["pair"])) @ mesh
    return mesh


def _isometric_hermitian_coordinates(matrix: np.ndarray) -> np.ndarray:
    coordinates: list[float] = [
        float(np.real(matrix[index, index]))
        for index in range(NUMBER_OF_MODES)
    ]
    for row in range(NUMBER_OF_MODES):
        for column in range(row + 1, NUMBER_OF_MODES):
            coordinates.extend(
                (
                    float(np.sqrt(2.0) * np.real(matrix[row, column])),
                    float(np.sqrt(2.0) * np.imag(matrix[row, column])),
                )
            )
    return np.asarray(coordinates)


def _projector_coordinates(unitary: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            _isometric_hermitian_coordinates(np.outer(row.conj(), row))
            for row in unitary
        ]
    )


def _basis_error(unitary: np.ndarray, target: np.ndarray) -> float:
    """Ordered detector-basis distance, invariant under output row phases."""
    differences = [
        np.outer(row.conj(), row) - np.outer(target_row.conj(), target_row)
        for row, target_row in zip(unitary, target)
    ]
    return float(
        np.sqrt(sum(la.norm(difference) ** 2 for difference in differences))
        / np.sqrt(2.0 * NUMBER_OF_MODES)
    )


def _align_output_phases(
    unitary: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    phases = np.asarray(
        [np.angle(np.vdot(row, target_row)) for row, target_row in zip(unitary, target)]
    )
    return np.exp(1j * phases[:, None]) * unitary, phases


def _fit_residual(
    parameters: np.ndarray,
    architecture: Sequence[Layer],
    target: np.ndarray,
) -> np.ndarray:
    depth = len(architecture)
    mesh = _evaluate_mesh(parameters[:depth], architecture)
    aligned = np.exp(1j * parameters[depth:, None]) * mesh
    difference = (aligned - target) / np.sqrt(NUMBER_OF_MODES)
    return np.concatenate((np.real(difference).ravel(), np.imag(difference).ravel()))


def _fit_independent_omegas(
    target: np.ndarray,
    depth: int,
    *,
    restarts: int,
    seed_base: int,
    maximum_evaluations: int,
    omega_margin: float,
) -> dict[str, object]:
    architecture = _architecture(depth)
    omega_minimum = omega_margin
    omega_maximum = 2.0 * np.pi / 3.0 - omega_margin
    lower = np.concatenate(
        (np.full(depth, omega_minimum), np.full(NUMBER_OF_MODES, -np.pi))
    )
    upper = np.concatenate(
        (np.full(depth, omega_maximum), np.full(NUMBER_OF_MODES, np.pi))
    )

    restart_results: list[dict[str, object]] = []
    best_result = None
    best_error = np.inf
    for restart in range(restarts):
        rng = np.random.default_rng(seed_base + depth * 100 + restart)
        initial = rng.uniform(lower, upper)
        fit = least_squares(
            _fit_residual,
            initial,
            args=(architecture, target),
            bounds=(lower, upper),
            max_nfev=maximum_evaluations,
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
        )
        mesh = _evaluate_mesh(fit.x[:depth], architecture)
        aligned = np.exp(1j * fit.x[depth:, None]) * mesh
        normalized_unitary_error = float(
            la.norm(aligned - target) / np.sqrt(NUMBER_OF_MODES)
        )
        basis_error = _basis_error(mesh, target)
        restart_results.append(
            {
                "restart": restart,
                "normalized_phase_aligned_unitary_error": normalized_unitary_error,
                "ordered_detector_basis_error": basis_error,
                "function_evaluations": int(fit.nfev),
                "first_order_optimality": float(fit.optimality),
                "optimizer_success": bool(fit.success),
            }
        )
        if basis_error < best_error:
            best_error = basis_error
            best_result = fit

    if best_result is None:
        raise AssertionError("the optimizer did not produce a candidate")
    omegas = np.asarray(best_result.x[:depth])
    mesh = _evaluate_mesh(omegas, architecture)
    aligned, analytic_phases = _align_output_phases(mesh, target)
    return {
        "depth": depth,
        "architecture": architecture,
        "omegas": omegas,
        "mesh": mesh,
        "output_alignment_phases": analytic_phases,
        "ordered_detector_basis_error": _basis_error(mesh, target),
        "normalized_phase_aligned_unitary_error": float(
            la.norm(aligned - target) / np.sqrt(NUMBER_OF_MODES)
        ),
        "successful_restarts": int(
            sum(
                row["ordered_detector_basis_error"] < TARGET_SUCCESS_THRESHOLD
                for row in restart_results
            )
        ),
        "restart_results": restart_results,
    }


def _score_metrics(
    mesh: np.ndarray,
    target: np.ndarray,
    observable: np.ndarray,
    states: np.ndarray,
) -> dict[str, object]:
    diagonalized = mesh @ observable @ mesh.conj().T
    fitted_weights = np.real(np.diag(diagonalized))
    reconstructed = mesh.conj().T @ np.diag(fitted_weights) @ mesh
    difference = reconstructed - observable
    targets = np.real(
        np.einsum("bi,ij,bj->b", states.conj(), observable, states)
    )
    predictions = np.real(
        np.einsum("bi,ij,bj->b", states.conj(), reconstructed, states)
    )
    aligned, _ = _align_output_phases(mesh, target)
    return {
        "ordered_detector_basis_error": _basis_error(mesh, target),
        "normalized_phase_aligned_unitary_error": float(
            la.norm(aligned - target) / np.sqrt(NUMBER_OF_MODES)
        ),
        "phase_aligned_process_infidelity": float(
            max(
                0.0,
                1.0
                - abs(np.trace(target.conj().T @ aligned)) ** 2
                / NUMBER_OF_MODES**2,
            )
        ),
        "relative_observable_frobenius_error": float(
            la.norm(difference) / la.norm(observable)
        ),
        "relative_observable_operator_error": float(
            la.norm(difference, 2) / la.norm(observable, 2)
        ),
        "held_out_normalized_score_rmse": float(
            np.sqrt(np.mean((predictions - targets) ** 2)) / np.std(targets)
        ),
        "fitted_readout_weights": fitted_weights.tolist(),
        "maximum_unitarity_error": float(
            la.norm(mesh.conj().T @ mesh - np.eye(NUMBER_OF_MODES))
        ),
    }


def _projector_jacobian_diagnostics(
    omegas: np.ndarray,
    architecture: Sequence[Layer],
    *,
    step: float = 1e-6,
) -> dict[str, object]:
    base = _projector_coordinates(_evaluate_mesh(omegas, architecture))
    jacobian = np.empty((len(base), len(omegas)))
    for index in range(len(omegas)):
        forward = omegas.copy()
        backward = omegas.copy()
        forward[index] += step
        backward[index] -= step
        jacobian[:, index] = (
            _projector_coordinates(_evaluate_mesh(forward, architecture))
            - _projector_coordinates(_evaluate_mesh(backward, architecture))
        ) / (2.0 * step)
    singular_values = la.svd(jacobian, compute_uv=False)
    threshold = singular_values[0] * 1e-8
    rank = int(np.sum(singular_values > threshold))
    nonzero = singular_values[:rank]
    return {
        "detector_basis_manifold_dimension": NUMBER_OF_MODES * (NUMBER_OF_MODES - 1),
        "finite_difference_step": step,
        "rank_threshold_relative_to_largest": 1e-8,
        "numerical_rank": rank,
        "singular_values": singular_values.tolist(),
        "nonzero_condition_number": float(nonzero[0] / nonzero[-1]),
        "smallest_nonzero_singular_value": float(nonzero[-1]),
    }


def _score_reconstruction_error(mesh: np.ndarray, observable: np.ndarray) -> float:
    transformed = mesh @ observable @ mesh.conj().T
    diagonal = np.diag(np.diag(transformed))
    return float(la.norm(transformed - diagonal) / la.norm(observable))


def _shared_omega_search(
    observable: np.ndarray,
    *,
    depths: Sequence[int],
    templates_per_depth: int,
    omega_grid_points: int,
    seed: int,
    omega_margin: float,
) -> list[dict[str, object]]:
    omega_minimum = omega_margin
    omega_maximum = 2.0 * np.pi / 3.0 - omega_margin
    omega_grid = np.linspace(omega_minimum, omega_maximum, omega_grid_points)

    operations = np.empty(
        (omega_grid_points, len(NEAREST_NEIGHBOR_PAIR_CYCLE), 4, 4, 4),
        dtype=complex,
    )
    for omega_index, omega in enumerate(omega_grid):
        blocks = [
            _burau_letter(generator, power, float(omega))
            for generator, power in ((1, 1), (1, -1), (2, 1), (2, -1))
        ]
        for pair_index, pair in enumerate(NEAREST_NEIGHBOR_PAIR_CYCLE):
            for code, block in enumerate(blocks):
                operations[omega_index, pair_index, code] = _embed(block, pair)

    results = []
    for depth in depths:
        rng = np.random.default_rng(seed + depth)
        best_error = np.inf
        best_codes = None
        best_grid_index = None
        best_template_index = None
        for template_index in range(templates_per_depth):
            codes = rng.integers(0, 4, size=depth)
            meshes = np.broadcast_to(
                np.eye(NUMBER_OF_MODES, dtype=complex),
                (omega_grid_points, NUMBER_OF_MODES, NUMBER_OF_MODES),
            ).copy()
            for layer_index, code in enumerate(codes):
                meshes = operations[:, layer_index % 3, int(code)] @ meshes
            transformed = meshes @ observable @ np.swapaxes(meshes.conj(), 1, 2)
            off_diagonal = transformed.copy()
            diagonal_indices = np.arange(NUMBER_OF_MODES)
            off_diagonal[:, diagonal_indices, diagonal_indices] = 0.0
            errors = la.norm(off_diagonal, axis=(1, 2)) / la.norm(observable)
            grid_index = int(np.argmin(errors))
            if errors[grid_index] < best_error:
                best_error = float(errors[grid_index])
                best_codes = codes.copy()
                best_grid_index = grid_index
                best_template_index = template_index

        if best_codes is None or best_grid_index is None:
            raise AssertionError("shared-omega template search produced no candidate")
        bracket_left = omega_grid[max(0, best_grid_index - 1)]
        bracket_right = omega_grid[min(omega_grid_points - 1, best_grid_index + 1)]
        refined = minimize_scalar(
            lambda omega: _score_reconstruction_error(
                _evaluate_shared_mesh(float(omega), best_codes), observable
            ),
            bounds=(bracket_left, bracket_right),
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 200},
        )
        results.append(
            {
                "depth": depth,
                "best_relative_observable_frobenius_error": float(refined.fun),
                "omega": float(refined.x),
                "template_index": int(best_template_index),
                "application_order_letter_codes": best_codes.tolist(),
                "application_order_local_generators": [
                    [
                        int(PRIMITIVE_LETTERS[int(code)][0]),
                        int(PRIMITIVE_LETTERS[int(code)][1]),
                    ]
                    for code in best_codes
                ],
            }
        )
    return results


def _shared_omega_positive_control(
    *,
    seed: int = 8801,
    omega_margin: float = OMEGA_MARGIN,
    omega_grid_points: int = 129,
) -> dict[str, object]:
    """Recover a hidden omega when the in-family discrete template is known."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 4, size=12)
    hidden_omega = 0.91
    mesh = _evaluate_shared_mesh(hidden_omega, codes)
    weights = np.asarray([-1.4, -0.3, 0.55, 1.25])
    observable = mesh.conj().T @ np.diag(weights) @ mesh
    omega_grid = np.linspace(
        omega_margin,
        2.0 * np.pi / 3.0 - omega_margin,
        omega_grid_points,
    )
    errors = np.asarray(
        [
            _score_reconstruction_error(
                _evaluate_shared_mesh(float(omega), codes), observable
            )
            for omega in omega_grid
        ]
    )
    grid_index = int(np.argmin(errors))
    bracket_left = omega_grid[max(0, grid_index - 1)]
    bracket_right = omega_grid[min(omega_grid_points - 1, grid_index + 1)]

    def residual(parameters: np.ndarray) -> np.ndarray:
        trial = _evaluate_shared_mesh(float(parameters[0]), codes)
        transformed = trial @ observable @ trial.conj().T
        off_diagonal = transformed.copy()
        diagonal_indices = np.arange(NUMBER_OF_MODES)
        off_diagonal[diagonal_indices, diagonal_indices] = 0.0
        normalized = off_diagonal / la.norm(observable)
        return np.concatenate((normalized.real.ravel(), normalized.imag.ravel()))

    recovered = least_squares(
        residual,
        x0=np.asarray((omega_grid[grid_index],)),
        bounds=(np.asarray((bracket_left,)), np.asarray((bracket_right,))),
        max_nfev=200,
        ftol=1e-14,
        xtol=1e-14,
        gtol=1e-14,
    )
    recovered_error = float(la.norm(residual(recovered.x)))
    return {
        "seed": seed,
        "depth": len(codes),
        "control_scope": (
            "continuous omega recovery for a known in-family letter template; "
            "does not validate exhaustive discrete-template recovery"
        ),
        "hidden_omega": hidden_omega,
        "recovered_omega": float(recovered.x[0]),
        "absolute_omega_error": float(abs(recovered.x[0] - hidden_omega)),
        "relative_reconstruction_error": recovered_error,
        "function_evaluations": int(recovered.nfev),
        "application_order_letter_codes": codes.tolist(),
    }


def _haar_unitary(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(NUMBER_OF_MODES, NUMBER_OF_MODES)) + 1j * rng.normal(
        size=(NUMBER_OF_MODES, NUMBER_OF_MODES)
    )
    unitary, triangular = la.qr(matrix)
    diagonal = np.diag(triangular)
    phases = diagonal / np.abs(diagonal)
    return unitary @ np.diag(phases.conj())


def _held_out_target_benchmark(
    *,
    number_of_targets: int,
    depth: int,
    restarts: int,
    target_seed: int,
    restart_seed_base: int,
    maximum_evaluations: int,
    omega_margin: float,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(target_seed)
    results = []
    for target_index in range(number_of_targets):
        target = _haar_unitary(rng)
        fit = _fit_independent_omegas(
            target,
            depth,
            restarts=restarts,
            seed_base=restart_seed_base + target_index * 100 - depth * 100,
            maximum_evaluations=maximum_evaluations,
            omega_margin=omega_margin,
        )
        results.append(
            {
                "target_index": target_index,
                "ordered_detector_basis_error": fit[
                    "ordered_detector_basis_error"
                ],
                "normalized_phase_aligned_unitary_error": fit[
                    "normalized_phase_aligned_unitary_error"
                ],
                "successful_restarts": fit["successful_restarts"],
                "restart_results": fit["restart_results"],
            }
        )
    return results


def _generic_givens_baseline(target: np.ndarray) -> dict[str, object]:
    """Construct an exact six-cell detector-basis baseline.

    Applying left Givens rotations to target^dagger gives L target^dagger = D.
    Therefore target = D^dagger L.  D^dagger is an output phase matrix and is
    irrelevant to terminal square-law detection; L is the six-cell mesh.
    """
    working = target.conj().T.copy()
    embedded_cells = []
    serialized_cells = []
    for column in range(NUMBER_OF_MODES - 1):
        for row in range(NUMBER_OF_MODES - 1, column, -1):
            first = working[row - 1, column]
            second = working[row, column]
            radius = float(np.hypot(abs(first), abs(second)))
            if radius < 1e-15:
                block = np.eye(2, dtype=complex)
            else:
                block = np.asarray(
                    [
                        [first.conjugate() / radius, second.conjugate() / radius],
                        [-second / radius, first / radius],
                    ]
                )
            pair = (row - 1, row)
            embedded = _embed(block, pair)
            working = embedded @ working
            embedded_cells.append(embedded)
            serialized_cells.append(
                {"pair": list(pair), "unitary": _complex_payload(block)}
            )

    mesh = np.eye(NUMBER_OF_MODES, dtype=complex)
    for embedded in embedded_cells:
        mesh = embedded @ mesh
    output_phases = np.diag(working).conj()
    reconstructed = np.diag(output_phases) @ mesh
    return {
        "pair_cells": len(embedded_cells),
        "basis_continuous_controls": NUMBER_OF_MODES * (NUMBER_OF_MODES - 1),
        "irrelevant_output_phases": NUMBER_OF_MODES,
        "serialized_cell_schedule": "triangular Givens elimination order",
        "rectangular_parallel_layers": NUMBER_OF_MODES,
        "rectangular_parallel_layers_scope": (
            "known Clements rectangular realization with the same universal "
            "six-cell count; not the depth of the serialized triangular schedule"
        ),
        "mesh": mesh,
        "output_alignment_phases": np.angle(output_phases),
        "full_unitary_reconstruction_error": float(la.norm(reconstructed - target)),
        "ordered_detector_basis_error": _basis_error(mesh, target),
        "triangular_givens_cells": serialized_cells,
    }


def _parallel_layers(number_of_letters: int) -> int:
    complete_cycles, remainder = divmod(number_of_letters, 3)
    return 2 * complete_cycles + (1 if remainder else 0)


def _serialize_fit(fit: dict[str, object]) -> dict[str, object]:
    architecture = fit["architecture"]
    omegas = np.asarray(fit["omegas"])
    return {
        "depth": int(fit["depth"]),
        "pair_cells": int(fit["depth"]),
        "braid_letters": int(fit["depth"]),
        "independent_omega_controls": int(fit["depth"]),
        "parallel_pair_layers": _parallel_layers(int(fit["depth"])),
        "ordered_detector_basis_error": float(fit["ordered_detector_basis_error"]),
        "normalized_phase_aligned_unitary_error": float(
            fit["normalized_phase_aligned_unitary_error"]
        ),
        "successful_restarts": int(fit["successful_restarts"]),
        "layers": [
            {
                "pair": list(layer.pair),
                "word": [[layer.generator, layer.power]],
                "omega": float(omega),
                "omega_branch": "0 < omega < 2*pi/3",
                "squier_form_condition_number": float(
                    la.cond(positive_form(float(omega)))
                ),
            }
            for layer, omega in zip(architecture, omegas)
        ],
        "restart_results": fit["restart_results"],
    }


def run_compiler_study(
    *,
    independent_depths: Sequence[int] = INDEPENDENT_DEPTHS,
    independent_restarts: int = 5,
    independent_seed_base: int = 199_200,
    independent_maximum_evaluations: int = 1000,
    held_out_targets: int = 3,
    held_out_depth: int = 24,
    held_out_restarts: int = 5,
    held_out_maximum_evaluations: int = 500,
    held_out_target_seed: int = 4567,
    held_out_restart_seed_base: int = 600_000,
    shared_depths: Sequence[int] = SHARED_DEPTHS,
    shared_templates_per_depth: int = 512,
    shared_omega_grid_points: int = 129,
    shared_seed: int = 70_000,
    shared_positive_control_seed: int = 8801,
    constructive_shared_omega: float = CONSTRUCTIVE_SHARED_OMEGA,
    constructive_half_word_depth: int = CONSTRUCTIVE_HALF_WORD_DEPTH,
    constructive_candidates_per_cell: int = CONSTRUCTIVE_CANDIDATES_PER_CELL,
    constructive_beam_width: int = CONSTRUCTIVE_BEAM_WIDTH,
    constructive_query_neighbors: int = CONSTRUCTIVE_QUERY_NEIGHBORS,
    omega_margin: float = OMEGA_MARGIN,
    validation_states: int = 2000,
    seed: int = 424_242,
) -> dict[str, object]:
    """Run target compilation, controls, and the generic-MZI comparison."""
    target, observable, target_weights = _load_target_model()
    target_consistency_error = float(
        la.norm(target.conj().T @ np.diag(target_weights) @ target - observable)
    )
    target_unitarity_error = float(
        la.norm(target.conj().T @ target - np.eye(NUMBER_OF_MODES))
    )
    if not (
        omega_margin
        <= constructive_shared_omega
        <= 2.0 * np.pi / 3.0 - omega_margin
    ):
        raise ValueError("constructive shared omega lies outside the declared margin")

    validation_rng = np.random.default_rng(seed)
    states = validation_rng.normal(size=(validation_states, NUMBER_OF_MODES))
    states = states + 1j * validation_rng.normal(
        size=(validation_states, NUMBER_OF_MODES)
    )
    states /= la.norm(states, axis=1, keepdims=True)

    fitted_by_depth = [
        _fit_independent_omegas(
            target,
            int(depth),
            restarts=independent_restarts,
            seed_base=independent_seed_base,
            maximum_evaluations=independent_maximum_evaluations,
            omega_margin=omega_margin,
        )
        for depth in independent_depths
    ]
    successful_fits = [
        fit
        for fit in fitted_by_depth
        if fit["ordered_detector_basis_error"] < TARGET_SUCCESS_THRESHOLD
    ]
    selected = successful_fits[0] if successful_fits else fitted_by_depth[-1]
    selected_mesh = np.asarray(selected["mesh"])
    selected_score_metrics = _score_metrics(
        selected_mesh, target, observable, states
    )
    selected_jacobian = _projector_jacobian_diagnostics(
        np.asarray(selected["omegas"]), selected["architecture"]
    )
    selected_squier_conditions = [
        float(la.cond(positive_form(float(omega))))
        for omega in selected["omegas"]
    ]

    held_out = _held_out_target_benchmark(
        number_of_targets=held_out_targets,
        depth=held_out_depth,
        restarts=held_out_restarts,
        target_seed=held_out_target_seed,
        restart_seed_base=held_out_restart_seed_base,
        maximum_evaluations=held_out_maximum_evaluations,
        omega_margin=omega_margin,
    )
    held_out_success_fraction = float(
        np.mean(
            [
                row["ordered_detector_basis_error"] < TARGET_SUCCESS_THRESHOLD
                for row in held_out
            ]
        )
    )

    shared_search = _shared_omega_search(
        observable,
        depths=shared_depths,
        templates_per_depth=shared_templates_per_depth,
        omega_grid_points=shared_omega_grid_points,
        seed=shared_seed,
        omega_margin=omega_margin,
    )
    shared_positive_control = _shared_omega_positive_control(
        seed=shared_positive_control_seed,
        omega_margin=omega_margin,
        omega_grid_points=shared_omega_grid_points,
    )

    generic = _generic_givens_baseline(target)
    generic_metrics = _score_metrics(
        np.asarray(generic["mesh"]), target, observable, states
    )
    constructive = _constructive_shared_omega_compile(
        target,
        generic,
        omega=constructive_shared_omega,
        half_word_depth=constructive_half_word_depth,
        candidates_per_cell=constructive_candidates_per_cell,
        beam_width=constructive_beam_width,
        query_neighbors=constructive_query_neighbors,
    )
    constructive_mesh = np.asarray(constructive["mesh"])
    constructive_metrics = _score_metrics(
        constructive_mesh, target, observable, states
    )
    constructive_omega_sensitivity = []
    for omega_offset in (-0.003, -0.001, 0.001, 0.003):
        shifted_mesh = _replay_constructive_cell_words(
            constructive["cells"], constructive_shared_omega + omega_offset
        )
        shifted_metrics = _score_metrics(
            shifted_mesh, target, observable, states
        )
        constructive_omega_sensitivity.append(
            {
                "common_omega_offset": omega_offset,
                "omega": constructive_shared_omega + omega_offset,
                "ordered_detector_basis_error": shifted_metrics[
                    "ordered_detector_basis_error"
                ],
                "relative_observable_frobenius_error": shifted_metrics[
                    "relative_observable_frobenius_error"
                ],
                "held_out_normalized_score_rmse": shifted_metrics[
                    "held_out_normalized_score_rmse"
                ],
            }
        )

    serialized_depth_sweep = [_serialize_fit(fit) for fit in fitted_by_depth]
    selected_serialized = _serialize_fit(selected)
    selected_depth = int(selected["depth"])
    resources = {
        "generic_rectangular_mzi": {
            "pair_cells": int(generic["pair_cells"]),
            "continuous_basis_controls": int(generic["basis_continuous_controls"]),
            "parallel_pair_layers": int(generic["rectangular_parallel_layers"]),
        },
        "selected_independent_omega_burau": {
            "pair_cells": selected_depth,
            "braid_letters": selected_depth,
            "continuous_basis_controls": selected_depth,
            "parallel_pair_layers": _parallel_layers(selected_depth),
            "cell_count_ratio_to_generic": float(
                selected_depth / int(generic["pair_cells"])
            ),
            "parallel_depth_ratio_to_generic_rectangular": float(
                _parallel_layers(selected_depth)
                / int(generic["rectangular_parallel_layers"])
            ),
        },
        "constructive_shared_omega_burau": {
            "logical_givens_cells": int(constructive["logical_givens_cells"]),
            "braid_letters": int(constructive["total_braid_letters"]),
            "target_informed_design_time_global_omega_choices": 1,
            "per_letter_continuous_omega_controls": 0,
            "runtime_continuous_omega_controls_after_library_choice": 0,
            "conservative_sequential_pair_layers": int(
                constructive["conservative_sequential_pair_layers"]
            ),
            "letter_count_ratio_to_generic_pair_cells": float(
                int(constructive["total_braid_letters"])
                / int(generic["pair_cells"])
            ),
        },
        "idealized_loss_comparison": (
            "All modeled matrices are unitary, so the simulation contains no "
            "insertion loss. If every programmed pair-cell layer costs ell dB, "
            "the selected Burau schedule contributes "
            f"{_parallel_layers(selected_depth)}*ell "
            "dB, the conservative fixed-omega cell-word schedule contributes "
            f"{int(constructive['conservative_sequential_pair_layers'])}*ell dB, "
            f"and a four-mode rectangular generic mesh contributes "
            f"{int(generic['rectangular_parallel_layers'])}*ell dB, before "
            "routing and coupling."
        ),
    }

    diagnostics: dict[str, object] = {
        "scope": (
            "numerical four-mode compiler study for terminal direct detection; "
            "not a universality, scaling, loss, energy, or fabrication claim"
        ),
        "target_model": str(TARGET_MODEL_PATH.relative_to(HERE.parent.parent)),
        "target_consistency_error": target_consistency_error,
        "target_unitarity_error": target_unitarity_error,
        "validation_sample": {
            "seed": seed,
            "normalized_complex_states": validation_states,
        },
        "primitive_letter_codebook": {
            str(code): {"generator": generator, "power": power}
            for code, (generator, power) in enumerate(PRIMITIVE_LETTERS)
        },
        "serialized_circuit_order_convention": APPLICATION_ORDER_CONVENTION,
        "omega_domain": {
            "branch": "0 < omega < 2*pi/3",
            "boundary_margin": omega_margin,
            "minimum": omega_margin,
            "maximum": float(2.0 * np.pi / 3.0 - omega_margin),
        },
        "independent_omega_compiler": {
            "interpretation": (
                "one independently tuned omega per primitive Burau letter; an "
                "optimistic engineering ansatz, not one fixed Burau-derived "
                "block library"
            ),
            "objective": (
                "ordered detector-basis fit with four nuisance output phases; "
                "the phases do not affect terminal intensity detection"
            ),
            "success_threshold": TARGET_SUCCESS_THRESHOLD,
            "restarts_per_depth": independent_restarts,
            "restart_seed_base": independent_seed_base,
            "maximum_function_evaluations_per_restart": (
                independent_maximum_evaluations
            ),
            "depth_sweep": serialized_depth_sweep,
            "shallowest_successful_tested_depth": (
                selected_depth if successful_fits else None
            ),
            "selected_score_metrics": selected_score_metrics,
            "selected_projector_jacobian": selected_jacobian,
            "selected_maximum_squier_form_condition_number": float(
                max(selected_squier_conditions)
            ),
            "selected_active_boundary_count": int(
                np.sum(
                    np.minimum(
                        np.asarray(selected["omegas"]) - omega_margin,
                        2.0 * np.pi / 3.0
                        - omega_margin
                        - np.asarray(selected["omegas"]),
                    )
                    < 1e-5
                )
            ),
        },
        "additional_seeded_haar_basis_benchmark": {
            "targets": held_out_targets,
            "depth": held_out_depth,
            "restarts_per_target": held_out_restarts,
            "target_seed": held_out_target_seed,
            "restart_seed_base": held_out_restart_seed_base,
            "maximum_function_evaluations_per_restart": (
                held_out_maximum_evaluations
            ),
            "success_threshold": TARGET_SUCCESS_THRESHOLD,
            "success_fraction": held_out_success_fraction,
            "results": held_out,
        },
        "constructive_shared_omega_compiler": {
            "interpretation": (
                "one fixed omega and one 2x2 B3 Burau-derived block library across "
                "every primitive letter; six target-specific exponent-neutral "
                "cell words are embedded on successive mode pairs. This is not a "
                "single four-mode Burau representation or one global B_n word"
            ),
            "omega_selection_provenance": (
                "sqrt(2) was selected after exploratory comparisons on this "
                "target and then frozen; it counts as one target-informed "
                "design-time global choice, not as an a priori universal value"
            ),
            "construction": (
                "decompose the target detector basis into six determinant-one "
                "nearest-neighbor Givens cells; synthesize each cell by a "
                "deterministic meet-in-the-middle search; retain several cell "
                "approximants and choose their joint combination by beam search"
            ),
            "determinant_constraint": (
                "each cell word has zero total Artin exponent, cancelling the "
                "common generator determinant phase without per-cell phase knobs"
            ),
            "ordered_detector_basis_success_threshold": (
                CONSTRUCTIVE_BASIS_SUCCESS_THRESHOLD
            ),
            "relative_score_matrix_success_threshold": (
                CONSTRUCTIVE_SCORE_SUCCESS_THRESHOLD
            ),
            "physical_assumptions": [
                (
                    "beta_1, beta_2, and their inverses are available at equal "
                    "primitive-letter cost on every addressed mode pair"
                ),
                (
                    "two-mode blocks are embedded and routed ideally on the "
                    "declared adjacent pair supports"
                ),
                (
                    "the Squier basis change is absorbed into the ideal block and "
                    "its hardware cost is not counted"
                ),
                (
                    "one common omega is calibrated and remains coherent across "
                    "all six cell words"
                ),
                (
                    "the discrete cell words are programmed specifically for this "
                    "exported detector basis"
                ),
            ],
            **{
                key: value
                for key, value in constructive.items()
                if key != "mesh"
            },
            "score_metrics": constructive_metrics,
            "fixed_word_common_omega_sensitivity": (
                constructive_omega_sensitivity
            ),
        },
        "shared_omega_control": {
            "interpretation": (
                "a direct end-to-end random-template search with one omega for "
                "the complete fixed-pair-cycle circuit; retained as a negative "
                "search-budget control beside the constructive fixed-block-library "
                "compiler"
            ),
            "search_method": (
                "seeded random generator/inverse choices on the declared mode-pair "
                "cycle, followed by a bounded omega grid and scalar polish"
            ),
            "fixed_mode_pair_cycle": [
                list(pair) for pair in NEAREST_NEIGHBOR_PAIR_CYCLE
            ],
            "templates_per_depth": shared_templates_per_depth,
            "omega_grid_points": shared_omega_grid_points,
            "template_seed": shared_seed,
            "results": shared_search,
            "fixed_template_omega_recovery_control": shared_positive_control,
            "failure_interpretation": (
                "nonzero error means no fit was found under this declared "
                "direct random-template and omega-search budget; it is not a "
                "non-reachability result and does not contradict the constructive fit"
            ),
        },
        "generic_mzi_baseline": {
            key: value
            for key, value in generic.items()
            if key
            not in {
                "mesh",
                "output_alignment_phases",
                "triangular_givens_cells",
            }
        }
        | {
            "output_alignment_phases": np.asarray(
                generic["output_alignment_phases"]
            ).tolist(),
            "score_metrics": generic_metrics,
        },
        "resource_comparison": resources,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "limitations": [
            (
                "Independent omega values do not define one fixed Burau-derived "
                "block library and may not map to one physical tuning knob."
            ),
            (
                "Squier unitarization is treated as part of each ideal block; "
                "its basis-changing hardware and calibration cost are not modeled."
            ),
            (
                "The generic and Burau matrices are ideal and lossless; cell "
                "counts do not determine measured insertion loss or energy."
            ),
            (
                "Finite seeded target samples and numerical optimization do not "
                "prove exact universality or favorable scaling."
            ),
            (
                "The fixed-omega construction is a finite target-specific "
                "approximation with a long word; it does not prove density, exact "
                "synthesis, efficient asymptotic compilation, or a loss advantage."
            ),
            (
                "Holding the six discrete words fixed while shifting their common "
                "omega by only 0.001 materially increases the error, so calibration "
                "and coherence tolerance require a separate hardware study."
            ),
            (
                "Output-phase equivalence is valid here because the mesh ends "
                "in direct square-law detection."
            ),
        ],
    }

    if target_consistency_error > 1e-10 or target_unitarity_error > 1e-10:
        raise AssertionError("the exported digital-twin target is inconsistent")
    if not successful_fits:
        raise AssertionError("independent-omega compiler did not reach its target")
    if selected_score_metrics["relative_observable_frobenius_error"] > 1e-9:
        raise AssertionError("compiled mesh did not reproduce the target score")
    if selected_jacobian["numerical_rank"] != NUMBER_OF_MODES * (
        NUMBER_OF_MODES - 1
    ):
        raise AssertionError("selected compiler architecture lacks full local rank")
    if held_out_success_fraction < 2.0 / 3.0:
        raise AssertionError("held-out target compiler success was not reproducible")
    if shared_positive_control["relative_reconstruction_error"] > 1e-10:
        raise AssertionError("the fixed-template omega recovery control failed")
    if generic["full_unitary_reconstruction_error"] > 1e-10:
        raise AssertionError("generic Givens baseline did not reconstruct the target")
    if (
        constructive_metrics["ordered_detector_basis_error"]
        > CONSTRUCTIVE_BASIS_SUCCESS_THRESHOLD
    ):
        raise AssertionError("constructive shared-omega compiler missed its threshold")
    if (
        constructive_metrics["relative_observable_frobenius_error"]
        > CONSTRUCTIVE_SCORE_SUCCESS_THRESHOLD
    ):
        raise AssertionError("constructive shared-omega score error missed its threshold")
    if constructive_metrics["maximum_unitarity_error"] > 1e-10:
        raise AssertionError("constructive shared-omega mesh is not unitary")
    if constructive["maximum_serialized_word_replay_error"] > 1e-12:
        raise AssertionError("a serialized fixed-omega cell word did not replay")
    if constructive["maximum_beta_word_algebraic_replay_error"] > 1e-12:
        raise AssertionError("a beta_word-compatible cell word did not replay")
    if constructive["full_mesh_serialized_word_replay_error"] > 1e-12:
        raise AssertionError(
            "serialized fixed-omega cell words did not replay the mesh"
        )
    if any(
        cell["artin_exponent_sum"] != 0 for cell in constructive["cells"]
    ):
        raise AssertionError("a constructive cell word violates exponent neutrality")

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.7))
    depth_values = [row["depth"] for row in serialized_depth_sweep]
    independent_errors = [
        row["ordered_detector_basis_error"] for row in serialized_depth_sweep
    ]
    axes[0, 0].semilogy(
        depth_values,
        np.maximum(independent_errors, 1e-16),
        marker="o",
        color="#7f3c8d",
        label="independent omega / letter",
    )
    axes[0, 0].scatter(
        [generic["pair_cells"]],
        [max(generic["ordered_detector_basis_error"], 1e-16)],
        marker="*",
        s=130,
        color="#1b9e77",
        label="generic exact mesh",
        zorder=4,
    )
    axes[0, 0].axhline(
        TARGET_SUCCESS_THRESHOLD,
        color="0.4",
        linestyle="--",
        linewidth=1.0,
        label="success threshold",
    )
    axes[0, 0].set_xlabel("two-mode cells / primitive letters")
    axes[0, 0].set_ylabel("ordered detector-basis error")
    axes[0, 0].set_title("(a) Exported scorer compilation")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].semilogy(
        [row["depth"] for row in shared_search],
        [row["best_relative_observable_frobenius_error"] for row in shared_search],
        marker="s",
        color="#d95f02",
        label="direct random templates",
    )
    axes[0, 1].scatter(
        [constructive["total_braid_letters"]],
        [constructive_metrics["relative_observable_frobenius_error"]],
        marker="*",
        s=140,
        color="#1b9e77",
        label="constructive shared omega",
        zorder=4,
    )
    axes[0, 1].axhline(
        CONSTRUCTIVE_SCORE_SUCCESS_THRESHOLD,
        color="0.4",
        linestyle="--",
        linewidth=1.0,
        label="1% score-error threshold",
    )
    axes[0, 1].set_xlabel("total primitive letters")
    axes[0, 1].set_ylabel("relative score-matrix error")
    axes[0, 1].set_title("(b) One fixed Burau block library")
    axes[0, 1].legend(fontsize=8)

    singular_values = np.asarray(selected_jacobian["singular_values"])
    axes[1, 0].semilogy(
        np.arange(1, len(singular_values) + 1),
        singular_values / singular_values[0],
        marker="o",
        color="#4c78a8",
    )
    axes[1, 0].axvline(12.5, color="0.4", linestyle=":", linewidth=1.0)
    axes[1, 0].set_xlabel("projector-Jacobian singular-value index")
    axes[1, 0].set_ylabel("normalized singular value")
    axes[1, 0].set_title("(c) 12-dimensional detector-basis tangent")

    held_out_errors = np.asarray(
        [row["ordered_detector_basis_error"] for row in held_out]
    )
    axes[1, 1].semilogy(
        np.arange(1, held_out_targets + 1),
        np.maximum(held_out_errors, 1e-16),
        marker="o",
        linestyle="none",
        markersize=7,
        color="#7f3c8d",
        label=f"{held_out_depth}-letter compiler",
    )
    axes[1, 1].axhline(
        TARGET_SUCCESS_THRESHOLD,
        color="0.4",
        linestyle="--",
        linewidth=1.0,
        label="success threshold",
    )
    axes[1, 1].set_xticks(np.arange(1, held_out_targets + 1))
    axes[1, 1].set_xlabel("additional seeded Haar target")
    axes[1, 1].set_ylabel("ordered detector-basis error")
    axes[1, 1].set_title("(d) Small additional reachability check")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Restricted Burau compilation: reachability and resource controls",
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
    compiled_model = {
        "schema_version": 2,
        "scope": (
            "independently tuned short-depth and constructive fixed-omega "
            "Burau compilers for a terminal direct-detection quadratic scorer"
        ),
        "transfer_convention": "output_amplitudes = unitary @ input_amplitudes",
        "primitive_letter_codebook": {
            str(code): {"generator": generator, "power": power}
            for code, (generator, power) in enumerate(PRIMITIVE_LETTERS)
        },
        "serialized_circuit_order_convention": APPLICATION_ORDER_CONVENTION,
        "target_model": str(TARGET_MODEL_PATH.relative_to(HERE.parent.parent)),
        "omega_domain": diagnostics["omega_domain"],
        "architecture": selected_serialized,
        "compiled_unitary": _complex_payload(selected_mesh),
        "independent_omega_output_phase_alignment_radians": np.asarray(
            selected["output_alignment_phases"]
        ).tolist(),
        "compiled_readout_weights": selected_score_metrics[
            "fitted_readout_weights"
        ],
        "score_metrics": selected_score_metrics,
        "constructive_shared_omega_architecture": {
            key: value
            for key, value in constructive.items()
            if key != "mesh"
        }
        | {
            "compiled_unitary": _complex_payload(constructive_mesh),
            "score_metrics": constructive_metrics,
            "omega_selection_provenance": diagnostics[
                "constructive_shared_omega_compiler"
            ]["omega_selection_provenance"],
            "physical_assumptions": diagnostics[
                "constructive_shared_omega_compiler"
            ]["physical_assumptions"],
            "ordered_detector_basis_success_threshold": (
                CONSTRUCTIVE_BASIS_SUCCESS_THRESHOLD
            ),
            "relative_score_matrix_success_threshold": (
                CONSTRUCTIVE_SCORE_SUCCESS_THRESHOLD
            ),
        },
        "generic_givens_baseline": {
            "pair_cells": generic["pair_cells"],
            "serialized_cell_schedule": generic["serialized_cell_schedule"],
            "rectangular_parallel_layers": generic[
                "rectangular_parallel_layers"
            ],
            "rectangular_parallel_layers_scope": generic[
                "rectangular_parallel_layers_scope"
            ],
            "output_alignment_phases": np.asarray(
                generic["output_alignment_phases"]
            ).tolist(),
            "mesh": _complex_payload(np.asarray(generic["mesh"])),
            "triangular_givens_cells": generic["triangular_givens_cells"],
        },
    }
    MODEL_PATH.write_text(
        json.dumps(compiled_model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {FIGURE_PATH.relative_to(Path.cwd())}")
    print(f"Wrote {RESULTS_PATH.relative_to(Path.cwd())}")
    print(f"Wrote {MODEL_PATH.relative_to(Path.cwd())}")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return diagnostics


if __name__ == "__main__":
    run_compiler_study()
