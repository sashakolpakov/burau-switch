"""Attack-first feasibility probe for a Burau-switch security device.

This experiment treats a short braid word as a hidden device parameter and
asks how many scalar T--S challenge responses are needed to identify its
response class by exhaustive model matching.  It is a deliberately favorable
test for an attacker, not a claim of cryptographic security.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import numpy.linalg as la

from burau_switch import (
    beta_generator,
    definiteness_sign,
    helstrom_unitary_success,
    positive_form,
    positive_square_root,
    rx,
    rz,
    switch_unitary,
)


HERE = Path(__file__).resolve().parent
FIGURE_PATH = HERE / "figures" / "security_probe.png"
RESULTS_PATH = HERE / "results" / "security_probe.json"
LETTERS = (-2, -1, 1, 2)
TARGET_WORD = (1, 2, -1, 2, 1)


def _letter_matrix(letter: int, s: complex) -> np.ndarray:
    generator = beta_generator(abs(letter), s)
    return generator if letter > 0 else la.inv(generator)


def _response_library(
    words: list[tuple[int, ...]], omegas: np.ndarray
) -> np.ndarray:
    """Return the braid-dressed-minus-bare response for every word/challenge."""
    first_target = rx(1.5)
    second_target = rz(0.75)
    identity_two = np.eye(2, dtype=complex)
    identity_four = np.eye(4, dtype=complex)
    responses = np.empty((len(words), len(omegas)), dtype=float)

    for challenge_index, omega in enumerate(omegas):
        if definiteness_sign(float(omega)) is None:
            raise ValueError("all challenge phases must lie in Omega_+")

        s = np.exp(0.5j * omega)
        form = positive_form(float(omega))
        root = positive_square_root(form)
        inverse_root = la.inv(root)
        letter_matrices = {
            letter: _letter_matrix(letter, s) for letter in LETTERS
        }
        bare_switch = switch_unitary(first_target, second_target, float(omega))
        bare_score = helstrom_unitary_success(identity_four, bare_switch)

        for word_index, word in enumerate(words):
            raw_word = identity_two
            for letter in word:
                raw_word = raw_word @ letter_matrices[letter]
            mixer = root @ raw_word @ inverse_root
            lifted_mixer = np.kron(mixer, identity_two)
            dressed_switch = lifted_mixer @ bare_switch @ lifted_mixer
            responses[word_index, challenge_index] = (
                helstrom_unitary_success(identity_four, dressed_switch)
                - bare_score
            )

    return responses


def _deduplicate_response_classes(
    words: list[tuple[int, ...]], responses: np.ndarray
) -> tuple[list[tuple[int, ...]], np.ndarray, np.ndarray]:
    """Collapse words whose complete sampled response curves coincide."""
    fingerprints = np.round(responses, decimals=11)
    _, first_indices, inverse = np.unique(
        fingerprints, axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(first_indices)
    first_indices = first_indices[order]

    old_to_new = np.empty(len(order), dtype=int)
    old_to_new[order] = np.arange(len(order))
    remapped_inverse = old_to_new[inverse]
    unique_words = [words[index] for index in first_indices]
    return unique_words, responses[first_indices], remapped_inverse


def _probe_order(response_classes: np.ndarray) -> np.ndarray:
    """Greedily choose diverse challenges by pivoted residual variance."""
    residual = response_classes - np.mean(response_classes, axis=0, keepdims=True)
    remaining = list(range(response_classes.shape[1]))
    selected: list[int] = []
    while remaining:
        scores = np.sum(residual[:, remaining] ** 2, axis=0)
        best_position = int(np.argmax(scores))
        best = remaining.pop(best_position)
        selected.append(best)
        direction = residual[:, best]
        norm_squared = float(direction @ direction)
        if norm_squared > 1e-24:
            residual -= np.outer(direction, direction @ residual) / norm_squared
    return np.asarray(selected, dtype=int)


def _matching_attack(
    response_classes: np.ndarray,
    probe_order: np.ndarray,
    probe_counts: list[int],
    noise_standard_deviation: float,
    trials: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average model matching over uniformly sampled hidden response classes."""
    rng = np.random.default_rng(seed)
    success_rates = np.empty(len(probe_counts), dtype=float)
    prediction_errors = np.empty(len(probe_counts), dtype=float)

    for count_index, count in enumerate(probe_counts):
        selected = probe_order[:count]
        successes = 0
        trial_errors = []
        for _ in range(trials):
            target_class = int(rng.integers(len(response_classes)))
            target = response_classes[target_class]
            observed = target[selected] + rng.normal(
                scale=noise_standard_deviation, size=count
            )
            squared_errors = np.mean(
                (response_classes[:, selected] - observed) ** 2, axis=1
            )
            predicted_class = int(np.argmin(squared_errors))
            successes += int(predicted_class == target_class)
            trial_errors.append(
                float(
                    np.sqrt(
                        np.mean((response_classes[predicted_class] - target) ** 2)
                    )
                )
            )
        success_rates[count_index] = successes / trials
        prediction_errors[count_index] = float(np.mean(trial_errors))

    return success_rates, prediction_errors


def run_security_probe(
    *,
    word_length: int = 5,
    challenges_per_component: int = 36,
    noise_standard_deviations: tuple[float, ...] = (5e-4, 2e-3, 1e-2),
    trials: int = 1500,
    target_word: tuple[int, ...] = TARGET_WORD,
    seed: int = 1701,
) -> dict[str, object]:
    """Generate the response library, run the attack, and save diagnostics."""
    if len(target_word) != word_length or any(
        letter not in LETTERS for letter in target_word
    ):
        raise ValueError("target_word must contain word_length valid letters")
    if challenges_per_component < 1:
        raise ValueError("challenges_per_component must be positive")
    if not noise_standard_deviations or any(
        noise < 0.0 for noise in noise_standard_deviations
    ):
        raise ValueError("noise_standard_deviations must be nonempty and nonnegative")
    if trials < 1:
        raise ValueError("trials must be positive")

    margin = 0.06
    left = np.linspace(
        margin, 2.0 * np.pi / 3.0 - margin, challenges_per_component
    )
    right = np.linspace(
        4.0 * np.pi / 3.0 + margin,
        2.0 * np.pi - margin,
        challenges_per_component,
    )
    omegas = np.concatenate((left, right))
    words = list(itertools.product(LETTERS, repeat=word_length))
    responses = _response_library(words, omegas)
    unique_words, response_classes, class_for_word = _deduplicate_response_classes(
        words, responses
    )

    target_word_index = words.index(target_word)
    target_class = int(class_for_word[target_word_index])
    order = _probe_order(response_classes)
    probe_counts = [
        count
        for count in (1, 2, 3, 4, 6, 8, 12, 16)
        if count <= len(omegas)
    ]
    attack_results: dict[str, dict[str, list[float]]] = {}
    for noise_index, noise_standard_deviation in enumerate(
        noise_standard_deviations
    ):
        success_rates, prediction_errors = _matching_attack(
            response_classes,
            order,
            probe_counts,
            noise_standard_deviation,
            trials,
            seed + 1 + noise_index,
        )
        attack_results[f"{noise_standard_deviation:.6g}"] = {
            "exact_class_recovery_rate": success_rates.tolist(),
            "mean_full_curve_rmse": prediction_errors.tolist(),
        }

    centered = response_classes - np.mean(response_classes, axis=0, keepdims=True)
    singular_values = la.svd(centered, compute_uv=False)
    explained = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    dimensions_99 = int(np.searchsorted(explained, 0.99) + 1)
    dimensions_999 = int(np.searchsorted(explained, 0.999) + 1)
    class_sizes = np.bincount(class_for_word)

    diagnostics: dict[str, object] = {
        "attack_model": "enumeration of all fixed-length braid words",
        "attack_target_sampling": "uniform over distinct response classes",
        "word_length": word_length,
        "seed": seed,
        "trials_per_attack_point": trials,
        "challenges_per_definiteness_component": challenges_per_component,
        "enumerated_words": len(words),
        "distinct_sampled_response_classes": len(unique_words),
        "largest_equivalent_word_class": int(np.max(class_sizes)),
        "challenge_count": len(omegas),
        "challenge_phases": omegas.tolist(),
        "noise_standard_deviations": list(noise_standard_deviations),
        "challenge_selection": "pivoted residual variance",
        "selected_probe_indices": order[: probe_counts[-1]].tolist(),
        "selected_probe_phases": omegas[order[: probe_counts[-1]]].tolist(),
        "target_word": list(target_word),
        "target_response_class": target_class,
        "dimensions_for_99_percent_variance": dimensions_99,
        "dimensions_for_99_9_percent_variance": dimensions_999,
        "probe_counts": probe_counts,
        "attack_results_by_noise": attack_results,
    }

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    target_response = response_classes[target_class]
    sample_indices = np.linspace(
        0, len(response_classes) - 1, min(24, len(response_classes)), dtype=int
    )
    for index in sample_indices:
        axes[0].plot(omegas, response_classes[index], color="0.72", alpha=0.35)
    axes[0].plot(
        omegas,
        target_response,
        color="#7f3c8d",
        linewidth=2.2,
        label="target response",
    )
    axes[0].axvspan(2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0, color="white")
    axes[0].set_xlabel(r"challenge phase $\omega$")
    axes[0].set_ylabel(r"$\Delta_{\rm int}$")
    axes[0].set_title("(a) Enumerable response family")
    axes[0].legend(fontsize=8)

    normalized_singular_values = singular_values / singular_values[0]
    axes[1].semilogy(
        np.arange(1, len(normalized_singular_values) + 1),
        normalized_singular_values,
        marker="o",
        markersize=2.5,
        linewidth=1.1,
    )
    axes[1].axvline(dimensions_99, color="#d95f02", linestyle="--", label="99%")
    axes[1].axvline(
        dimensions_999, color="#1b9e77", linestyle=":", label="99.9%"
    )
    axes[1].set_xlabel("response-space component")
    axes[1].set_ylabel("normalized singular value")
    axes[1].set_title("(b) Response-space spectrum")
    axes[1].legend(fontsize=8)

    attack_colors = plt.cm.viridis(
        np.linspace(0.15, 0.85, len(noise_standard_deviations))
    )
    for color, noise_standard_deviation in zip(
        attack_colors, noise_standard_deviations, strict=True
    ):
        rates = attack_results[f"{noise_standard_deviation:.6g}"][
            "exact_class_recovery_rate"
        ]
        axes[2].plot(
            probe_counts,
            rates,
            color=color,
            marker="o",
            label=rf"noise $\sigma={noise_standard_deviation:g}$",
        )
    axes[2].set_xlabel("observed challenge responses")
    axes[2].set_ylabel("mean exact-class recovery")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_title("(c) Enumeration attack")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Current low-dimensional braid response is a model-extraction target",
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
    run_security_probe()
