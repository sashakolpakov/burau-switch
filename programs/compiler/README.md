# Restricted Burau-block compiler

This study consumes the generic four-mode spectral scorer exported by the
[classical digital twin](../digital_twin/README.md) and asks the missing
engineering question: can the same terminal detector basis be synthesized
from Burau--Squier two-mode blocks?

Run it from the repository root with:

    python -m programs.digital_twin.reproduce
    python -m programs.compiler.reproduce

The compiler writes:

- `figures/compiler.png`;
- `results/compiler.json`, with the full search budgets and diagnostics; and
- `results/compiled_mesh.json`, replayable independent-omega and fixed-library
  programs plus the exact generic baseline.

Serialized layers and letter codes are listed in temporal application order
for column vectors, so their matrix product is last-listed block through
first-listed block. Each synthesized cell also records the reversed algebraic
word accepted by the repository's `beta_word()` helper, while each JSON report
includes an explicit four-entry letter-code table. Both interpretations are
replayed numerically, so reproduction does not rely on an implicit ordering
convention.

## Result and scope

The primary fixed architecture uses only nearest-neighbor pairs, alternates
the two positive Burau generators, and assigns one independently tunable
$\omega$ to every primitive letter. Output row phases are treated as nuisance
parameters because the mesh terminates in square-law photodetectors. With the
checked-in seeds, the exported scorer is recovered to numerical precision at
20 letters. A 24-letter architecture also compiles three additional seeded
Haar bases under the declared restart budget.

This is an optimistic engineering family. Giving every primitive letter its
own $\omega$ does **not** define one fixed Burau-derived gate library.

The constructive path removes the per-letter freedom. Exploratory comparisons
on this target selected $\omega=\sqrt{2}$ as one target-informed, design-time
global choice; the compiler then freezes that value for every primitive
letter. It decomposes the target into six determinant-one Givens cells and
synthesizes each cell by deterministic meet-in-the-middle search over
exponent-neutral words. A modest beam search selects the joint six-cell
program. The checked-in configuration uses 132
primitive letters and reaches ordered detector-basis error $2.54\times10^{-3}$,
relative score-matrix error $5.42\times10^{-3}$, and normalized score RMSE
$5.82\times10^{-3}$. Every serialized word is replayed from its letter codes,
and all six have zero total Artin exponent.

This is one fixed **local** $2\times2$ $B_3$ Burau-derived block library,
embedded on successive mode-pair supports. It is not a single four-mode Burau
representation or one global $B_n$ word. The implementation assumes ideal
embedding and routing, equal primitive-letter cost for $\beta_1$, $\beta_2$,
and their inverses, an absorbed Squier basis change, and a common coherent
$\omega$ across all six target-specific cell words.

The study also retains a direct random-template shared-$\omega$ search on a
fixed cycle of mode pairs as a negative search-budget control. Failure there
is reported only as “not found under the stated template and $\omega$-search
budget,” never as a proof of non-reachability. A separate known-template
control hides and recovers its $\omega$; it validates the continuous search,
not exhaustive discrete-circuit recovery.

The comparator is analytic rather than optimizer-limited. Complex Givens
elimination reconstructs the same four-mode detector basis to numerical
precision with six generic two-mode cells. This agrees with the
$N(N-1)/2$ cell count for universal multiport meshes described by Clements et
al., [doi:10.1364/OPTICA.3.001460](https://doi.org/10.1364/OPTICA.3.001460).
The replayable JSON records the triangular elimination schedule. Separately,
the known four-mode Clements rectangular layout realizes the same six-cell
generic comparator in four pair-cell layers, versus 13 layers for the first
successful 20-letter independent-$\omega$ schedule. The fixed-library
construction is much longer: its 132 primitive letters are conservatively
counted as 132 sequential pair layers.

The numerical fit is consequently a target-informed reachability result, not a
Burau-specific speed, depth, loss, energy, or calibration advantage.

## Metrics and safeguards

The primary distance compares the four ordered rank-one detector projectors,
so output phases cancel exactly. The report also includes reconstructed-score
Frobenius and operator-norm errors, held-out score RMSE, unitarity, the local
projector-Jacobian rank and conditioning, Squier-form conditioning, every
restart, and every compiled layer.

All modeled blocks are ideal unitaries. The report gives cell and parallel
layer counts and a symbolic per-layer loss comparison, but it does not insert
an invented component-loss number. Squier unitarization is treated as part of
the ideal block; a physical design must separately account for the
basis-changing optics, tuning range, drift, and calibration burden. Results
at four modes and on a finite target set are not a universality or scaling
proof. Holding the fixed-library words constant and shifting their common
$\omega$ by $\pm 0.001$ raises detector-basis error to roughly 0.96--1.05%;
at $\pm 0.003$ it rises to roughly 2.89--2.98%. These are sensitivity
diagnostics, not a fabrication-tolerance model. The finite construction does
not prove density, exact synthesis, optimal word length, or efficient
asymptotic compilation.
