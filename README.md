# Burau representation, Squier's form, and non-Abelian anyons

Reproducible calculations for [arXiv:2510.18186](https://arxiv.org/abs/2510.18186).

The code reproduces the manuscript's controlled braid-order experiment.  The
two elementary unitarized Burau generators occupy the operation slots, and the
two branches apply `U1 @ U2` and `U2 @ U1`.

The scripts verify:

- the exact Squier form and its two definite intervals;
- Euclidean unitarity of the sign-normalized Burau generators;
- the Yang--Baxter relation and noncommutativity of the $B_3$ generators;
- the controlled-order unitary with branches $U_1U_2$ and $U_2U_1$;
- the closed-form Helstrom witness for distinguishing those two branches; and
- the closed-form minimum control-fringe visibility of their relative group
  commutator.

The witness certifies the implemented non-Abelian $B_3$ action together with
the braid relation.  It is not a causal-nonseparability witness and does not by
itself establish anyonic quasiparticles.

Using Python 3.11 or newer, run everything from a clean environment with:

    python -m pip install -r requirements.txt
    python reproduce.py

The single reproduction command writes:

- figures/witness_gap_summary.png
- results/verification.json

burau_switch.py contains the reusable numerical implementation, while
symbolic_checks.py verifies the defining algebraic identities exactly.

## Independent follow-on programs

The [device research studies](programs/README.md) are kept
separate from this manuscript and from its reproduction command. Each track
has its own assumptions, code, figures, and results. The associated classical
device blueprint, digital twin, and compiler benchmark are engineering
artifacts, not additions to the paper.
