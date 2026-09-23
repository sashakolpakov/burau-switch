# Burau representation, Squier's form, and non-Abelian anyons

Reproducible calculations for [arXiv:2510.18186](https://arxiv.org/abs/2510.18186).

The code reproduces the manuscript's algebraic Gedankenexperiment. A single
braid word produces the control mixer, while the target switch retains the
two orders `BA` and `AB`.

The scripts verify:

- the exact Squier form and its two definite intervals;
- Euclidean unitarity of the sign-normalized Burau generators;
- the Yang--Baxter relation and noncommutativity of the $B_3$ generators;
- the mixer from the one braid word
  $[\sigma_1,\sigma_2]=\sigma_1\sigma_2\sigma_1^{-1}\sigma_2^{-1}$;
- the braid-dressed-versus-bare $T$--$S$ response with no imposed
  $\omega$-dependent switch phase; and
- the exact null obtained by replacing the generator images with commuting
  diagonal matrices.

The reported Helstrom contrast compares two specified closed unitaries. It is
not used as a causal-nonseparability witness.

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
