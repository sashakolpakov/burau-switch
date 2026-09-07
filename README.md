# Burau representation, Squier's form, and non-Abelian anyons

Reproducible calculations for [arXiv:2510.18186](https://arxiv.org/abs/2510.18186).

The central Gedankenexperiment asks whether an ordinary coherent device can
realize the operational content of non-Abelian braid statistics without
assuming that microscopic anyons were present at the outset. It has two
complementary physical readings:

- an **effective-anyonic realization**, if the encoded state space and its
  exchange-like operations form a genuine emergent excitation sector; or
- an **anyonic simulacrum**, if the apparatus reproduces the same
  matrix-valued braid statistics only at the input-output level.

The present calculation certifies the representation-theoretic core shared by
both readings. Distinguishing them physically requires additional fusion,
locality, degeneracy, and robustness tests.

The scripts verify:

- the exact Squier form and its two definite intervals;
- Euclidean unitarity of the sign-normalized Burau generators;
- the Yang--Baxter relation and noncommutativity of both $B_3$ generators;
- the closed-form Helstrom witness for distinguishing $U_1U_2$ from
  $U_2U_1$; and
- the original braid-dressed-versus-bare $T$--$S$ response, together with a
  phase-only control showing why this phenomenology must be paired with the
  two-generator test.

Thus the $T$--$S$ experiment remains the observable device-response layer; the
ordered-generator experiment identifies the non-Abelian origin of that layer.

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
