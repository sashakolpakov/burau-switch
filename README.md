# Burau representation, Squier's form, and non-Abelian anyons

Reproducible calculations for [arXiv:2510.18186](https://arxiv.org/abs/2510.18186).

The scripts verify:

- the exact signature of Squier's form,
  $\det J(\omega)=4\cos^2(\omega/2)-1$;
- Euclidean unitarity on
  $\Omega_+=(0,2\pi/3)\cup(4\pi/3,2\pi)$, using the
  sign-normalized positive form $H(\omega)=\varepsilon(\omega)J(\omega)$;
- the Yang--Baxter relation and noncommutativity of the two unitarized
  $B_3$ generators;
- the closed-form Helstrom witness for distinguishing $U_1U_2$ from
  $U_2U_1$; and
- the braid-dressed-versus-bare switch response as an interference
  diagnostic, including a phase-only counterexample that shows why this
  response is not a causal or non-Abelian witness.

Using Python 3.10 or newer, run everything from a clean environment with:

    python -m pip install -r requirements.txt
    python reproduce.py

The single reproduction command writes:

- figures/witness_gap_summary.png
- results/verification.json

burau_switch.py contains the reusable numerical implementation, while
symbolic_checks.py verifies the defining algebraic identities exactly.
