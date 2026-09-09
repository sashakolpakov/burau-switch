# Independent device research tracks

These follow-on studies are intentionally separate from the
Burau--anyon Gedankenexperiment manuscript and from its main reproduction
script.

The [classical device blueprint](DEVICE_BLUEPRINT.md) turns the numerical
results into explicit inference and security hardware proposals. It is an
engineering note, not another manuscript.

Run all six applied studies from the repository root with:

    python -m programs.reproduce

This command is intentionally separate from the manuscript-level
`python reproduce.py` entry point.

- [security](security/README.md) starts from an attacker model and asks what
  additional physical entropy would be required for authentication.
- [passive security](passive_security/README.md) gives the constructive
  high-dimensional T-key route.  It separately verifies the exact passive
  sum/difference mixer and the photon-starved matched-detection model; the
  Burau block supplies no state-estimation advantage by itself.
- [inference](inference/README.md) derives the passive quadratic ceiling and
  tests Burau blocks as a complete quadratic feature bank.
- [inference scaling](inference_scaling/README.md) compares generic spectral,
  tight MUB, and random measurement banks beyond four modes; it does not claim
  that the MUB bases have Burau-derived compilations.
- [digital twin](digital_twin/README.md) models a bright classical coherent
  processor with detector noise and drift, with an optional Qiskit
  statevector cross-check.
- [compiler](compiler/README.md) fits the exported four-mode scorer with
  restricted Burau--Squier blocks, including a constructive approximation from
  one shared-$\omega$ two-mode block library, and compares them with an exact
  generic interferometer decomposition.

Each directory owns its code, generated figures, and results. Any later
manuscript will live in its own directory; none of these artifacts changes the
claims of the original paper.
