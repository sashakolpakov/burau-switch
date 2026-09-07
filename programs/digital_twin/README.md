# Classical coherent digital twin

This prototype is separate from the Burau--anyon Gedankenexperiment
manuscript. It models the proposed inference hardware as bright classical
coherent fields passing through passive multiport interferometers and into
ordinary square-law photodiodes. It does not require a single-photon source,
coincidence counting, entanglement, or a quantum processor.

Qiskit is an optional independent implementation of the same normalized
four-mode vector. Four optical mode amplitudes are mapped to the four basis
amplitudes of two simulated qubits, a Burau-mesh unitary is applied, and the
resulting statevector probabilities are compared with normalized classical
output intensities. This is a digital-twin consistency check, not a claim that
the optical device is quantum.

Run the always-available coherent model from the repository root:

    python -m programs.digital_twin.reproduce

For the Qiskit cross-check, use a separate Python 3.12 environment and install
the optional requirements:

    python3.12 -m venv .venv-qiskit
    .venv-qiskit/bin/python -m pip install -r requirements-qiskit.txt
    .venv-qiskit/bin/python -m programs.digital_twin.reproduce

The twin reports:

- exact coherent-field/Qiskit probability agreement when Qiskit is present;
- the one-mesh spectral implementation of one trained quadratic score;
- measurement-frame conditioning for five through eight parallel meshes;
- quadratic-score error versus a fixed total bright-light detector budget;
- sensitivity to unrecalibrated coherent transfer-matrix drift; and
- the detector count and ideal fanout loss of the proposed parallel layout.

It also exports `results/digital_twin_model.json`, containing the eight
fixed-bank 4x4 transfer matrices, the one-mesh spectral baseline, a
representative complex input, the target quadratic observable, and electronic
readout weights. Those are concrete programming targets for a
photonic-circuit compiler or measured bench model.

The exported one-mesh spectral unitary is generic. The separate
[compiler study](../compiler/README.md) matches its detector basis numerically
with an optimistic 20-letter construction in which every primitive Burau
letter has an independently tuned specialization phase $\omega$. That is not
a word in one fixed Burau representation, and it uses more pair cells than the exact
six-cell generic interferometer baseline. A constructive shared-$\omega$
compiler also approximates the target with one fixed $\omega=\sqrt{2}$: its
six exponent-neutral cell words use one fixed two-dimensional $B_3$ block
library, total 132 primitive letters, and reach 0.254% detector-basis error.
The global value was selected after exploratory comparisons on this target.
This establishes finite-target approximate reachability, not a single global
$B_n$ representation or an exact, robust, resource-competitive decomposition.

The detector and drift parameters are explicit sensitivity-study assumptions,
not claimed component specifications. The photoelectron sweep is the total
detected budget after fanout, coupling, and mesh losses; a source-power budget
must account for those losses separately. A measured component model can
replace these assumptions without changing the ideal transfer-matrix layer.
Inputs in this prototype are amplitude-normalized. A device accepting vectors
with variable norm must retain a calibrated total-power channel or carry the
norm separately instead of normalizing it away.
