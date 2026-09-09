# Classical device blueprint

This is an engineering note, not part of the Burau--anyon Gedankenexperiment
manuscript and not a claim of demonstrated performance. Both proposed devices
use ordinary photonic hardware but operate in different regimes.  The
inference scorer uses bright coherent light and photodiodes.  The security
reader uses privately phase-randomized weak coherent pulses, a matched optical
analyzer, a power tap, and photon counting; its token remains fixed and
unpowered.  Neither proposal requires anyons, entanglement, cryogenics, or a
deterministic single-photon source.

## Passive quadratic inference

### Recommended first build: one task-compiled quadratic scorer

The cleanest near-term device is a four-input, one-output quadratic scorer:

    one coherent laser
      -> four complex-amplitude input channels
      -> one task-compiled 4x4 unitary mesh
      -> four direct-detection photodiodes
      -> eigenvalue-weighted electronic sum

For a trained Hermitian score $Q$, diagonalize
$Q=V\operatorname{diag}(\lambda)V^\dagger$. Programming the optical mesh as
$U=V^\dagger$ gives

$$
x^\dagger Qx=\sum_j \lambda_j |(Ux)_j|^2.
$$

Thus one mesh computes one arbitrary quadratic score. One laser should feed
all four input channels so their relative phases are coherent, and four
amplitude/phase modulators encode the complex vector. A multi-output model
needs one mesh per noncommuting score in the worst case, although scores that
share an eigenbasis can share the same detected intensities.

This spectral construction is a generic MZI baseline. The numerical
[compiler study](compiler/README.md) now reaches the checked-in detector basis
to numerical precision with 20 nearest-neighbor primitive Burau letters. It
does so only in an optimistic model with one independently tuned $\omega$ per
letter, however, and a small additional-target check uses 24 letters. A second,
constructive route first decomposes the target into six determinant-one Givens
cells and then approximates each cell by an exponent-neutral word in one fixed
two-dimensional $B_3$ Burau-derived block library, embedded on successive mode
pairs at the shared value $\omega=\sqrt{2}$. The checked-in result uses 132
primitive letters and reaches 0.254% detector-basis error, 0.542% relative
score-matrix error, and 0.582% normalized validation-score RMSE. The analytic
generic baseline still needs only six pair cells and is exact. Thus this target
is approximately reachable with one fixed block library, but exact, efficient,
and scalable fixed-$\omega$ compilation are not established. This is not a
single four-mode Burau representation or a global $B_n$ word. The value
$\sqrt{2}$ was selected after exploratory comparisons on this target, so it is
one target-informed design-time choice rather than an a priori universal
parameter.

The present twin assumes amplitude-normalized inputs. If input norm carries
information, the hardware must measure total input power as an additional
scalar (or carry the norm electronically) and restore the $\lVert x\rVert^2$
factor in the quadratic score.

The seeded digital twin recovers the spectral one-mesh score to numerical
precision. At a fixed total budget of $10^6$ detected photoelectrons and an
assumed 5-electron RMS read noise per output, its normalized score RMSE is
about 0.20%. An effective coherent perturbation of $10^{-2}$ produces about
0.96% error. These are sensitivity calculations for one random score, not
component guarantees or worst-case bounds.

### Alternative: a fixed task-agnostic feature bank

If the optical hardware must remain fixed while arbitrary $Q$ matrices are
learned only in electronics, then five four-mode measurement bases are the
exact algebraic minimum. The present digital twin says that minimum can be
poorly conditioned: about 70 in the seeded example, versus about 12 with eight
settings. At the same $10^6$ total detected-photoelectron budget, normalized
score RMSE is about 1.60% for five meshes and 1.25% for eight. A $10^{-2}$
coherent perturbation produces about 3.39% and 2.14% error, respectively.

Eight-way parallelism costs 32 photodiodes and at least
$10\log_{10}(8)=9.03$ dB of ideal fanout loss before coupling and mesh losses.
Sequential reuse of one mesh is appropriate for a bench validation, but it
multiplies latency by the number of settings and cannot establish the desired
throughput advantage.

The useful computation is specifically

$$
s(x)=b+x^\dagger Qx.
$$

That covers quadratic classifiers, covariance and anomaly scores, and
second-order feature maps. It is not a general deep-neural-network layer. A
signed analogue photocurrent summer could emit only the final score and avoid
digitizing every intensity; a programmable digital readout is easier for the
first prototype.

### Physical development sequence

1. **Generic one-mesh baseline.** Program the spectral unitary for a selected
   quadratic task, characterize its full complex transfer matrix, and compare
   measured scores with the checked-in digital twin.
2. **Physical Burau compiler test.** Replay both the checked-in 20-letter
   independent-control fit and the 132-letter shared-$\omega$ construction,
   determine whether their Squier basis changes and controls are physically
   meaningful, and measure approximation error, depth, conditioning, and loss
   against the exact six-cell generic decomposition.
3. **Fixed-bank test only if needed.** If electronic-only retraining matters,
   program five and eight matrices sequentially and measure the conditioning,
   drift, and detector-noise tradeoff before replicating them in parallel.
4. **End-to-end benchmark.** Count laser, input modulation, tuning or heater
   power, transimpedance amplifiers, ADCs, and readout—not merely propagation
   through the passive core.

Thermo-optic settings consume static power even when weights are unchanged.
A literally passive deployed core would require fabrication-fixed, trimmed,
or nonvolatile phase settings; the source, input encoding, detection, and
readout remain active in every case.

The hardware class is realistic: a 2026 silicon-photonic demonstration used a
40-cell 4x4 programmable MZI mesh, classical modulators and photodetectors, and
reported 4x4 unitary processing at 10 GBaud with 6.22-bit effective resolution.
Its 1.92 TOPS and 1.875 pJ/MAC figures apply to the reported photonic core and
must not be silently transferred to this design or to a whole system.

### Inference go/no-go tests

- the one-mesh spectral baseline reaches the task error budget;
- a Burau-block compiler matches that baseline without prohibitive depth,
  approximation error, loss, or calibration burden;
- any fixed feature bank retains a well-conditioned full quadratic span;
- score error remains within the task budget over temperature and time;
- the parallel or multiplexed system beats CPU/GPU latency or energy after
  input/output overhead is included;
- it matches or beats a generic Haar/MZI feature bank at equal optical depth,
  detector count, calibration effort, and precision; and
- scaling beyond four modes has a task-level benefit that justifies either
  $d$ detector outputs per task-compiled score or a $d+1$-setting universal
  bank (a straightforward full-output implementation records $d(d+1)$ values,
  while $d^2$ carefully selected outputs suffice at known norm).

The current mathematics passes the expressivity and optimistic numerical
reachability questions. It does not show a Burau-specific speed, accuracy,
conditioning, or energy advantage; the compiled circuit is deeper than the
generic baseline. The constructive shared-$\omega$ result closes reachability
for this one finite target only; exactness, competitive resources,
universality, and scaling remain open. It also assumes equal-cost access to
both Burau generators and their inverses on every pair, ideal routing, and a
stable common phase: shifting the compiled words' shared $\omega$ by only
$10^{-3}$ raises detector-basis error to roughly 1% in the checked-in
sensitivity test.

## Passive security device: high-dimensional T-key with quantum readout

The constructive prototype is an unpowered optical token used by a trusted,
active reader:

    phase-randomized weak coherent laser
      -> K-mode spatial or modal challenge shaper
      -> exact fixed Burau 50:50 T-mixer
      -> two sealed high-dimensional passive branch maps X and Y
      -> sum and difference ports
      -> computed phase-conjugate matched analyzer
      -> per-round total-power tap and photon counter

At `omega = pi/4`, the three-letter word
`sigma_2^-1 sigma_1 sigma_2^-1` is an exact balanced mixer.  With one fixed
branch bias it returns `(X+Y)x/2` and `(X-Y)x/2`, up to port phases.  If a
separate controlled-order reference realizes `X=BA` and `Y=AB`, the difference
is `[B,A]x/2`; the deployable key does not require physically dubious duplicate
copies of the same `A` and `B`.  Independently fabricated `X` and `Y` are
measured and enrolled as one full transfer operator `H`.

The Burau mixer does not create security dimension.  The reader must control
`K` genuinely independent challenge modes; expanding four amplitudes into many
waveguides or camera pixels still leaves `K<=4`.  Device-specific complexity
comes from the sealed high-dimensional branches, and the anti-emulation gap
comes from sending fewer photons than controlled modes.

Enrollment uses bright phase-sensitive basis probes to recover all columns of
`H` and measure its scaled-isometry defect.  Authentication never selects from
a finite public challenge table.  For every round the reader draws a new
Haar-random `K`-mode vector `x`, computes `Hx`, programs the matched analyzer,
uniformly randomizes the global optical phase, and attenuates to the declared
mean photon number.  Phase randomization is required for the arbitrary-POVM
bound used in the model.

The checked receiver point uses `K=1024`, mean photon number `50`, authentic
focus `0.60`, return efficiency `0.25`, detector efficiency `0.70`, and `0.05`
background clicks.  Under an explicit outcome-conditional returned-energy cap,
the main attacker projection bound is
`(nbar+1)/(nbar+K)=0.047486`; because this is an absolute projection bound,
the modeled attacker mean is `0.4655` clicks.  Five-round
Poisson-surrogate FAR/FRR are `1.58e-4` and `8.13e-5`.  A non-Poisson
click/no-click construction gives a much looser but adaptive-safe worst-error
bound `1.33e-4` after 20 rounds and `6.39e-9` after 50, conditional on the
stated per-round state-estimation and accepted-response energy bound.  The
energy gate and its allowance must be experimentally certified across every
spatial/modal, spectral, polarization, and timing degree of freedom that can
cause an accepted click; an average power monitor alone is not a proof.  All
numbers are receiver-model predictions, not measurements.

A bright laser, SLM/DMD, diffuser, and camera remain useful as the first
alignment, tomography, drift, and classical-emulation test bed.  They are a
baseline, not the positive remote-security protocol: unrestricted bright-light
access exposes a fixed linear map to transfer-matrix or quadratic-response
tomography.

The public four-mode response audit remains an adversarial control.  It reduces
1,024 length-five words to 36 sampled scalar-response classes and recovers the
class from a small probe set.  That failure does not invalidate the passive
T-mixer; it shows why neither the public word nor noncommutativity is the
secret.

### Security go/no-go tests

- the reader certifies a genuinely `K`-dimensional controlled subspace on
  which challenges are Haar-random (or proves a separate non-Haar ensemble
  bound) and `H` has a small scaled-isometry defect; singular spectrum and
  effective rank are diagnostics, not substitutes for theorem dimension;
- intra-device response remains stable and inter-device responses remain
  separated over temperature, alignment, aging, vibration, wavelength, and
  polarization;
- global-phase randomization, the conditional accepted-response energy gate,
  detector linearity, and timing checks hold under adversarial illumination;
- measured FAR/FRR agree with preregistered receiver models on held-out fresh
  challenges and devices;
- transfer-matrix tomography, replay, relay, substitution, physical cloning,
  Trojan-light, blinding, and adaptive attacks are executed at declared
  budgets; and
- the Burau mixer beats an ordinary 50:50 coupler on at least one measured
  engineering axis at equal key, loss, bandwidth, and reader trust.

The proposal is possession authentication or anti-counterfeit readout, not
encryption or unconditional unclonability.  Theft of the genuine token, a
fast low-loss coherent emulator of `H`, relay to the token, or compromise of
the verifier defeats the stated model.

## Sources anchoring the engineering claims

- Y. Zhu et al., “LightIN: a versatile silicon-integrated photonic field
  programmable gate array with an intelligent configuration framework for
  next-generation AI clusters,” *Light: Science & Applications* 15, 165
  (2026), [doi:10.1038/s41377-026-02209-5](https://doi.org/10.1038/s41377-026-02209-5).
- S. A. Goorden et al., “Quantum-secure authentication of a physical
  unclonable key,” *Optica* 1, 421--424 (2014),
  [doi:10.1364/OPTICA.1.000421](https://doi.org/10.1364/OPTICA.1.000421).
- B. Škorić, “Security analysis of Quantum-Readout PUFs in the case of
  challenge-estimation attacks,” *Quantum Information and Computation* 16,
  50--60 (2016), [ePrint 2013/479](https://eprint.iacr.org/2013/479).
- R. Pappu et al., “Physical One-Way Functions,” *Science* 297, 2026--2030
  (2002), [doi:10.1126/science.1074376](https://doi.org/10.1126/science.1074376).
- M. Akriotou et al., “Optimal performance of simple low-cost optical
  physical unclonable functions resilient to machine learning attacks,”
  *Scientific Reports* 15, 40079 (2025),
  [doi:10.1038/s41598-025-23840-z](https://doi.org/10.1038/s41598-025-23840-z).
- H. Kieninger et al., “Security assessment of photonic integrated
  circuit-based physically unclonable functions,” *Optics Express* 34,
  27621--27636 (2026),
  [doi:10.1364/OE.597136](https://doi.org/10.1364/OE.597136).
