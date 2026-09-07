# Classical device blueprint

This is an engineering note, not part of the Burau--anyon Gedankenexperiment
manuscript and not a claim of demonstrated performance. Both proposed devices
operate with bright coherent light and ordinary photodiodes or cameras. No
single-photon source, heralding, coincidence counter, entanglement, or quantum
computer is required.

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
  $d$ detector outputs per task-compiled score or at least $d(d+1)$ outputs
  for a universal fixed-bank construction.

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

## Security device: use physical disorder, not the public Burau map

A realistic security prototype is a classical optical challenge-response
token:

    diode laser
      -> spatial or modal challenge encoder
      -> Burau/order-dependent coherent preprocessor
      -> sealed high-dimensional random scatterer
      -> CMOS speckle camera or detector array
      -> extractor and authenticated verifier

The inaccessible scatterer—not the known Burau representation—must carry the
device-specific entropy. The Burau layer can enlarge or organize the challenge
family and provide order/phase control channels, but it cannot supply
unclonability by itself. The checked-in attack already reduces all 1,024
length-five words to 36 sampled scalar-response classes and recovers the class
with high probability from a small number of responses.

This is also an ordinary bright-light instrument. A recent experimental
scattering PUF used a 100 mW, 635 nm fiber-coupled diode laser, a 128x128 subset
of a digital micromirror device, a silica diffuser, and a cooled CMOS camera.
That free-space layout is a sensible security test bed before attempting an
integrated token.

The immediate prototype should enroll a sealed diffuser or multimode element,
issue uncorrelated high-dimensional challenges, derive repeatable response
bits with documented helper data, and test authentication under temperature,
alignment, aging, vibration, and source-power variation. It must then be
attacked with chosen-challenge model extraction, transfer-matrix tomography,
replay, emulation, and physical cloning attempts.

Linear integrated photonic PUFs deserve particular skepticism: a 2026
security analysis learned a representative simulated linear construction from
as few as 200 challenge-response pairs. The current Burau-only model is
therefore a negative control, not a security core.

### Security go/no-go tests

- stable intra-device response with separated inter-device distributions;
- false-accept and false-reject rates at the intended environmental envelope;
- independent entropy estimates after helper-data leakage is accounted for;
- failure of strong model-extraction and tomography attacks at a declared CRP
  budget; and
- a complete protocol analysis including replay resistance, rate limiting,
  enrollment trust, verifier compromise, and tamper response.

If the random medium is removed, or if its accessible linear transfer map can
be learned within verifier tolerance, this security direction is a no-go.

## Sources anchoring the engineering claims

- Y. Zhu et al., “LightIN: a versatile silicon-integrated photonic field
  programmable gate array with an intelligent configuration framework for
  next-generation AI clusters,” *Light: Science & Applications* 15, 165
  (2026), [doi:10.1038/s41377-026-02209-5](https://doi.org/10.1038/s41377-026-02209-5).
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
