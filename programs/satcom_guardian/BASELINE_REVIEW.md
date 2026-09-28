# SATCOM guardian baseline and technology trade study

This review is a decision aid for the reduced-order simulation in this
directory. It is not a flight qualification, a formal technology-readiness
assessment, or a procurement scorecard. The literature was reviewed through
2026-09-28.

## Decision

For the modeled pointing-bias fault, the primary practical baseline is a
four-quadrant pointing detector. Existing modem/FEC/PAT telemetry is a
mandatory system baseline, a conventional nominal-mode sorter is the
matched-observable control, and a scalar power tap is only a negative control.

The present evidence does **not** show a Burau-specific advantage:

- At the checked 8,000-km, 2.5-W corner, the selected 1.5-urad fault is detected
  with probability 0.99827 by the T guardian, 0.99850 by the loss-matched
  conventional mode sorter, and 0.99820 by the quadrant detector. Their 95%
  binomial intervals overlap. The scalar power tap reaches only 0.0583.
- At 8,000 km with the 155-mW seed, the T guardian reaches 0.689, the
  loss-matched mode sorter 0.683, and the quadrant detector 0.411 in a 100-ns
  window at the default 4-urad QPD spot scale. Reducing the assumed QPD scale to
  2 urad raises its separate sensitivity run to 0.689, at the cost of physical
  field-of-view margin that this ideal model does not represent. The apparent
  low-photon advantage is therefore design-assumption dependent and is not
  unique to the T construction: the conventional sorter measures the same
  projector and nearly matches it.
- The one T score is even in pointing displacement. It raises an alarm but does
  not say whether to steer left/right or up/down. A quadrant or pixel detector
  directly supplies those two signed control errors.
- At equal loss and detector noise, the T and conventional sorter measure the
  same nominal-mode-versus-residual observable. The T implementation adds
  linewidth, differential-delay, phase, and carrier-drift sensitivities; its
  algebraic provenance does not improve the observable.

The T route should therefore remain a research candidate for broad modal-fault
monitoring, not replace the terminal's pointing sensor or modem telemetry. It
earns a build decision only if measured hardware beats both the quadrant
detector and a conventional implementation of the same projector on loss,
bandwidth, stability, fault coverage, or system SWaP.

## Baseline hierarchy

1. **Quadrant detector (primary pointing baseline).** TBIRD used a quad sensor
   to generate two-axis pointing feedback, including on-orbit closed-loop
   operation. Four-quadrant spot estimators are also an active spaceborne
   laser-communications research topic. This is the closest practical baseline
   to the fault currently simulated.
2. **Modem/FEC/PAT telemetry (mandatory system baseline).** The SDA OCT standard
   treats PAT and communications together and defines fast received-power and
   frame-sync reporting. A new guardian must add warning lead time or fault
   isolation beyond signals the terminal already has.
3. **Conventional mode sorter (matched-observable control).** A nominal-mode
   projection and complementary residual sum implement the same score as the T
   core. Photonic-lantern mode-diversity receivers provide a plausible
   implementation family, although this repository has not designed one.
4. **Pixel focal-plane sensor (secondary PAT baseline).** This is appropriate
   when acquisition field of view, multiple spots, non-Gaussian imagery, or
   estimator flexibility matters. ESA's FastSwitching work explicitly replaces
   heritage cascaded quadrant diodes with a pixel detector.
5. **Scalar received power (negative control).** It is cheap and necessary, but
   the selected fault was deliberately constructed to change mode content much
   more than total captured power. It is not a credible headline baseline.

## Reproducible detector comparison

All simulated detectors see the same pointing/polarization population and RIN
realization. Threshold calibration, nominal evaluation, and fault evaluation
use independent populations. Each threshold targets 1% false alarms; this is a
visible ROC point, not an operational requirement.

| 8,000-km receiver | 155-mW detection | 155-mW AUC | 2.5-W detection | 2.5-W AUC |
|---|---:|---:|---:|---:|
| T guardian, two post-core detectors | 0.6893 | 0.9513 | 0.99827 | 0.99988 |
| Loss-matched conventional mode sorter | 0.6828 | 0.9521 | 0.99850 | 0.99991 |
| Pre-core four-quadrant detector | 0.4106 | 0.9355 | 0.99820 | 0.99987 |
| Pre-core scalar power tap | 0.0143 | 0.5435 | 0.05833 | 0.69488 |

The ideal quadrant model uses a gapless detector and a Gaussian
difference-over-sum response

```text
dx = erf(sqrt(2) * theta_x / theta_spot)
dy = erf(sqrt(2) * theta_y / theta_spot)
anomaly = sqrt(dx^2 + dy^2)
```

with `theta_spot = 4 urad`. It is intentionally charged for four independent
read-noise and dark-current contributions, while it avoids the assumed 1.5-dB
T-core loss. A real comparison must measure the focal-plane PSF, deliberate
defocus, quadrant gaps, clipping, background, saturation, and estimator
bandwidth. The generated JSON also sweeps 2, 4, and 8 urad spot radii to expose
the sensitivity-versus-field-of-view assumption. In the 155-mW, 8,000-km
sensitivity run, their detection probabilities are 0.689, 0.398, and 0.101,
respectively. The ideal model does not charge the narrower spot for its reduced
capture/tracking field of view, so this is a sensitivity diagnostic rather than
an optimized QPD claim.

The 100-ns decision window is much faster than a spacecraft steering loop. The
comparison is fair as a same-latency detector test, but it does not establish
which architecture gives the best control performance after realistic temporal
filtering. For context, TBIRD supplied feedback at 10 Hz; modern laboratory QPD
estimators can operate much faster, but terminal dynamics and mission false
handover costs set the useful rate.

## Engineering scores

Scores run from 1 (unfavorable) to 5 (favorable). They are ordinal engineering
judgements, not measured specifications or formal TRLs. They must not be summed:
mission weights and hard gates matter more than a synthetic total.

| Candidate | Signed pointing output | Calibration/control ease | Production readiness | Coherence/wavelength robustness |
|---|---:|---:|---:|---:|
| Burau/T guardian | 1 | 2 | 1 | 2 |
| Quadrant-detector PAT | 5 | 4 | 5 | 5 |
| Pixel focal-plane PAT | 5 | 3 | 4 | 5 |
| Conventional mode sorter | 1 | 3 | 3 | 4 |
| Modem/FEC link telemetry | 1 | 4 | 5 | 4 |
| Scalar power tap | 1 | 5 | 5 | 5 |

| Candidate | Photon/loss efficiency | Fault coverage | Incremental SWaP | Evidence maturity |
|---|---:|---:|---:|---:|
| Burau/T guardian | 3 | 3 | 2 | 1 |
| Quadrant-detector PAT | 4 | 2 | 4 | 5 |
| Pixel focal-plane PAT | 3 | 3 | 3 | 4 |
| Conventional mode sorter | 3 | 4 | 2 | 3 |
| Modem/FEC link telemetry | 5 | 5 | 5 | 5 |
| Scalar power tap | 5 | 1 | 5 | 5 |

The trade is not that the quadrant detector is universally better. It is better
matched to a pointing fault and to a steering loop. The T or mode-sorter score
can respond to any departure from one enrolled spatial/polarization mode, which
may give broader precursor coverage, but it compresses the departure to one
unsigned number and cannot identify the correction by itself.

## Control and calibration burden

### T guardian

A credible implementation needs all of the following:

- characterize the complex nominal input mode and both branch transfer maps;
- balance the two output gains and losses;
- hold the recombiner at its calibrated phase and limit differential delay;
- track wavelength, temperature, polarization, coupling, and aging drift;
- re-enroll the nominal mode or prove that a fixed reference remains valid; and
- add temporal filtering, hysteresis, and either dithering or extra projections
  if a signed steering correction is required.

The checked mathematics specifies a Householder observable, not a foundry-ready
Burau chip. There is no layout, PDK, tolerance stack, packaging design,
radiation result, thermal-control budget, or demonstrated control loop. Generic
MZI meshes are manufacturable, and self-calibration has been demonstrated, but
the literature treats fabrication variation, thermal gradients, and crosstalk
as control problems rather than free properties.

### Quadrant detector

A quadrant sensor still requires an angular calibration map, gain/offset
calibration, a spot-size/FOV choice, power-validity flags, thermal testing, and
loop filtering. TBIRD performed grid scans across power and thermal conditions
and mapped normalized discriminants to two-axis angle. That is a much closer
heritage path to the present pointing use case, and no optical phase lock is
needed.

### Pixel detector and mode sorter

A pixel detector buys field of view and estimator flexibility at the cost of
more readout, computation, radiation-sensitive electronics, and often more
latency. A mode sorter buys broader modal information, but its true comparison
depends on measured insertion loss, crosstalk, detector count, and calibration.
Neither should be credited with ideal performance without a component model.

## Distance and link-budget interpretation

No guardian technology has a universal satellite separation. For every direct
detector in this study, available photons scale approximately as `1 / range^2`;
transmit power, divergence, aperture, link loss, tap fraction, detector
efficiency, integration time, and the fault distribution set the useful range.

The simulated detection grid is 500--8,000 km. For the selected 1.5-urad fault
and 2.5-W carrier, all three spatial receivers still exceed 99% detection at the
farthest sampled point. With the 155-mW seed, the T and conventional sorter
exceed 99% through 4,000 km, while the quadrant detector does so through
2,000 km under the assumed 100-ns/four-read-noise model. These are sampled
stress-test results, not maximum operating ranges.

The SDA OCT v4 standard supplies useful system anchors: 25 uW/m2 at 5,500 km
for continuous interoperable modes and 6 uW/m2 long-term average at 20,000 km
for the defined low-rate burst modes. Its maximum-output capability floor is
2.5 W. SDA's current resources page identifies v3.2 as the Tranche 3 standard
of record and v4 for a smaller set of long-range-capable terminals, so the
20,000-km point is context rather than a generic constellation requirement.
Evaluating this repository's separate Gaussian envelope at those
distances gives about 2,029 post-T-core electrons per 100 ns at 5,500 km and 153
at 20,000 km. This does **not** demonstrate OCT compliance because the model
does not implement the standard waveform, burst duty cycle, receiver
sensitivity, PAT margin, or link testing.

The assumed 1.5-dB core loss transmits 0.708 of the photons. Under pure
inverse-square, equal-photon scaling, that alone reduces range to 0.841 of a
lossless pre-core receiver's range. A fabricated mode sorter and quadrant path
will also have losses, so the final comparison must use measured end-to-end
values.

## What would change the decision

The next experiment should put the same modulated pilot through a representative
telescope/focal-plane relay and split it among:

1. the T implementation;
2. a quadrant detector;
3. a conventional nominal-mode sorter or photonic lantern; and
4. the terminal modem/PAT telemetry path.

Preregister equal incident photons, integration latency, detector technology,
false-alarm budget, and calibration effort. Sweep signed pointing ramps,
polarization, defocus/aberration, wavelength, linewidth, temperature, RIN,
gain drift, WDM lane faults, and component failures. Measure warning lead time,
missed degradation, false handovers, steering recovery, insertion loss,
electrical power, calibration interval, and delivered goodput.

A Burau-specific go decision requires at least one measured advantage that a
generic realization of the same projector does not share.

## Literature basis

- The [SDA Optical Communications Terminal Standard v4.0.0](https://www.sda.mil/wp-content/uploads/2024/07/SDA_OCT_Standard_4.0.0_final-20240701.pdf)
  defines the PAT/communications context, range/irradiance anchors, and fast
  received-power and frame-sync reports. The [SDA resources page](https://www.sda.mil/home/work-with-us/resources/)
  supplies the current v3.2/v4 deployment context.
- Riesing et al. report the [TBIRD quad-sensor design and calibration](https://doi.org/10.1117/12.2615323)
  and the subsequent [on-orbit pointing results](https://ntrs.nasa.gov/citations/20230000001).
- Wei et al., [“Spot position scheme on a quadrant detector for a spaceborne
  laser communication system”](https://doi.org/10.1364/AO.517934), demonstrate
  why a quadrant detector is a current spaceborne baseline rather than a straw
  man.
- Li et al., [“An Improved Method for the Position Detection of a Quadrant
  Detector for Free Space Optical Communication”](https://doi.org/10.3390/s19010175),
  give the conventional normalized quadrant discriminants and a low-SNR FSO
  estimator.
- Nguyen et al., [“Beam position estimation method for fast beam tracking in
  free-space optical systems using a quadrant detector”](https://doi.org/10.1364/OL.576243),
  experimentally compare QPD estimators and tracking speed.
- ESA's [FastSwitching project](https://resilience.esa.int/archives/projects/fastswitching)
  identifies heritage quadrant diodes and a pixel-detector replacement in a
  practical terminal-development context.
- Wang et al., [“Free-Space Optical Communication Based on Mode Diversity
  Reception Using a Nonmode Selective Photonic Lantern and Equal Gain
  Combining”](https://doi.org/10.1109/JPHOT.2022.3225337), provide an
  experimental mode-diversity implementation anchor.
- Xu et al., [“Self-calibrating programmable photonic integrated
  circuits”](https://doi.org/10.1038/s41566-022-01020-z), demonstrate a route to
  photonic calibration while documenting the fabrication/thermal problem it
  must solve.
