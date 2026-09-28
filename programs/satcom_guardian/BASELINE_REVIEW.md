# SATCOM guardian baseline and technology trade study

This is a decision aid for the reduced-order simulation in this directory, not
a flight qualification, formal TRL assessment, or procurement scorecard. The
literature review date is 2026-09-28.

## Decision

The optimized directional Burau T bank **passes the Phase-0 performance gate**.
At 8,000 km, 2.5 W, a receiver-AoA-only 1.5-microradian bias, and the diagnostic
1% false-alarm point, it reaches `0.9331` detection after 1.5 dB of modeled
loss. The pre-core four-quadrant detector reaches `0.9332`; their AUCs and
conditional confidence intervals are likewise almost identical. With the
Burau-bank loss set to zero and all random streams held fixed, T detection
rises to `0.9688`.

That is a strong architecture result, but not yet a Burau-specific hardware
win. A generic pair of balanced interferometers realizes the same signed
piston/tangent observable. The right next decision is a matched prototype, not
replacement of the terminal pointing sensor.

The quadrant detector remains the primary practical baseline because it:

- measures the same two signed pointing axes;
- does not require coherent arm-phase control;
- has direct optical-terminal heritage;
- is simpler to manufacture, calibrate, and qualify today.

Existing PAT plus modem/FEC telemetry is the mandatory system baseline. The
generic directional interferometer is the matched-observable control. The
older radial Householder construction and its loss-matched mode sorter are
ablations, not the optimized Burau T. Scalar power is only a negative control.

## Why the old Burau T looked weak

The previous receiver measured

```text
2 |<h,x>|^2 - 1.
```

That score is even in displacement and quadratic at boresight, so it discards
the signed first-order information needed for fine pointing. Its `0.8270`
detection result is therefore not evidence that the optimized T architecture
is poor.

The corrected device uses two T cells. For each axis, `sigma_y` mixes the
enrolled piston and corresponding pupil-tangent mode, while the default `+I`
on the modal complement keeps residual-fault evidence:

```text
s_j = 2 a0 b_j + (1 - a0^2 - b_j^2).
```

The first term supplies a signed linear steering signal. The second preserves
the broader guardian response. Replacing `+I` by the pure-signed quadrature
ablation lowers detection from `0.9331` to `0.9254`, so both terms contribute.

The path mixer is exactly the reduced-Burau word
`sigma_2^-1 sigma_1 sigma_2^-1` at `omega=pi/4`. It is phase-equivalent to a
balanced coupler, which is why a generic interferometer must be tested beside
the Burau implementation.

## Reproducible detector comparison

All receivers use the same latent populations and independent named detector
streams. Threshold calibration, nominal evaluation, and fault evaluation are
independent 30,000-window populations. Detection uses strict exceedance of a
threshold calibrated at a nominal 1% false-alarm target.

| 8,000-km, 2.5-W receiver | Detectors | Detection | AUC | Evaluation FA |
|---|---:|---:|---:|---:|
| Directional Burau T, post-core | 4 | 0.9331 | 0.99601 | 0.01013 |
| Pre-core quadrant detector | 4 | 0.9332 | 0.99612 | 0.00967 |
| Radial Householder ablation | 2 | 0.8270 | 0.98056 | 0.00973 |
| Loss-matched radial mode sorter | 2 | 0.8322 | 0.98021 | 0.01120 |
| Pre-core scalar-power control | 1 | 0.0099 | 0.50206 | 0.00827 |

The directional-T detection interval is `[0.9302, 0.9359]`; the QPD interval
is `[0.9303, 0.9360]`. These Wilson intervals are conditional on each realized
calibration threshold and do not represent model or manufacturing uncertainty.

### Loss is the current crossover variable

| T-bank loss | Directional T | Pre-core QPD |
|---:|---:|---:|
| 0.0 dB | 0.9688 | 0.9332 |
| 0.5 dB | 0.9577 | 0.9332 |
| 1.0 dB | 0.9516 | 0.9332 |
| 1.5 dB | 0.9331 | 0.9332 |
| 2.0 dB | 0.9117 | 0.9332 |
| 3.0 dB | 0.8629 | 0.9332 |

This paired sweep is a counterfactual, not a claim that a zero-loss chip can be
built. It does show the engineering target cleanly: reducing end-to-end T-bank
loss below roughly the present 1.5-dB assumption turns parity into an advantage
in this selected detector model.

### Local control information

The ideal directional T has boresight slope `0.2027 / microradian`; the default
13.5-microradian-spot QPD has `0.1182 / microradian`. After the 50:50 x/y
fanout and 1.5-dB loss, the modeled local shot-noise Fisher information ratio
is `1.0407` in favor of the T axis. The T response stays within 1% of its linear
approximation to 1.092 microradians and remains positive-axis monotonic to
8.548 microradians.

Those values do not by themselves establish closed-loop superiority. The
retained complement creates calibrated quadratic cross-terms, and neither
model includes actuator dynamics, acquisition, aberration, background,
saturation, detector bandwidth, or temporal filtering. A real control test
must compare settling time, residual jitter, capture range, false reacquisition,
and stability margins.

## Baseline hierarchy

1. **Quadrant detector: primary practical pointing baseline.** TBIRD used a
   quad sensor for two-axis feedback and demonstrated closed-loop pointing on
   orbit. It is the closest fielded comparator to the simulated fault.
2. **Existing PAT plus modem/FEC telemetry: mandatory system baseline.** A new
   guardian must add useful lead time, fault isolation, or recovery—not merely
   duplicate received-power and frame-status signals.
3. **Generic two-cell balanced interferometer: matched directional control.**
   It implements the same four-output statistic without Burau provenance and
   is essential for identifying any genuinely Burau-specific hardware benefit.
4. **Radial mode sorter: matched ablation control.** It measures the same even
   nominal-versus-residual observable as the older Householder construction.
5. **Pixel focal plane: secondary PAT baseline.** It is the stronger comparator
   when acquisition FOV, multiple spots, or non-Gaussian imagery matters.
6. **Scalar power: negative control.** It is inexpensive and operationally
   useful, but the selected receiver-AoA fault does not materially change gross
   captured power.

## Engineering scores

Scores run from 1 (unfavorable) to 5 (favorable). They are ordinal engineering
judgements, not measured specifications or formal TRLs, and must not be summed.

| Candidate | Signed axes | Control/calibration ease | Production readiness | Coherence robustness |
|---|---:|---:|---:|---:|
| Directional Burau T | 5 | 2 | 1 | 2 |
| Quadrant-detector PAT | 5 | 4 | 5 | 5 |
| Pixel focal-plane PAT | 5 | 3 | 4 | 5 |
| Conventional mode sorter | 1 | 3 | 3 | 4 |
| Modem/FEC telemetry | 1 | 4 | 5 | 4 |
| Scalar power tap | 1 | 5 | 5 | 5 |

| Candidate | Photon/loss efficiency | Fault coverage | Incremental SWaP | Evidence maturity |
|---|---:|---:|---:|---:|
| Directional Burau T | 3 | 3 | 2 | 1 |
| Quadrant-detector PAT | 4 | 2 | 4 | 5 |
| Pixel focal-plane PAT | 3 | 3 | 3 | 4 |
| Conventional mode sorter | 3 | 4 | 2 | 3 |
| Modem/FEC telemetry | 5 | 5 | 5 | 5 |
| Scalar power tap | 5 | 1 | 5 | 5 |

The T now earns the maximum signed-output score; that is the key correction
from the radial design. It still scores poorly on production and evidence
because no foundry layout, PDK tolerance stack, packaged phase-control loop,
radiation result, or flight test exists. Those scores should change only with
hardware evidence.

## Control and production burden

### Directional Burau T

A credible implementation must:

- enroll the complex piston and two tangent modes;
- balance the x/y fanout and four output gains/losses;
- hold both recombiners at their calibrated phases and control differential
  delay across wavelength and temperature;
- track polarization, coupling, wavelength, thermal drift, and aging;
- calibrate a 2D signed-output lookup including complement cross-terms;
- provide wide-field acquisition or handoff to the terminal PAT sensor;
- implement temporal filtering, validity checks, hysteresis, and fail-safe
  behavior.

The exact three-letter word makes the optical path mixer shallow, but it does
not remove packaging, heaters, monitors, TIAs, controller electronics, or
qualification work. Generic programmable photonic circuits can self-calibrate;
the literature also shows that fabrication variation, thermal gradients, and
crosstalk are real control burdens.

### Quadrant detector

A QPD still needs spot-size/FOV design, normalized gain/offset maps, detector
gap and saturation characterization, power-validity logic, thermal testing,
loop filtering, and handoff logic. Its advantage is not perfection; it is a
much shorter heritage path and no need for an optical phase lock.

### Production comparison

The matched prototype should report end-to-end insertion loss—not only waveguide
propagation loss—plus yield, phase-heater power, calibration time and interval,
thermal sensitivity, polarization dependence, bandwidth, detector/TIA count,
package stability, radiation drift, and failure modes. Without these numbers,
the 1.5-dB crossover cannot support a production claim.

## Distance and link-budget interpretation

There is no universal supported inter-satellite distance for any guardian.
Useful range depends on transmit power, divergence, aperture, link loss, tap
fraction, detector efficiency, integration time, background, and fault size;
available photons scale approximately as `1/range^2` here.

For the selected fault on the sampled 500--8,000-km grid:

- at 2.5 W, both the directional T and QPD exceed 90% detection through 8,000
  km and 99% through 4,000 km;
- with the optimistic 155-mW selected carrier, both exceed 90% through 2,000
  km and 99% through 1,000 km.

These are sampled model points, not maximum operating ranges. The 155-mW,
8,000-km corner is photon-starved in a single 100-ns window, with directional-T
detection only 0.0156.

SDA OCT v4 provides useful 5,500-km continuous and 20,000-km burst reference
points. Evaluating the repository's separate Gaussian envelope at those
distances produces about 2,029 and 153 post-core electrons per 100 ns for the
2.5-W scenario. This does not demonstrate compliance because the model lacks
the standard waveform, burst duty cycle, sensitivity, PAT margin, and test
procedure.

The assumed 1.5-dB T loss transmits 70.8% of incident photons. Under pure
inverse-square equal-photon scaling, it reduces range to 84.1% of a lossless
receiver. Real QPD and generic-interferometer paths also have nonzero losses;
the bench comparison must use measured end-to-end values for every candidate.

## Go/no-go experiment

Put the same pilot behind a representative telescope/focal-plane relay and
split repeatable trials among:

1. the reduced-Burau directional T implementation;
2. a generic two-cell balanced interferometer;
3. a quadrant detector;
4. the terminal PAT and modem/FEC telemetry path.

Preregister equal incident photons, detector technology, integration latency,
false-alarm budget, and calibration labor. Sweep signed pointing ramps,
polarization, defocus and aberration, wavelength, linewidth, temperature, RIN,
background, gain drift, WDM lane faults, and component failures. Measure
warning lead time, missed degradation, false handovers, steering recovery,
insertion loss, electrical power, drift, calibration interval, and delivered
goodput.

A Burau-specific go decision requires a measured advantage in loss, bandwidth,
stability, fault coverage, calibration, or SWaP that the generic matched
interferometer does not share.

## Literature basis

- [SDA Optical Communications Terminal Standard v4.0.0](https://www.sda.mil/wp-content/uploads/2024/07/SDA_OCT_Standard_4.0.0_final-20240701.pdf)
- [TBIRD quad-sensor design and on-orbit pointing](https://ntrs.nasa.gov/citations/20230000001)
- [Spaceborne quadrant-detector spot positioning](https://doi.org/10.1364/AO.517934)
- Boucher et al., [two-dimensional calibrated spatial-mode displacement estimation](https://doi.org/10.1364/OPTICA.404746)
- [Free-space photonic-lantern mode-diversity reception](https://doi.org/10.1109/JPHOT.2022.3225337)
- [Self-calibrating programmable photonic circuits](https://doi.org/10.1038/s41566-022-01020-z)
- [ESA FastSwitching terminal development](https://resilience.esa.int/archives/projects/fastswitching)
