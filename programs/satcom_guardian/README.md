# Directional Burau T SATCOM guardian: Phase-0 model

This directory contains a deterministic reduced-order engineering study of an
out-of-path optical-link guardian for inter-satellite laser communication. It
is a simulation, not a hardware result, modem model, reliability prediction,
or space-qualification claim.

The design is a **two-axis directional Burau T bank**. At the checked
8,000-km, 2.5-W corner, the
optimized bank reaches `0.9331` fault detection with the same modeled 1.5-dB
loss, essentially identical to the pre-core quadrant detector's `0.9332`. In a
paired zero-loss counterfactual it reaches `0.9688`.

The practical decision is deliberately narrower: the architecture passes the
Phase-0 performance gate and merits a matched bench prototype, but the model
does not yet establish a uniquely Burau-specific advantage. A generic pair of
balanced interferometers can measure the same directional observable, while a
quadrant detector retains much stronger flight heritage and production
maturity.

## System role

The main payload light still goes to the terminal modem. A pilot wavelength or
small receive-power tap feeds the guardian:

```text
epitaxial source -> modulator / optional booster -> transmit telescope
        -> free-space link -> receive telescope / fine steering
        |-- main power --> modem --> FEC / CRC / ARQ --> router
        `-- pilot/tap --> 50:50 x/y fanout
                         |-- directional T_x --> two detectors --|
                         `-- directional T_y --> two detectors --+--> alarm / local steering
```

The model separates two causal angular variables:

- transmitter pointing controls gross Gaussian capture;
- receiver angle of arrival controls overlap with the enrolled pupil mode.

Both have 0.25-microradian per-axis Gaussian jitter by default. The primary
stress is a receiver-AoA-only 1.5-microradian mean x bias, so the scalar-power
tap is a negative causal control rather than a competing pointing sensor.

The parameterized source propagates Lorentzian linewidth, carrier-frequency
detuning, and one-sided white RIN. It is not fitted to one laser. The cited
155-mW InP-on-Si device is a multimode Fabry--Perot research laser and does not
establish the selected-carrier power, linewidth, RIN, coupling, or amplifier
assumptions used here.

## Optimized observable

Let `h` be the enrolled piston mode and `g_x,g_y` the normalized pupil-tangent
modes. After a lossless 50:50 fanout, each axis uses a balanced T cell. The
first branch is the identity. The second applies `sigma_y` to
`span(h,g_j)` and `exp(i phi_c) I` to the orthogonal complement.

The path mixer is the exact three-letter reduced-Burau word

```text
sigma_2^-1 sigma_1 sigma_2^-1,   omega = pi/4,
```

phase-gauged to balanced sum/difference ports. The code verifies its unitarity,
balanced magnitudes, and equivalence to an ordinary balanced coupler to
floating-point precision.

For real piston amplitude `a0` and tangent coefficient `b_j` (physical field
coefficient `i b_j`), the ideal normalized output is

```text
s_j = 2 a0 b_j + cos(phi_c) (1 - a0^2 - b_j^2).
```

The first term is signed and linear at boresight. The default `phi_c=0` keeps
the complementary residual-modal evidence; `phi_c=pi/2` removes it and is
reported as a pure-signed ablation. The alarm statistic is
`sqrt(s_x^2+s_y^2)`, while the signed axes can serve as local steering errors.
The bank uses four photodiodes total, matching the QPD channel count, and all
post-core photons are allocated between the two cells.

For the stated 1550-nm wavelength and 10-cm circular aperture, the directional
T's local signed slope is `0.2027 / microradian`, versus `0.1182 /
microradian` for the default QPD model. After the 50:50 fanout and 1.5-dB loss,
its local shot-noise Fisher-information ratio to that QPD is `1.0407`. Its 1%
small-angle linear range is 1.092 microradians and its first positive-axis
monotonic limit is 8.548 microradians. These are local ideal-model quantities;
precision control needs a calibrated 2D lookup, and acquisition still needs a
wider-field PAT sensor.

## Reproduce

From the repository root:

```bash
python -m programs.satcom_guardian.reproduce
```

The run writes:

- `results/satcom_guardian.json`, containing every assumption, identity check,
  Monte Carlo result, loss sweep, design characterization, and limitation;
- `figures/satcom_guardian.png`, containing link, source, detector, and
  fault-severity panels.

The literature-routed baseline hierarchy, engineering scores, control burden,
production assessment, and distance interpretation are in
[BASELINE_REVIEW.md](BASELINE_REVIEW.md).

Calibration, nominal evaluation, and fault evaluation use independent
30,000-window populations and named independent random streams. Thresholds use
strict `score > threshold` at a diagnostic 1% per-window false-alarm target.
At 100-ns decisions that target would produce about 100,000 raw flags/s before
filtering, so it is not an operational alarm specification.

## Checked result

At 8,000 km with a parameterized 2.5-W monitored carrier, 100-ns windows, a 1%
tap, and a receiver-AoA-only 1.5-microradian bias:

| Receiver | Modeled loss/location | Detection | AUC |
|---|---:|---:|---:|
| Optimized directional Burau T | 1.5 dB, post-core | 0.9331 | 0.99601 |
| Four-quadrant detector | pre-core | 0.9332 | 0.99612 |
| Loss-matched conventional mode sorter | 1.5 dB, post-core | 0.8322 | 0.98021 |
| Scalar-power negative control | pre-core | 0.0099 | 0.50206 |

The directional T and QPD 95% conditional Wilson intervals are respectively
`[0.9302, 0.9359]` and `[0.9303, 0.9360]`. Their observed evaluation false
alarm fractions are 0.01013 and 0.00967.

The paired loss sweep reuses the exact same latent and detector streams:

| Burau-bank insertion loss | T detection | Pre-core QPD detection |
|---:|---:|---:|
| 0.0 dB | 0.9688 | 0.9332 |
| 0.5 dB | 0.9577 | 0.9332 |
| 1.0 dB | 0.9516 | 0.9332 |
| 1.5 dB | 0.9331 | 0.9332 |
| 2.0 dB | 0.9117 | 0.9332 |
| 3.0 dB | 0.8629 | 0.9332 |

At the same 1.5-dB loss, discarding the complement with `phi_c=pi/2` lowers T
detection to `0.9254`. This confirms that the retained residual term is useful
rather than an accidental implementation detail.

The 8,000-km post-core photon budget is about 959 photoelectrons per 100 ns for
the 2.5-W scenario and 59 for the optimistic 155-mW selected-carrier scenario.
The latter is photon-starved: directional-T detection is only 0.0156 in one
100-ns window. This is evidence for amplification, longer integration, or
temporal fusion—not a claim that the cited epitaxial laser directly supplies a
flight-ready carrier.

## Baselines and engineering decision

The baseline hierarchy is:

1. quadrant detector for the primary pointing comparison;
2. existing PAT plus modem/FEC telemetry as the mandatory system baseline;
3. a generic two-cell balanced interferometer as the matched directional
   control;
4. a conventional mode sorter as an independent modal-sensing baseline;
5. a pixel focal-plane sensor when acquisition FOV or multi-spot estimation
   matters;
6. scalar power only as a negative control.

The QPD remains the wiser practical baseline because it already supplies two
signed axes without optical phase control and has strong terminal heritage.
The Burau bank is competitive in the selected model and has slightly higher
local information per incident photon at 1.5 dB, but it is charged with active
phase/delay, wavelength, mode-enrollment, polarization, gain, and thermal
calibration. A build decision should therefore compare both devices at equal
incident photons, latency, detector technology, false-alarm budget, and
calibration effort.

## Distance context

No detector has one universal supported satellite separation. Photons scale
approximately as inverse range squared, and useful distance depends on power,
divergence, aperture, loss, tap fraction, detector efficiency, integration
time, background, and fault size.

On the sampled 500--8,000-km grid, the 2.5-W directional T and QPD both exceed
90% detection through 8,000 km and 99% through 4,000 km for this selected
fault. With the optimistic 155-mW carrier they exceed 90% through 2,000 km and
99% through 1,000 km. These are sampled stress-test points, not maximum ranges.

For external context, SDA OCT v4 contains 5,500-km continuous-mode and
20,000-km burst-mode reference points. Applying this repository's separate
continuous Gaussian envelope gives about 2,029 and 153 post-core
photoelectrons per 100 ns at those distances in the 2.5-W scenario. It does
not model the standard waveform, burst duty cycle, receiver sensitivity, PAT
margin, or demonstrate compliance. The assumed 1.5-dB core loss transmits
70.8% of photons and, by inverse-square scaling alone, reduces equal-photon
range to 84.1% of a lossless receiver's range.

## Main limitations

The model omits measured pupil/photonic-lantern transfer matrices, finite modal
closure, aberrations, stray light and optical background, detector bandwidth
and saturation, amplifier signal--ASE and ASE--ASE beat noise, source side
modes and optical feedback, measured T-arm dispersion and thermo-optic control,
Doppler/carrier loops, modulation/FEC, actuator dynamics, modem loss of lock,
routing/goodput, correlated alarm fusion, radiation, packaging, aging, and
launch environment.

The next decisive experiment is a matched bench test of the Burau bank, a
generic directional interferometer, and a QPD behind the same representative
telescope relay. It should measure insertion loss, bandwidth, drift,
calibration interval, warning lead time, steering recovery, electrical power,
and modal/polarization fault coverage.

## Literature anchors

- [TBIRD quad-sensor design and on-orbit pointing](https://ntrs.nasa.gov/citations/20230000001)
- [SDA Optical Communications Terminal Standard v4.0.0](https://www.sda.mil/wp-content/uploads/2024/07/SDA_OCT_Standard_4.0.0_final-20240701.pdf)
- [Spaceborne quadrant-detector spot positioning](https://doi.org/10.1364/AO.517934)
- [Two-dimensional calibrated spatial-mode displacement estimation](https://doi.org/10.1364/OPTICA.404746)
- [Free-space photonic-lantern mode-diversity reception](https://doi.org/10.1109/JPHOT.2022.3225337)
- [Self-calibrating programmable photonic circuits](https://doi.org/10.1038/s41566-022-01020-z)
