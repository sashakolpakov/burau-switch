# Epitaxial-laser / T-chip SATCOM Phase-0 model

This study models a concrete engineering use of the T architecture: an
out-of-path optical link-assurance guardian for a high-rate inter-satellite
laser terminal.  The laser is presently a parameterized source envelope, not
a rate-equation model fitted to one measured device.  This is a numerical
design study, not a hardware result or a claim of space qualification.

The modeled system chain begins with a selected carrier from an epitaxial
source.  An optional booster represents a terminal amplifier, not extra laser
output.  At the receiver, the main payload light continues to the ordinary
modem, FEC, ARQ, and router.  A dedicated pilot wavelength or small
receive-power tap feeds the guardian:

```text
epitaxial source -> modulator / optional booster -> transmit telescope
        -> free-space optical link -> receive telescope / fine steering
        |-- main power --> modem --> FEC / CRC / ARQ --> router
        `-- pilot/tap --> modal receiver --> T sum/difference detector
                                            `--> warning / reacquire / failover
```

The model propagates three assumed source properties that matter to a coherent
T interferometer:

- Lorentzian laser linewidth, through the arm visibility
  `exp(-pi * linewidth * |delay mismatch|)`;
- carrier-frequency detuning, through the differential phase accumulated by
  unequal arms; and
- relative-intensity noise (RIN), applied as common multiplicative power
  noise over the decision bandwidth.  The default treats the quoted dBc/Hz
  value as a one-sided white power-noise density and uses the `1/(2T)` noise
  bandwidth of a rectangular integrate-and-dump window.

The cited high-power InP-on-Si device is a multimode Fabry--Perot research
laser and does not supply the linewidth or RIN defaults used here.  Those
defaults define a hypothetical selected carrier.  A device-faithful model
must ingest its measured optical spectrum, RIN, L--I curves, and temperature
and feedback response; for a multimode spectrum the coherence function must
come from the Fourier transform of that spectrum rather than one Lorentzian.

It then adds a Gaussian-beam link budget, pointing-dependent capture,
pointing and polarization leakage into four received modes, guardian tap and
insertion losses, detector quantum efficiency, Poisson shot noise, read noise,
and detector-gain mismatch.  The link budget uses the small-receive-aperture
approximation

```text
P_rx / P_tx = eta_link * D_rx^2 / (2 * (theta * range)^2)
              * exp(-2 * (pointing / theta)^2).
```

Here `theta` is the Gaussian beam's 1/e^2-intensity half-angle and `D_rx` is
the receiver diameter.  The code uses the exact centered circular-aperture
factor and the displayed small-aperture pointing approximation; the default
beam radius is at least 7.5 m versus a 0.1-m aperture.  This is not an optical
terminal design code.  It omits the general off-axis aperture integral,
aberrations, stray light, amplifier noise, Doppler tracking loops, coding, and
network routing.

## Exact T observable

Let `P0` project onto the nominal received spatial/polarization mode and set

```text
Q = 2 P0 - I,       X = I,       Y = Q.
```

Both branch maps are unitary.  For

```text
u = (X + Y)x / 2,       v = (X - Y)x / 2,
```

the ideal complementary powers obey

```text
I_plus - I_minus = x^dagger Q x,
I_plus + I_minus = ||x||^2.
```

Thus the normalized difference is a nominal-mode-versus-residual score while
the sum is an energy checksum.  This chosen score needs only a phase flip
between the nominal and residual subspaces; it does not require the current
132-letter general Burau compiler.  For this observable it is mathematically
equivalent to an ideal conventional nominal-mode sorter.  The model tests the
value of modal monitoring over scalar power monitoring; it does not establish
an advantage over another implementation of the same mode projection.

## Numerical experiments

Run from the repository root:

```bash
python -m programs.satcom_guardian.reproduce
```

The deterministic run writes:

- `figures/satcom_guardian.png`, with the photon link budget, coherence and
  detuning sensitivities, practical detector comparison, fault-severity sweep,
  and RIN test;
- `results/satcom_guardian.json`, containing every input parameter and
  computed result.

The literature-routed baseline choice, engineering scores, control and
production assessment, and range interpretation are recorded in the
[baseline review](BASELINE_REVIEW.md).

The default comparison is intentionally difficult for a scalar received-power
alarm: a selected 1.5-urad pointing-bias stress case redistributes the coherent
field among collected modes while causing only a small total-power change.
This bias is six times the assumed 0.25-urad nominal jitter and 0.375 times the
assumed 4-urad modal scale; it is not a fitted distribution of incipient flight
faults.  A separate sweep includes smaller biases.  At each range, independent
threshold-calibration, nominal-evaluation, and fault-evaluation Monte Carlo
populations compare:

- the normalized two-output T residual after the assumed 1.5-dB core loss;
- a loss-matched conventional mode sorter measuring the same projector without
  T-arm coherence sensitivity;
- a four-segment Gaussian-spot quadrant detector before core loss; and
- a one-detector scalar power tap before core loss, retained only as a negative
  control.

The quadrant detector is the primary practical baseline because the simulated
fault is pointing bias and a quadrant sensor returns signed azimuth/elevation
errors. Existing modem/FEC/PAT telemetry is a mandatory system baseline, but it
cannot yet be simulated fairly because this Phase-0 model has no symbol
waveform, frame synchronizer, or decoder.

The threshold is calibrated at a target 1% false-alarm probability and then
evaluated on an independent nominal population.  At 100-ns decisions, 1%
corresponds to roughly 100,000 flagged windows per second before voting or
hysteresis.  It is therefore only a visible ROC operating point, not an
operational SATCOM requirement.  A deployable guardian needs a mission-derived
false-alarm, miss, latency, correlation, and undetected-fault budget.

## Baseline result

Under the checked-in assumptions, the 1% guardian tap costs an idealized
0.0436 dB from the main path before splitter excess loss.  At 8,000 km it
collects about 59 photoelectrons per 100-ns decision window from a monitored
155-mW seed carrier, or 959 photoelectrons after an illustrative boost of that
carrier to 2.5 W.  These are per-monitored-carrier powers, not aggregate WDM
terminal powers.  The boost is a system scenario, not a claim that the cited
epitaxial laser itself emits 2.5 W; amplifier noise is not yet in the model.

For the selected 1.5-urad pointing fault, total collected power falls only
1.98%, but the assumed four-mode receiver sees a much larger redistribution.
Using thresholds fitted on independent nominal samples, the 8,000-km boosted
case gives:

- T detection 0.99827 (95% binomial interval 0.99773--0.99868) with an observed
  false-alarm fraction of 0.00950;
- loss-matched conventional-mode-sorter detection 0.99850
  (0.99799--0.99888), statistically indistinguishable from T;
- pre-core quadrant-detector detection 0.99820 (0.99765--0.99862), also
  statistically indistinguishable for this large fault; and
- pre-core scalar-power detection 0.0583 (0.0557--0.0610).

The seed-only 8,000-km case falls to 0.689 T detection, 0.683 for the
conventional sorter, and 0.411 for the four-segment detector in the same 100-ns
window at the default 4-urad QPD spot scale. A separate QPD sensitivity run
reaches 0.689 with a 2-urad spot assumption, whose reduced physical field of
view is not charged by this ideal model. The apparent low-photon ranking is
therefore design-assumption dependent. The near-equal conventional-sorter
result also means it is not a Burau-specific advantage. It is useful evidence
for adding photons, lengthening the integration window, or voting across
windows; it is not evidence for mission-level reliability.

Most importantly, the result depends on assumed 4-urad modal and quadrant-spot
scales. Those must be replaced by measured telescope, focal-plane, and
photonic-lantern fields before the detection result can support a device
decision. The JSON includes a 2/4/8-urad quadrant-spot sensitivity check. At the
boosted 8,000-km corner, T detection falls to 0.845 at 1.0 urad, 0.517 at 0.75
urad, 0.184 at 0.5 urad, and 0.037 at 0.25 urad; the quadrant and conventional
sorter curves are similar under the default assumptions.

The source/interferometer coupling itself looks forgiving for the hypothetical
narrowband baseline: a 1-MHz Lorentzian linewidth with 1-ps arm mismatch retains
0.999997 ideal visibility.  Broad or multimode emission is different: the
model predicts that a 5-GHz linewidth needs less than 0.64 ps mismatch for
99% visibility.  For the default 1-ps mismatch, a 99% score-gain limit occurs
at 22.53 GHz of drift from the calibrated carrier (about 0.296 K if one applies
the prototype's 76-GHz/K low-end stage-temperature slope).  On-chip
thermo-optic phase drift is not modeled.  The cited Fabry--Perot prototype's
large thermal wavelength
shift also means that wavelength selection or locking remains a system
requirement for dense WDM.

## Distance context

Distance is a link-budget outcome, not a fixed property of any guardian. The
present detector grid covers 500--8,000 km and available photoelectrons scale
approximately as inverse range squared. Under the selected 1.5-urad stress and
2.5-W scenario, T, the loss-matched sorter, and the quadrant detector all remain
above 99% detection at the farthest sampled point. With the 155-mW seed, T and
the sorter remain above 99% through 4,000 km, while the quadrant model does so
through 2,000 km in the assumed 100-ns window. These are sampled points, not
maximum supported distances.

For external context, SDA OCT v4 specifies a 5,500-km continuous-mode
irradiance point and 20,000-km low-rate burst modes. Applying this repository's
separate continuous Gaussian envelope at those distances gives about 2,029 and
153 post-core photoelectrons per 100 ns, respectively, in the 2.5-W scenario.
That calculation does not model the standard waveform, burst duty cycle,
receiver sensitivity, PAT margin, or demonstrate compliance. The assumed
1.5-dB T-core loss transmits 70.8% of incident photons and, by inverse-square
scaling alone, reduces equal-photon range to 84.1% of a lossless pre-core
sensor's range.

To explore another deterministic configuration from Python, construct a
`GuardianConfig` and pass it to `run_satcom_guardian_study`; the generated
baseline files are overwritten deliberately:

```python
from programs.satcom_guardian.reproduce import (
    GuardianConfig,
    run_satcom_guardian_study,
)

run_satcom_guardian_study(
    GuardianConfig(
        laser_linewidth_hz=10e6,
        t_arm_delay_mismatch_ps=10.0,
        guardian_tap_fraction=0.02,
    )
)
```

## Interpretation boundaries

- The T core observes a pilot or tapped field; it does not carry or decode the
  payload and therefore cannot increase Shannon capacity.
- The quadrant detector is the primary pointing baseline, existing
  fine-pointing and modem/FEC telemetry are mandatory system baselines, and the
  conventional mode sorter is the matched-observable control. The scalar tap is
  only a negative control.
- One T score is unsigned and cannot close a two-axis pointing loop without
  additional projections, dithering, or another pointing sensor. Its potential
  role is link-assurance alarm generation, not replacement of PAT.
- A faster warning can improve delivered goodput only if the terminal and
  network can use it to steer, change lane, or reroute before loss of lock.
- Each optical carrier must remain coherent with itself across the two T arms.
  Mutually incoherent WDM carriers can contribute additive port powers for
  this wavelength-preserving score; they need not be mutually phase locked.
  Lane-specific scores require demultiplexing, and wavelength-dependent T-arm
  errors still require per-lane calibration.
- The default Lorentzian coherence calculation is safest for a selected
  narrowband CW pilot.  A tapped modulated payload requires the planned
  frequency-dependent waveform and differential-delay model.
- The sum/difference identity is exact only for the ideal branch model.
  Measured loss, dispersion, imbalance, detector mismatch, and aging must be
  calibrated and monitored.
- Only the feed-forward optical weighting core can be passive.  The laser,
  amplifier, detector, TIA, controller, fine-steering hardware, and redundant
  modem remain active.

## Engineering anchors

The default values are explicit simulation assumptions, selected to expose
tradeoffs rather than to describe a particular qualified part.  The following
primary and official results anchor the explored regime:

- NASA's TBIRD mission demonstrated a 200-Gbit/s space-to-ground optical link:
  [NASA TBIRD](https://www.nasa.gov/centers-and-facilities/goddard/nasa-partners-achieve-fastest-space-to-ground-laser-comms-link/).
- SDA OCT v4 defines C-band terminal wavelengths on a 100-GHz grid and a
  2.5-Gbaud interoperable waveform family, treats PAT as part of the physical
  layer, and supplies the 5,500/20,000-km range anchors used above:
  [SDA OCT v4](https://www.sda.mil/wp-content/uploads/2024/07/SDA_OCT_Standard_4.0.0_final-20240701.pdf).
- TBIRD used a quad sensor for two-axis pointing feedback and subsequently
  demonstrated closed-loop pointing on orbit:
  [Riesing et al. (2023)](https://ntrs.nasa.gov/citations/20230000001).
- A spaceborne quadrant-detector positioning scheme was designed around limited
  onboard compute and memory:
  [Wei et al. (2024)](https://doi.org/10.1364/AO.517934).
- Experimental non-mode-selective photonic-lantern reception anchors the
  conventional mode-diversity alternative:
  [Wang et al. (2023)](https://doi.org/10.1109/JPHOT.2022.3225337).
- A wafer-scale InP-on-silicon platform demonstrated 1.55-um electrically
  pumped CW lasers above 155 mW per facet and operation to 120 C:
  [Sun et al. (2024)](https://doi.org/10.1038/s41377-024-01389-2).
- Directly grown quantum-dot lasers on patterned 300-mm silicon demonstrated
  approximately 1.3-um emission, 126.6-mW double-side output, and CW lasing to
  60 C:
  [Shang et al. (2022)](https://doi.org/10.1038/s41377-022-00982-7).
- Foundry silicon photonic circuits were characterized before and after an
  approximately eleven-month exposure outside the ISS; the component-dependent
  changes motivate explicit radiation and end-of-life margins rather than a
  blanket radiation-hard claim:
  [Mao et al. (2024)](https://doi.org/10.1126/sciadv.adi9171).

## What would make the route credible

The next model should ingest a measured laser spectrum/RIN trace and a measured
complex transfer matrix.  The first bench experiment should then compare the
T score with a quadrant detector, a conventional implementation of the same
mode projector, and modem/PAT telemetry under controlled pointing,
polarization, detuning, temperature, and component faults. Incident photons,
integration latency, detector technology, and false-alarm cost must be matched.
The relevant outputs are signed control usefulness, warning lead time, missed
degradation, false handovers, added optical loss, calibration interval,
manufacturing tolerance, and end-to-end power--not optical propagation latency
alone. A Burau-specific go decision requires a measured advantage that a generic
projector does not share.
