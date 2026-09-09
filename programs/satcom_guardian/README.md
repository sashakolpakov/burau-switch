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
  detuning sensitivities, detector operating characteristic, fault-severity
  sweep, and RIN test;
- `results/satcom_guardian.json`, containing every input parameter and
  computed result.

The default comparison is intentionally difficult for a scalar received-power
alarm: a selected 1.5-urad pointing-bias stress case redistributes the coherent
field among collected modes while causing only a small total-power change.
This bias is six times the assumed 0.25-urad nominal jitter and 0.375 times the
assumed 4-urad modal scale; it is not a fitted distribution of incipient flight
faults.  A separate sweep includes smaller biases.  At each range, independent
threshold-calibration, nominal-evaluation, and fault-evaluation Monte Carlo
populations compare:

- the normalized T/ideal-mode-sorter residual; and
- a one-detector scalar power tap placed before T-core insertion loss.

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

- T/mode-sorter detection 0.99827 (95% binomial interval 0.99773--0.99868) with
  an observed false-alarm fraction of 0.00950;
- pre-core scalar-power detection 0.0609 (0.0583--0.0637) with an observed
  false-alarm fraction of 0.0102.

The seed-only 8,000-km case falls to 0.689 T detection.  This is useful design
evidence for adding photons, lengthening the integration window, or voting
across windows; it is not evidence for mission-level reliability.  Most
importantly, the result depends on the assumed 4-urad modal scale.  That scale
must be replaced by measured telescope and photonic-lantern fields before the
detection result can support a device decision.  The severity sweep makes the
dependence visible: at the boosted 8,000-km corner, detection falls to 0.845 at
1.0 urad, 0.517 at 0.75 urad, 0.184 at 0.5 urad, and 0.037 at 0.25 urad.

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
- The scalar baseline uses the same optical tap but avoids T-core insertion
  loss and uses one detector rather than two.  Existing fine-pointing and modem
  telemetry, a quadrant detector, and a digital matched filter remain mandatory
  system baselines.
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
  2.5-Gbaud interoperable waveform family:
  [SDA OCT v4](https://www.sda.mil/wp-content/uploads/2024/07/SDA_OCT_Standard_4.0.0_final-20240701.pdf).
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
T score with a quadrant detector and modem telemetry under controlled
pointing, polarization, detuning, temperature, and component faults.  The
relevant outputs are warning lead time, missed degradation, false handovers,
added optical loss, calibration interval, and end-to-end power--not optical
propagation latency alone.
