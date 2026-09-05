# Passive security device: attack-first feasibility track

This track is separate from the Burau--anyon Gedankenexperiment manuscript.
It asks whether a braid-controlled passive interferometer can become an
authentication device.

The current answer is deliberately conservative. A public, deterministic
$2\times2$ Burau representation and its $4\times4$ switch are not a security
primitive: their responses are low-dimensional, inexpensive to simulate, and
open to parameter estimation or direct model matching. Noncommutativity does
not create one-wayness.

The first numerical probe therefore gives an attacker the correct model
family, hides a short braid word, and measures how few challenge-response
pairs are needed to recover its complete response curve across three assumed
measurement-noise levels. Run it from the repository root with:

    python -m programs.security.reproduce

It writes an attack summary and figure under this directory. The response is
the estimated scalar Helstrom contrast, so the noise sweep stands in for the
finite number of physical trials required to estimate that probability. The
result is a baseline to defeat, not evidence of security.

In the default run, the 1,024 length-five words collapse to only 36 distinct
sampled scalar-response classes; the largest equivalence class contains 220
words. Ten response-space components explain 99% of the variance. With
independent Gaussian response noise of standard deviation $0.002$, exhaustive
model matching identifies the exact response class in about 97.3% of 1,500
trials after 16 selected challenges. At noise $0.0005$, the rate is about
99.9%. This does not model every laboratory imperfection, but it rules out
treating the present public low-dimensional map as a standalone secret.

## What could still become a device

The credible direction is a physically keyed challenge-response transducer in
which the braid-controlled layer interrogates inaccessible, high-dimensional
fabrication disorder. Challenges could vary phase or frequency, prepared
state, path, and generator order; responses could include intensities and both
commutator quadratures. The phase-only channel is useful as a spoofing and
calibration control.

Such a prototype must be evaluated under a stated attacker model. At minimum:

1. repeatability under drift and shot noise;
2. inter-device separation and response entropy;
3. model extraction from chosen challenge-response pairs;
4. full transfer-matrix tomography and emulation attacks;
5. replay, side-channel, and helper-data attacks; and
6. authentication false-accept and false-reject rates.

The decisive gate is simple: if an attacker can learn the optical transfer
map within the verifier's tolerance, the device is not a strong PUF. A recent
security analysis makes this concern explicit for linear integrated photonic
PUFs, while optical-scattering PUF experiments show why challenge decorrelation
and high-dimensional disorder matter.

## Primary references

- R. Pappu et al., “Physical One-Way Functions,” *Science* 297, 2026–2030
  (2002), [doi:10.1126/science.1074376](https://doi.org/10.1126/science.1074376).
- H. Kieninger et al., “Security assessment of photonic integrated
  circuit-based physically unclonable functions,” *Optics Express* 34,
  27621–27636 (2026),
  [doi:10.1364/OE.597136](https://doi.org/10.1364/OE.597136).
- M. Akriotou et al., “Optimal performance of simple low-cost optical
  physical unclonable functions resilient to machine learning attacks,”
  *Scientific Reports* 15 (2025),
  [doi:10.1038/s41598-025-23840-z](https://doi.org/10.1038/s41598-025-23840-z).
