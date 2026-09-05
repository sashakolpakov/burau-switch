# Fast passive inference: quadratic-feature feasibility track

This track is separate from the Burau--anyon Gedankenexperiment manuscript
and from the security-device track. It asks what a passive network assembled
from Burau--Squier blocks can compute during neural inference.

There is an exact expressivity boundary. Let a passive optical circuit have
transfer matrix $U$, followed by square-law detection and a linear readout.
Every output score has the form

$$
s(x)=b+\sum_j a_j |(Ux)_j|^2=b+x^\dagger Qx.
$$

Cascading more passive linear components before the same detector does not
change this: their product is another linear transfer matrix. A bank of
circuits can provide many quadratic measurements, but it does not by itself
implement an arbitrary multilayer neural network.

There is also an exact setting-count bound. In $d$ modes, the $d$ intensity
projectors from one unitary setting sum to the identity. After the first
setting has supplied that direction, every additional orthonormal basis can
add at most $d-1$ independent Hermitian directions. Consequently, $k$
settings have rank at most

$$
\min\{d^2,\ d+(k-1)(d-1)\},
$$

and a complete $d^2$-dimensional quadratic bank needs at least $d+1$
settings (for $d>1$).

This is a lower bound for a **fixed, task-independent optical feature bank**
whose electronic readout may later be retrained to any $Q$. It is not a lower
bound for one known score. Once a particular Hermitian $Q$ is trained, the
spectral decomposition $Q=V\Lambda V^\dagger$ gives

$$
x^\dagger Qx=\sum_j \lambda_j |(V^\dagger x)_j|^2,
$$

so one programmable unitary mesh and $d$ detectors suffice for that score.
Several noncommuting output scores generally require several meshes, unless
they share an eigenbasis. Compiling an arbitrary trained eigenbasis into the
restricted Burau-block family is a separate reachability problem; the present
rank experiment does not solve it.

The first numerical experiment asks a narrower, useful question: do
four-mode meshes assembled only from two-mode unitarized Burau blocks span the
full $4^2=16$ dimensional real vector space of Hermitian quadratic forms? It
compares their measurement rank and regression error with phase-only and Haar
unitary baselines. Run it from the repository root with:

    python -m programs.inference.reproduce

The experiment fits both a quadratic target, which a complete passive feature
bank should recover exactly, and a quartic target, which exposes the ceiling.
It also repeats the minimum-size Burau-bank construction over independent
random draws, so the reported full rank is not evidence from a single lucky
mesh.

With the default seed, both the Burau and Haar banks attain ranks
$4,7,10,13,16$: the Burau bank reaches the complete four-mode quadratic space
using the rank-bound minimum of five settings. All 32 independently drawn
five-setting Burau banks are full rank. The quadratic target is recovered to
numerical precision, while the normalized test RMSE on the quartic target is
about $0.66$. The phase-only baseline remains rank four and fails even on a
generic quadratic target. These are numerical results for the stated
ensemble, not a hardware speed or energy advantage.

Full rank is not the same as noise robustness. Across those 32 minimum banks,
the median condition number is about 46.7 for Burau meshes and 51.7 for Haar
meshes, with long ill-conditioned tails in both ensembles. The separate
[classical digital twin](../digital_twin/README.md) therefore compares the
minimum bank with an overcomplete eight-setting design under detector noise
and coherent drift.

## Device interpretation

A positive result supports a fixed-feature or kernel front end:

1. encode inputs as coherent mode amplitudes;
2. fan them into several fixed Burau block meshes or wavelength channels;
3. detect intensities in parallel; and
4. apply a small trained electronic readout.

The optical bank can remain static while only the electronic readout is
trained, making the transform itself passive. This can be interesting only
after comparison with a generic Mach--Zehnder or diffractive network. Burau
structure must earn its place through easier calibration, useful spectral
multiplexing, robustness, or task accuracy; non-Abelianity alone is not a
speedup.

A hardware benchmark must include source and modulation energy, insertion
loss, detector noise, ADC and readout energy, precision, drift, throughput,
and end-to-end latency. A deeper model requires intermediate optical
nonlinearity or detection and re-encoding.

## Primary references

- M. Reck et al., “Experimental realization of any discrete unitary
  operator,” *Physical Review Letters* 73, 58–61 (1994),
  [doi:10.1103/PhysRevLett.73.58](https://doi.org/10.1103/PhysRevLett.73.58).
- Y. Shen et al., “Deep learning with coherent nanophotonic circuits,”
  *Nature Photonics* 11, 441–446 (2017),
  [doi:10.1038/nphoton.2017.93](https://doi.org/10.1038/nphoton.2017.93).
- X. Lin et al., “All-optical machine learning using diffractive deep neural
  networks,” *Science* 361, 1004–1008 (2018),
  [doi:10.1126/science.aat8084](https://doi.org/10.1126/science.aat8084).
- R. Hamerly et al., “Large-Scale Optical Neural Networks Based on
  Photoelectric Multiplication,” *Physical Review X* 9, 021032 (2019),
  [doi:10.1103/PhysRevX.9.021032](https://doi.org/10.1103/PhysRevX.9.021032).
- Y. Zhu et al., “LightIN: a versatile silicon-integrated photonic field
  programmable gate array with an intelligent configuration framework for
  next-generation AI clusters,” *Light: Science & Applications* 15, 165
  (2026),
  [doi:10.1038/s41377-026-02209-5](https://doi.org/10.1038/s41377-026-02209-5).
