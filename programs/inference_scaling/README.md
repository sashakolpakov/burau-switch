# Robust quadratic inference scaling

This study asks whether the inference track's exact four-mode expressivity
result remains numerically useful as the number of modes grows. It compares
generic-unitary spectral scorers, complete mutually unbiased basis (MUB)
banks, and Haar-random banks. **It does not compile any of these bases into
the restricted Burau-derived block library, and it makes no Burau scaling or
reachability claim.**

Run it from the repository root with:

    python -m programs.inference_scaling.reproduce

The command writes `figures/inference_scaling.png` and
`results/inference_scaling.json`.

## Why conditioning, not only rank, matters

For a task-independent bank of $d$-output orthonormal measurements, at least
$d+1$ settings are needed to span all $d^2$ Hermitian quadratic forms. Random
banks almost surely reach that rank, but reconstruction noise is controlled by
the projector frame's smallest singular value, not rank alone.

The checked experiment forms a real Hilbert--Schmidt measurement matrix whose
rows are the rank-one output projectors. Its primary condition number is
$\kappa_2=\sigma_{\max}/\sigma_{\min}$, including the identity direction, with
a numerical rank threshold of $10^{-10}$. A complete MUB has condition
$\sqrt{d+1}$ under this convention and condition one after removing the
exactly known identity direction: it is a tight frame on the traceless
Hermitian subspace.

The implemented exact constructions are:

- odd primes $d\in\{3,5,7,11,13\}$: the computational basis plus the $d$
  chirp bases with rows
  $d^{-1/2}\exp[2\pi i(a n^2+b n)/d]$ for $a,b,n\in\mathbb Z_d$; and
- $d=4$: five two-qubit stabilizer bases from an explicit partition of the
  Pauli operators into commuting classes.

Complete MUBs exist in every prime-power dimension, but not all prime-power
constructions are implemented here, and existence of a complete set is not
known in every composite dimension.

## Checked results

The random minimum-bank condition numbers become both large and heavy-tailed,
whereas the MUB values follow the exact $\sqrt{d+1}$ curve. For example, at
$d=13$, 12 random $d+1$-setting Haar banks have median condition 2466,
10th--90th percentile 1181--10180, and maximum 16069. The complete MUB value
is 3.742. A $2d$-setting random bank is much better conditioned (median 13.18)
but uses nearly twice the rank-bound number of settings.

The detector simulation uses normalized complex-Gaussian inputs, random
traceless unit-Frobenius-norm Hermitian observables, $1{,}200$ input samples
per trial, a fixed total of $10^6$ **detected** photoelectrons per inference,
and 5-electron RMS read noise per output. The total budget is divided equally
among settings, Poisson and read noise are applied, and every setting is
renormalized by its measured total intensity, matching the separate digital
twin's idealized detector model.

Median normalized score RMSE is:

| $d$ | trials | spectral, 1 setting | complete MUB, $d+1$ | Haar, $d+1$ | Haar, $2d$ |
|---:|---:|---:|---:|---:|---:|
| 3 | 20 | 0.173% | 0.348% | 1.34% | 0.473% |
| 4 | 20 | 0.201% | 0.451% | 2.38% | 0.652% |
| 5 | 20 | 0.223% | 0.550% | 3.43% | 0.741% |
| 7 | 16 | 0.268% | 0.756% | 4.30% | 1.14% |
| 11 | 10 | 0.327% | 1.14% | 17.3% | 1.61% |
| 13 | 8 | 0.362% | 1.33% | 25.1% | 2.05% |

For this state and observable ensemble, the spectral one-score design follows
approximately $\sqrt{d/N}$ normalized error, so constant error needs detected
photon number $N=O(d)$. The complete MUB bank follows approximately
$d/\sqrt N$, so a fixed task-independent bank needs $N=O(d^2)$. The random
minimum bank becomes unusable substantially earlier because inversion of its
ill-conditioned frame amplifies noise.

These are generic-unitary, ideal-transfer numerical results. The photon count
is after fanout, coupling, and mesh loss. The study does not include source
energy, insertion loss, gain mismatch, quantization, laser noise, thermal
crosstalk, or drift.

## Resource interpretation

One trained Hermitian score needs one spectral unitary, $d$ photodiodes, and a
generic Clements implementation with $d(d-1)/2$ two-mode cells and $O(d)$
path depth. A fully parallel bank using $d+1$ universal mesh instances would
use $d(d+1)$ detectors and $(d+1)d(d-1)/2=O(d^3)$ two-mode cells. Specializing
the computational-basis branch as a direct detector path removes one mesh,
leaving $d^2(d-1)/2$ cells. Either layout also has ideal branch attenuation
$10\log_{10}(d+1)$ dB. Time multiplexing can instead reuse one $O(d^2)$-cell
mesh and $d$ detectors, but requires $d+1$ sequential configurations.

That generic count is not an optimal implementation claim for MUBs. The
odd-prime chirp matrices factor into a Fourier transform and diagonal chirp
phases, so fixed or shared structured optics may be substantially cheaper than
$d+1$ unrelated universal meshes. Conversely, whether the same bases admit
short words from one fixed Burau-derived local block library remains an open
compiler experiment.

## Primary references

- W. K. Wootters and B. D. Fields, “Optimal state-determination by mutually
  unbiased measurements,” *Annals of Physics* 191, 363--381 (1989),
  [doi:10.1016/0003-4916(89)90322-9](https://doi.org/10.1016/0003-4916(89)90322-9).
- A. J. Scott, “Tight informationally complete quantum measurements,”
  *Journal of Physics A* 39, 13507 (2006),
  [doi:10.1088/0305-4470/39/43/009](https://doi.org/10.1088/0305-4470/39/43/009).
- W. R. Clements et al., “Optimal design for universal multiport
  interferometers,” *Optica* 3, 1460--1465 (2016),
  [doi:10.1364/OPTICA.3.001460](https://doi.org/10.1364/OPTICA.3.001460).
- R. Hamerly et al., “Large-Scale Optical Neural Networks Based on
  Photoelectric Multiplication,” *Physical Review X* 9, 021032 (2019),
  [doi:10.1103/PhysRevX.9.021032](https://doi.org/10.1103/PhysRevX.9.021032).
