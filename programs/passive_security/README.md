# Passive T-chip security route: a high-dimensional optical key

This study repairs, rather than discards, the security idea behind the T-chip.
The exact four-mode Burau circuit is a public passive branch mixer.  Two fixed,
independently fabricated high-dimensional branch maps `X` and `Y` carry the
device-specific structure; the mixer exposes their sum and difference ports.
Privately phase-randomized weak coherent challenges and photon-counting
verification then create a measurable gap between the true key and a bounded
challenge-estimation emulator.

The checked calculation is a receiver-level design study.  It is **not a
hardware demonstration, not a proof that a fabricated key is unclonable, and
not a complete cryptographic security proof**.

Run it from the repository root with:

    MPLCONFIGDIR=/tmp/burau-mpl python -m programs.passive_security.reproduce

It deterministically writes:

- `results/passive_security.json`, containing every parameter, factor,
  threshold, FAR, FRR, scope condition, and numerical assertion; and
- `figures/passive_security.png`, comparing the high-dimensional design with
  the bare four-mode circuit.

## What is right in the original T-chip idea

There is a useful exact passive component.  At
`omega = pi/4`, the application-order word

    sigma_2^-1 sigma_1 sigma_2^-1

or primitive codes `[3, 0, 3]` is a symmetric unitary whose four entry
magnitudes are all `1/sqrt(2)`.  The reproduced residuals are:

- unitarity: `1.05e-15`;
- maximum entry-magnitude error: `3.33e-16`; and
- equivalence, after input/output port phases, to a standard 50:50 coupler:
  `6.73e-16`.

With the same mixer on both sides of a two-branch block and a fixed relative
branch phase

    theta_0 = 4.539962982671842 radians,

the two output ports carry, up to fixed port phases, `(X + Y)x/2` and
`(X - Y)x/2`.  The checked coefficient residual is `4.89e-16`.  This is a real
engineering simplification: a fixed passive Burau-derived word can perform the
sum/difference readout without a tuned universal mesh.

For unitary `X`, `Y` and normalized `x`, the difference-port probability is

    P_minus = [1 - Re x^dagger X^dagger Y x] / 2,

and `P_plus + P_minus = 1`.  In the controlled-order specialization
`X = BA`, `Y = AB`, this becomes

    P_minus = ||[B,A]x||^2 / 4
            = [1 - Re x^dagger A^dagger B^dagger A B x] / 2,

The script verifies these identities on a fixed noncommuting test pair to below
`5e-16`.  The deployable key does not require identical copies of `A` and `B`:
it enrolls the complete measured map of independent branches `X` and `Y`.
Calling the difference a commutator is reserved for a separately verified
controlled-order reference, because duplicated components and reciprocal
counter-propagation generally change the maps.

A conventional 50:50 directional coupler or balanced MZI performs the same
recombination up to port phases.  It is therefore the mandatory baseline for
loss, footprint, stability, and price.  The braid word has to beat that
baseline on an engineering metric; its algebraic provenance alone is not a
security advantage.

For ideal unitary branches, the stacked map
`Vx = [(X+Y)x; (X-Y)x]/2` is an isometry.  The code verifies its Gram matrix
and overlap preservation.  For a measured map `H`, the study also checks the
robustness inequality obtained from
`epsilon = ||H^dagger H/tau - I||_2`: normalized response overlap is at most
`(|<x,xhat>|+epsilon)^2/(1-epsilon)^2`.  A fabricated device must measure this
defect on its advertised challenge subspace; calibrating an arbitrary lossy
transfer matrix is not enough.

## What was missing from the security argument

The public deterministic T-chip is a known `4 x 4` linear optical map.  An
attacker does not have to invert a braid word: the attacker can implement or
estimate the same transfer matrix.  If that direct emulator has the same loss
and noise distribution, the verifier sees exactly the honest count
distribution.  Its total-variation distance is zero and the equal-prior Bayes
error is `1/2`, no matter how many times the same type of test is repeated.

Nor does a fixed fan-out of four amplitudes create a high-dimensional quantum
challenge.  If only four complex amplitudes are independently controlled, the
challenge subspace has dimension at most four even if those amplitudes
illuminate thousands of pixels.  The security parameter must count
independent challenge modes, not waveguides, camera pixels, or nominal device
features.

The positive architecture is instead:

    trusted phase-randomized K-mode weak-pulse shaper
      -> exact public Burau T-mixer
      -> sealed passive high-dimensional branches X and Y
      -> sum/difference ports
      -> computed phase-conjugate analyzer
      -> per-round power tap and photon counter

Enrollment uses bright phase-sensitive basis probes to recover the full complex
transfer operator `H`; it does not store a finite reusable challenge bank.
Each authentication round samples a genuinely new Haar-random state across the
full `K`-dimensional challenge space, computes `Hx`, and programs its matched
analyzer.  A private uniform global phase makes the weak coherent state
photon-number diagonal.  The independently controlled `K` modes supply the
state-estimation gap; fixed four-port fan-out cannot manufacture them.

## Attack factors used

Let `nbar` be the mean photons in a challenge and `K` its dimension.  For
exactly `N` photons, arbitrary-POVM state estimation gives

    q_N = (N + 1) / (N + K).

For a weak coherent pulse, the arbitrary-POVM extension requires the reader to
privately randomize the pulse's global phase.  The resulting state is a
photon-number-diagonal Poisson mixture; block-diagonalizing any POVM preserves
all its outcome statistics.  Averaging `q_N` gives the exact Poisson sum

    q_coherent = E[(N + 1) / (N + K)],   N ~ Poisson(nbar).

Concavity gives the convenient safe bound

    q_coherent <= q_safe = (nbar + 1) / (nbar + K).

For the main theorem, every refined attacker outcome `y` (including any
measured photon-number sector) must obey an outcome-conditional cap `Lambda`
on mean returned photons.  If `gamma_y` is the normalized one-particle state
of an arbitrary prepared response, the same state-estimation calculation
bounds its mean matched fraction by `q_coherent <= q_safe`.  The illustrative
model sets `Lambda = eta_return * nbar`; a fabricated reader must certify an
accepted-response energy gate and its allowance over every spatial/modal,
wavelength, polarization, and timing degree of freedom capable of causing an
accepted click.  Verified filters define that accepted optical space.  An
average power monitor alone does not establish this conditional assumption.

The code additionally records a sector-proportional resend diagnostic,

    q_click_exact = E[(M + 2) / (M + K + 1)],  M ~ Poisson(nbar),
    q_click_safe  = (nbar + 2) / (nbar + K + 1),

but it is not the main flux-gated bound.  If `f` is the empirical authentic
focused fraction, the main attacker-to-honest signal ratio is `q_safe/f`, not
`q_safe`.  Both results assume that the adversary cannot instead build a fast,
sufficiently low-loss physical realization of the enrolled `K`-input-mode
transfer map `H` (or an isometric dilation).  That physical-emulator premise is
plausible only for a genuinely large, disorder-bearing system; it is not
credible for the bare four-mode T-chip.

For comparison, the code records the simplified Goorden factor

    q_Goorden = 1 / (1 + K/nbar)

and the finite-focus quadrature expression

    q_quad ~= [1 + 1/(nbar f)] / (1 + K/nbar),

where `f` is the correct-response focused fraction.  The latter is the
large-`K`, `K > nbar` result for the best equal-split quadrature
challenge-estimation attack in that optical model; it is not a bound on every
possible adversary.  These factors are reported separately rather than
silently conflated.

The primary sources are:

- S. A. Goorden, M. Horstmann, A. P. Mosk, B. Škorić, and P. W. H. Pinkse,
  “Quantum-secure authentication of a physical unclonable key,” *Optica* 1,
  421--424 (2014),
  [doi:10.1364/OPTICA.1.000421](https://doi.org/10.1364/OPTICA.1.000421).
  The experiment used a multiple-scattering key, weak coherent pulses, about
  1100 controlled modes, a phase-conjugate analyzer, and photon counting.
- G. Sarantoglou et al., “Quantum-Secure Physical Unclonable Function enabled
  by Silicon Photonics Integrated Circuits,”
  [arXiv:2605.14959](https://arxiv.org/abs/2605.14959) (2026).  This
  contemporaneous preprint experimentally characterizes a 6x6 thermo-optic
  SiN mesh and numerically studies an orthogonal single-photon time-bin
  protocol.  The present route is distinct: an unpowered token,
  nonorthogonal Haar-mode weak-coherent challenges, and an explicit
  challenge-estimation bound.
- B. Škorić, “Security analysis of Quantum-Readout PUFs in the case of
  challenge-estimation attacks,” *Quantum Information and Computation* 16,
  50--60 (2016),
  [ePrint 2013/479](https://eprint.iacr.org/2013/479).  This supplies the
  arbitrary-POVM state-estimation bounds and states the no-efficient-coherent-
  emulator assumption explicitly.
- B. Škorić, A. P. Mosk, and P. W. H. Pinkse, “Security of Quantum-Readout
  PUFs against quadrature-based challenge-estimation attacks,”
  [ePrint 2013/084](https://eprint.iacr.org/2013/084).  Equation (20) supplies
  the finite-focus quadrature comparison used here.
- W. R. Clements et al., “Optimal design for universal multiport
  interferometers,” *Optica* 3, 1460--1465 (2016),
  [doi:10.1364/OPTICA.3.001460](https://doi.org/10.1364/OPTICA.3.001460).
  A small public unitary has a standard passive-mesh emulator, which is why
  the four-mode transfer function cannot be the physical secret.

## Photon-counting model

For one fresh challenge, the true-key focused-detector signal mean is

    mu_signal = nbar * f * eta_return * eta_detector,

and the total honest mean is `mu_H = mu_signal + mu_bg`.  The conservative
emulator mean uses the absolute projection bound before authentic focus:

    mu_A = mu_bg + q_safe * Lambda * eta_detector,
    Lambda = eta_return * nbar                 [illustrative point].

This deliberately does not grant the attacker the authentic `f` penalty.
Coherent light, linear loss, and independent background motivate Poisson click
counts.  The conditional energy ceiling is an explicit attack-model premise:
an accepted-response gate must be experimentally calibrated to reject excess
energy for every response class.  A simple average-power statement is not a
proof of this condition.

The verifier adds the counts from `R` fresh independent challenges and
accepts when the total is at least an integer threshold `T`.  A sum of
independent Poisson variables is Poisson, so the program evaluates without
Monte Carlo error

    FRR = Pr[Poisson(R mu_H) < T],
    FAR = Pr[Poisson(R mu_A) >= T].

For each point it exhaustively selects the integer threshold minimizing
`max(FAR, FRR)`.  These are exact distribution tails **inside this count
model**.  A bound on mean state-estimation fidelity alone does not prove that
an adaptive or correlated adversary has Poisson tails; that additional model
assumption is recorded in the JSON.

The code also reports a slower click/no-click decision that does not assume an
attacker Poisson law.  If the attacker count mean is conditionally bounded in
every fresh round, Markov's inequality bounds that round's click probability;
the adaptive total is then upper-tail dominated by a binomial distribution.
This is the appropriate conservative table when correlated count shapes are
in scope, but it still relies on the conditional energy and state-estimation
assumptions.

## Illustrative T-chip-compatible design point

The concrete point is deliberately close to already demonstrated optical
ingredients but is not claimed as fabricated:

| parameter | value |
|---|---:|
| independently controlled modes `K` | 1024 |
| mean challenge photons `nbar` | 50 |
| `K/nbar` | 20.48 |
| true focused fraction `f` | 0.60 |
| key return efficiency | 0.25 |
| detector efficiency | 0.70 |
| background clicks per round | 0.05 |
| detectable return clicks before focus | 8.75 |
| honest focused signal clicks per round | 5.25 |
| honest total mean | 5.30 |
| attacker total mean, conditional-energy bound | 0.4655 |

At this point the main Jensen bound is `0.047486`, while its exact
phase-randomized coherent-state sum is `0.047445`.  The main bound is
`0.079143` relative to the authentic focused signal because `f=0.60`.  The
sector-proportional resend diagnostic is `0.048331` exactly and `0.048372`
after Jensen; it is recorded separately rather than used as though a power tap
could resolve the intercepted Poisson sector.

The optimized exact Poisson decisions under the main factor are:

| fresh rounds | accept at total clicks | FAR | FRR | worst error |
|---:|---:|---:|---:|---:|
| 1 | 2 | `7.993e-2` | `3.145e-2` | `7.993e-2` |
| 2 | 4 | `1.505e-2` | `6.635e-3` | `1.505e-2` |
| 5 | 10 | `1.582e-4` | `8.127e-5` | `1.582e-4` |
| 10 | 20 | `1.143e-7` | `6.995e-8` | `1.143e-7` |
| 20 | 40 | `8.206e-14` | `6.883e-14` | `8.206e-14` |

The adaptive-safe click/no-click bound is intentionally looser:

| fresh rounds | accept if clicked rounds | FAR upper bound | FRR | worst bound |
|---:|---:|---:|---:|---:|
| 5 | 5 | `2.186e-2` | `2.471e-2` | `2.471e-2` |
| 10 | 9 | `5.964e-3` | `1.092e-3` | `5.964e-3` |
| 20 | 18 | `6.265e-5` | `1.330e-4` | `1.330e-4` |
| 50 | 44 | `1.023e-9` | `6.391e-9` | `6.391e-9` |

These numbers say the route is worth a bench experiment.  They do not say
that a real device will retain 60% focus, independent Poisson statistics, or
stable calibration over temperature, vibration, aging, and adversarial
illumination.

## Why four modes alone fail the intended route

With the same loss budget, one expected honest **signal** click per round
requires `nbar = 9.52`, already greater than `K=4`.  At the more usable
`nbar=50` point, `K/nbar=0.08`; it is outside the `n < K` theorem setup and
the state-estimation upper bound, shown only diagnostically, is `0.9444`,
already above the authentic focus `f=0.60`, so it gives no useful separation
for the one-sided count test.  The sector-proportional diagnostic is `0.9455`.
More decisively, a direct emulator of the public four-mode map can attenuate
to reproduce the honest distribution, so the equal-prior Bayes error remains
`1/2` regardless of repetition.

One can force `n < K` by taking `nbar=3`, but the expected honest focused
signal then falls to `0.315` clicks per round.  More importantly, neither
photon choice repairs the central scope failure: the public four-dimensional
transfer map can be directly emulated.  For that relevant attacker the signal
factor is one and the count distributions are identical.  The bare T-chip can
be an elegant passive braid/order interferometer; it cannot also be the
unclonable key merely because its generators do not commute.

## What this route could achieve

If a fabricated disorder-bearing token passes the missing attack tests, the
natural deliverable is fast possession authentication or anti-counterfeit
readout of a passive optical object, potentially with a cheap token and a
more capable trusted reader.  The count gap is compatible with weak coherent
laser pulses; it does not require an anyon, entangled source, or deterministic
single-photon source.

The calculation does not establish encryption, signatures, general key
exchange, or secret-key extraction.  Those need separate protocols and threat
models.  Before any positive security claim, hardware must measure
intra-device repeatability, inter-device separation, drift, calibration
leakage, finite challenge reuse, FAR/FRR over the environmental envelope, and
resistance to transfer-matrix learning, physical cloning/substitution,
replay/relay, Trojan illumination, detector attacks, and verifier compromise.

The decisive comparison is attack-first: the high-dimensional Burau-assisted
device must beat both (i) the same random key with a conventional 50:50
coupler and (ii) a direct optical emulator at equal loss, bandwidth, and
reader trust.  Until then, “T-chip-compatible” means a concrete experiment
architecture, not a demonstrated security advantage.
