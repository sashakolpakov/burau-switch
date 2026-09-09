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

## Exact response symmetries

Part of the collapse is forced analytically by the particular same-sided
dressing used in this model. For $\omega$ in either open Squier-definite
component, let

\[
U_i=H_\omega^{1/2}\beta_iH_\omega^{-1/2},\qquad
M_w=U_{a_1}\cdots U_{a_L},\qquad
D_S(M)=(M\otimes I)S(M\otimes I),
\]

where $S=\operatorname{diag}(A,B)$ is block diagonal. If $R(w)$ reverses the
signed letters and $\tau$ exchanges generator indices $1\leftrightarrow2$
without changing their exponents, then every response depending only on the
projective spectrum of $D_S(M_w)$ obeys

\[
r_w=r_{R(w)}=r_{\tau(w^{-1})}.
\]

Here is the proof. The positive square root of the sign-normalized Squier form
has the shape $C=\left(\begin{smallmatrix}p&q\\q&p\end{smallmatrix}\right)$.
The relations $p^2+q^2=\epsilon(s+s^{-1})$, $2pq=-\epsilon$, and
$1+s^2=(s+s^{-1})s$ show exactly that each signed, unitarized generator is
symmetric. Thus $M_{R(w)}=M_w^T$. Also $U_2=P U_1P$ for the exchange matrix
$P$, which gives $M_{\tau(w^{-1})}=PM_w^{-1}P$.

For $M=\left(\begin{smallmatrix}a&b\\c&d\end{smallmatrix}\right)$, direct block
multiplication gives

\[
D_S(M)=
\begin{pmatrix}
a^2A+bcB & b(aA+dB)\\
c(aA+dB) & bcA+d^2B
\end{pmatrix}.
\]

Exchanging $b$ and $c$ produces a similar matrix, so $D_S(M)$ and
$D_S(M^T)$ have the same spectrum. If an off-diagonal entry vanishes,
unitarity forces both to vanish and the conclusion is immediate. Finally,
with $Z=\operatorname{diag}(1,-1)$,

\[
PM^{-1}P=\det(M)^{-1}ZM^TZ.
\]

The $Z\otimes I$ factor commutes with block-diagonal $S$; the determinant
factor contributes only a global phase. This proves the two response
identities. The argument applies to the same-sided, projective-spectrum
response above. It is not a claim about arbitrary optical readouts, the more
usual conjugate dressing $(M\otimes I)S(M^\dagger\otimes I)$, or Burau-based
devices in general.

The two word involutions generate a Klein-four action. Burnside's lemma gives
the following upper bound on distinguishable response orbits among all
length-$L$ signed words:

\[
N_L\leq\frac{4^L+4^{\lceil L/2\rceil}
+\mathbf 1_{2\mid L}4^{L/2}}{4}.
\]

For $L=1,\ldots,6$, the bounds are respectively
$2,6,20,72,272,1056$. The reproduction script verifies the symbolic matrix
identities, enumerates these word orbits, and asserts both response
symmetries on the full default grid and on the seeded off-grid validation
phases. The observed 36 sampled classes at $L=5$ lie well below the bound
$272$, so these two exact involutions explain only part of the empirical
collapse.

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
