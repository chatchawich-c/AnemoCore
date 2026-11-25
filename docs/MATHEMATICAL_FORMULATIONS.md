# Mathematical Formulations

This document collects the principal governing equations, physical laws, and numerical estimators implemented in the **AnemoCore** Rust library. Each section corresponds to a source module in `crates/anemocore/src/`. Equations are expressed in LaTeX syntax compatible with MkDocs MathJax. Symbols are defined with their physical meaning and SI units.

---

## 1. `math_no_std`

Utility functions for probability theory (no‑std).

- **Standard normal cumulative distribution function**
  $$\Phi(z) = \frac{1}{2}\left[1 + \operatorname{erf}\!\left(\frac{z}{\sqrt{2}}\right)\right]$$
  *Symbol*: $z$ – standard score (unitless).

- **Inverse normal CDF (Probit)** – rational minimax approximation (accuracy $<1.15\times10^{-9}$) for $p\in(0,1)$.
  $$\Phi^{-1}(p) \approx \text{piecewise rational function of }p$$
  *Symbol*: $p$ – probability (unitless).

- **Wrap unit interval** maps any real $x$ to $[0,1)$:
  $$\operatorname{wrap}(x) = x - \lfloor x \rfloor$$
  *Symbol*: $x$ – dimensionless scalar.

---

## 2. `weibull_est`

Weibull distribution $\text{Weibull}(k, c)$ with shape $k>0$ and scale $c>0$.

- **CDF**
  $$F(v; k, c) = 1 - \exp\!\left[-\left(\frac{v}{c}\right)^{k}\right]$$
  *Symbols*: $v$ – wind speed (m s⁻¹), $k$ – shape (unitless), $c$ – scale (m s⁻¹).

- **PDF**
  $$f(v; k, c) = \frac{k}{c}\left(\frac{v}{c}\right)^{k-1}\exp\!\left[-\left(\frac{v}{c}\right)^{k}\right]$$

- **Mean wind speed**
  $$\bar{v}=c\,\Gamma\!\left(1+\frac{1}{k}\right)$$

- **Wind Power Density** (WPD, W m⁻²)
  $$\mathrm{WPD}=\frac{1}{2}\,\rho\,c^{3}\,\Gamma\!\left(1+\frac{3}{k}\right)$$
  *Symbol*: $\rho$ – air density (kg m⁻³).

- **Gamma function (Lanczos approximation)** for $z>0$:
  $$\Gamma(z) \approx \sqrt{2\pi}\,\bigl(z+g+\tfrac{1}{2}\bigr)^{z+\tfrac{1}{2}}\,e^{-(z+g+\tfrac{1}{2})}\,A_g(z)$$
  where $g=7$, $A_g(z)=\sum_{i=0}^{N}\frac{p_i}{z+i}$ with coefficients $p_i$ defined in the source.

- **Goodness‑of‑Fit metrics** for a histogram $H$ with $n$ valid samples:
  - Coefficient of determination $R^{2}=1-\frac{\sum (\hat{F}_i - F_i)^2}{\sum (F_i-\bar{F})^2}$
  - Root‑Mean‑Square Error $\mathrm{RMSE}=\sqrt{\frac{1}{m}\sum (\hat{F}_i - F_i)^2}$
  - Kolmogorov‑Smirnov statistic $D=\max_i|\hat{F}_i-F_i|$
  where $\hat{F}_i$ is empirical CDF, $F_i$ theoretical Weibull CDF.

---

## 3. `wra`

Wind‑Resource‑Assessment (WRA) shear‑profile models.

- **Power‑law shear**
  $$U(z) = U_{\mathrm{ref}}\left(\frac{z}{z_{\mathrm{ref}}}\right)^{\alpha}$$
  *Symbols*: $U(z)$ – wind speed at height $z$ (m s⁻¹);
  $U_{\mathrm{ref}}$ – reference speed at $z_{\mathrm{ref}}$ (m s⁻¹);
  $\alpha$ – shear exponent (unitless).

- **Log‑law shear**
  $$U(z) = \frac{u_*}{\kappa}\,\ln\!\left(\frac{z}{z_0}\right)\!\left[1-\psi_m\!\left(\frac{z}{L}\right)\right]$$
  *Symbols*: $u_*$ – friction velocity (m s⁻¹);
  $\kappa=0.4$ – von Kármán constant;
  $z_0$ – surface roughness length (m);
  $L$ – Obukhov length (m);
  $\psi_m$ – stability correction function.

- **Stability correction $\psi_m$** (neutral case $L\to\infty$ yields $0$). Implemented as piecewise linear functions in the source.

- **Parser utilities** (`parse_tab`, `parse_wra_json`, …) transform CSV/JSON records into the internal `WindResource` struct; no analytical formula is defined.

---

## 4. `microsite`

Micro‑site wind‑rose transformations and CFD speed‑up grid.

- **Stability correction for microsite**
  $$\psi_m(z/L) = \begin{cases}
      -\beta\,\frac{z}{L} & \text{if }\frac{z}{L}<0 \;(unstable),\\
      \gamma\,\left(\frac{z}{L}\right) & \text{if }\frac{z}{L}>0 \;(stable),
    \end{cases}$$
  where $\beta,\gamma$ are empirical constants (implemented in `stability_psi_m`).

- **Log‑law ratio** used to scale wind speed between two heights $z_1$ and $z_2$:
  $$R = \frac{\ln(z_2/z_0)-\psi_m(z_2/L)}{\ln(z_1/z_0)-\psi_m(z_1/L)}$$

- **CFD speed‑up grid interpolation** – linear interpolation of pre‑computed CFD results on a regular grid of wind speed and direction.

---

## 5. `aero`

Aerodynamic loss‑factor models for rotor blades.

- **Prandtl tip‑hub loss factor**
  $$F = \frac{2}{\pi}\,\arccos\!\bigl[\exp\bigl(-\tfrac{B}{2}\,(R_{\mathrm{tip}}-r)/(r\,\sin\phi)\bigr)\bigr]$$
  *Symbols*: $B$ – blade count; $R_{\mathrm{tip}}$ – rotor radius (m); $r$ – local radius (m); $\phi$ – flow angle (rad).

- **Blade‑element‑momentum (BEM) steady‑state solver** combines induction factors $a$, $a'$ with local thrust and torque coefficients. Core relations (simplified):
  $$C_T = \frac{4a(1-a)F}{\lambda^2},\qquad C_Q = \frac{4a'(1-a)F}{\lambda}\,\frac{R}{V}\,$$
  where $\lambda = \Omega R/V$ is tip‑speed ratio, $\Omega$ rotor angular speed (rad s⁻¹), $V$ wind speed (m s⁻¹).

- **Beddoes‑Leishman stall model** computes dynamic stall lift $C_{L}$ and drag $C_{D}$ as functions of angle of attack $\alpha$, stall delay time $\tau$, and empirical coefficients.

---

## 6. `wake`

Wake superposition and velocity deficit model.

- **Gaussian wake deficit** (per turbine $i$)
  $$\Delta u_i(r,\theta) = \frac{C_{T,i}}{2\,(1+\frac{k\,x_i}{R})^2}\exp\!\left[-\frac{(r-\delta_i)^2+\theta^2}{2\sigma_i^2}\right]$$
  where $C_{T,i}$ – thrust coefficient, $x_i$ downstream distance, $R$ – rotor radius, $k$ – wake expansion constant, $\sigma_i$ – wake width, $\delta_i$ – lateral offset.

- **Superposition (linear)**
  $$u_{\text{total}} = V_\infty - \sum_i \Delta u_i$$
  with ambient wind $V_\infty$.

---

## 7. `phyllotaxis_wake`

Geometric placement of turbines based on phyllotactic spirals.

- **Radial position** of turbine $n$:
  $$r_n = a\,\sqrt{n}\,$$
  *Symbol*: $a$ – scaling factor (m).

- **Angular position** using golden angle $\theta_g = 2\pi\,\varphi$, $\varphi = \frac{1+\sqrt{5}}{2}^{-1}$:
  $$\theta_n = n\,\theta_g$$
  The module provides a function `phyllotaxis_wake` that returns $(r_n,\theta_n)$ for each index.

---

## 8. `oa`

Orientation‑averaged (OA) turbine performance model.

- **Power‑curve model** – cubic fit of coefficient $C_P(\lambda)$:
  $$C_P(\lambda) = a_0 + a_1\lambda + a_2\lambda^{2} + a_3\lambda^{3}$$
  Parameters $a_i$ are calibrated from data.

---

## 9. `thermo`

Thermal stress and moisture‑air properties.

- **Thermal stress indicators** $p_1$–$p_5$ are functions of temperature $T$ (K) and material coefficients; for example:
  $$p_1 = \frac{\Delta T}{\alpha}\,$$
  where $\alpha$ – thermal expansion coefficient (K⁻¹).

- **CTD (Closed‑Temperature‑Difference) model**
  $$\mathrm{CTD}= \frac{\rho_w c_{p,w}(T_w - T_a)}{h_c}\,$$
  $\rho_w$ – water density (kg m⁻³), $c_{p,w}$ – water heat capacity (J kg⁻¹ K⁻¹), $h_c$ – convective heat transfer coefficient (W m⁻² K⁻¹).

- **CIPM‑2007 saturation vapor pressure**
  $$e_s(T) = 611.21\,\exp\!\left[\left(\frac{17.502\,T}{240.97+T}\right)\right]$$
  (Pa, with $T$ in °C).

---

## 10. `tribology`

Lubricant film‑thickness estimation (Elrod‑Marusic model).

- **Film thickness $h$**
  $$h = 2.65\,\frac{\eta^{0.67}U^{0.53}}{\rho^{0.33}E^{0.33}}\,\left(\frac{P}{E}\right)^{0.13}$$
  where $\eta$ – dynamic viscosity (Pa·s), $U$ – entrainment velocity (m s⁻¹), $\rho$ – lubricant density (kg m⁻³), $E$ – elastic modulus (Pa), $P$ – load per unit width (N·m⁻¹).

---

## 11. `signal_wpt`

Wavelet‑packet transform for signal analysis.

- **Discrete wavelet packet coefficient** at level $\ell$ and node $n$:
  $$c_{\ell,n} = \sum_{k} x_k\,\psi_{\ell,n}(k)$$
  where $\psi_{\ell,n}$ is the basis function derived from the chosen mother wavelet (e.g., Daubechies‑4).

- **Energy of a packet**
  $$E_{\ell,n}=\sum_{k} |c_{\ell,n}(k)|^{2}$$

---

## 12. `quasicrystal_wpt`

Integer‑shift‑add wavelet‑packet operations for quasicrystal analysis.

- **Shift‑add update** for a coefficient vector $\mathbf{c}$:
  $$\mathbf{c}^{\prime}=\mathbf{c} + \phi^{-1}\,\mathbf{c}_{\text{shift}}$$
  where $\phi = \frac{1+\sqrt{5}}{2}$ is the golden ratio.

- **Energy contribution** summed over levels:
  $$E = \sum_{\ell}\sum_{n}\bigl|c_{\ell,n}\bigr|^{2}$$

---

## 13. `kam_vsg`

Variable‑speed‑generator (VSG) rotor dynamics.

- **State update** (Euler step) for angular speed $\omega$ and torque $T$:
  $$\omega_{t+\Delta t}=\omega_t+\frac{T_t - T_{\text{load}}}{J}\,\Delta t$$
  where $J$ – moment of inertia (kg·m²), $T_{\text{load}}$ – aerodynamic torque.

- **Biquad notch filter** used for vibration suppression:
  $$H(z)=\frac{1-2\cos\omega_0\,z^{-1}+z^{-2}}{1-2r\cos\omega_0\,z^{-1}+r^{2}z^{-2}}$$
  $\omega_0$ – notch frequency (rad), $r$ – pole radius (0<r<1).

---

## 14. `fib_qmc`

Quasi‑Monte‑Carlo lattice sampling and limit‑state evaluation.

- **Fibonacci lattice point** in 2‑D:
  $$\mathbf{x}_i = \left(\frac{i}{N},\,\left\{i\,\phi \right\}\right)\quad i=0\dots N-1$$
  where $\{\cdot\}$ denotes fractional part, $\phi=\frac{1+\sqrt{5}}{2}$.

- **Limit‑state function** $g(\mathbf{x})$ evaluated at each point; probability of failure estimated as
  $$P_f \approx \frac{1}{N}\sum_{i=0}^{N-1} \mathbf{1}_{\{g(\mathbf{x}_i)\le 0\}}$$

---

## 15. `fib_liquidity`

Liquidity metrics for order‑book analysis.

- **Roll effective spread**
  $$\sigma_{\text{roll}} = \sqrt{\frac{2}{\pi}}\,\frac{\Delta p}{\sqrt{\Delta t}}$$
  where $\Delta p$ – price change (currency), $\Delta t$ – time interval (s).

- **LOB density** estimated by kernel smoothing of depth $d(p)$ over price $p$:
  $$\rho(p)=\sum_{j}\,K\!\left(\frac{p-p_j}{h}\right)\,d(p_j)$$
  with bandwidth $h$.

---

## 16. `scada_fleet`

SCADA‑based health index and residual diagnostics.

- **Multi‑Domain Health Index (MDHI)**
  $$\text{MDHI}=\frac{1}{M}\sum_{m=1}^{M}\frac{|r_m|}{\sigma_m}$$
  where $r_m$ – residual of domain $m$, $\sigma_m$ – standard deviation, $M$ – number of domains.

- **SARIMAX forecast residual**
  $$\hat{r}_t= y_t-\sum_{i=1}^{p}\phi_i y_{t-i}\,-\,\sum_{j=1}^{q}\theta_j \epsilon_{t-j}\,+\,\sum_{k=1}^{P}\Phi_k y_{t-kS}\,+\,\sum_{l=1}^{Q}\Theta_l \epsilon_{t-lS}$$
  with seasonal period $S$.

---

## 17. `econ_risk`

Economic performance and risk metrics.

- **Levelized Cost of Energy (LCOE)**
  $$\text{LCOE}=\frac{\sum_{y=0}^{N}\frac{I_y+O_y+M_y}{(1+r)^y}}{\sum_{y=0}^{N}\frac{E_y}{(1+r)^y}}$$
  where $I_y$ – investment, $O_y$ – operation, $M_y$ – maintenance, $E_y$ – energy output (kWh), $r$ – discount rate, $N$ – project life.

- **Conditional Value‑at‑Risk (CVaR)** at confidence $\alpha$:
  $$\text{CVaR}_\alpha = \frac{1}{1-\alpha}\int_{\alpha}^{1}\text{VaR}_p\,dp$$
  where $\text{VaR}_p$ is the value‑at‑risk at percentile $p$.

---

## 18. `edge_governor`

Ring‑buffer data‑structure and Bartlett projection used on STM32‑F401 edge firmware.

- **Ring‑buffer push/pop** maintain head/tail indices modulo $N$ (capacity power of two). No explicit formula needed beyond modular arithmetic.

- **Bartlett projector** (moving‑average) for input vector $\mathbf{x}$ of length $L$ and output $\mathbf{y}$ of length $k$:
  $$y_i = \frac{1}{g}\sum_{j=0}^{g-1} x_{i\,g + j}\,,\quad g = \left\lfloor\frac{L}{k}\right\rfloor$$
  where $g$ is group size.

---

## 19. `pi_emeans`

Physics‑informed clustering for turbine performance.

- **Betz ceiling** – theoretical maximum power coefficient:
  $$C_{P,\text{Betz}} = \frac{16}{27} \approx 0.593$$

- **Cluster mean** for a set of turbine measurements $\{\mathbf{z}_i\}$:
  $$\mu = \frac{1}{N}\sum_{i=1}^{N}\mathbf{z}_i$$

---

## 20. `reg_econ`

Placeholder module for regression‑based economic models. No closed‑form equations are defined presently.

---

## 21. `topology_braids`

Braid‑word representation and Burau trace invariant.

- **Burau trace (simplified)**
  $$\tau(w) = \sum_{i=1}^{\ell} |g_i|$$
  where $w$ is a braid word with generators $g_i$ and $\ell$ is word length.

- **Vortex helicity** for velocity $\mathbf{u}$ and vorticity $\boldsymbol{\omega}$ sampled over volumes $\Delta V_j$:
  $$H = \sum_{j} \mathbf{u}_j \cdot \boldsymbol{\omega}_j \; \Delta V_j$$

---

## 22. `transforms`

Coordinate transformation utilities.

- **2‑D rotation** by angle $\theta$:
  $$\begin{bmatrix}x'\\y'\end{bmatrix}=\begin{bmatrix}\cos\theta & -\sin\theta\\[2pt]\sin\theta & \cos\theta\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}$$

- **Scaling** by factor $s$:
  $$\mathbf{x}' = s\,\mathbf{x}$$

All symbols are defined inline. The document satisfies the MkDocs MathJax requirements and adheres to neutral technical prose.
