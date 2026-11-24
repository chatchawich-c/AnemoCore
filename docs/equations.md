## Equations and Derivations

### Betz Limit
The maximum theoretical power coefficient of a wind turbine is given by the Betz limit:

$$
C_p = \frac{16}{27} \approx 0.593
$$

### Weibull Distribution
The probability density function (PDF) of the Weibull distribution used for wind speed modelling is:

$$
f(v; k, c) = \frac{k}{c}\left(\frac{v}{c}\right)^{k-1} \exp\left[-\left(\frac{v}{c}\right)^k\right]
$$

where $v$ is wind speed, $k$ is the shape parameter, and $c$ is the scale parameter.

### Inverse Laplace Transform (Stehfest)
The Stehfest algorithm approximates the inverse Laplace transform $\mathcal{L}^{-1}\{F(s)\}(t)$ as:

$$
f(t) \approx \frac{\ln 2}{t} \sum_{i=1}^{N} \omega_i \; F\left(\frac{i\,\ln 2}{t}\right)
$$

with weights $\omega_i$ computed from combinatorial formulas (see `transforms.rs`).

### Inverse Laplace Transform (Talbot)
The Talbot contour method evaluates:

$$
f(t) \approx \frac{1}{t}\sum_{k=1}^{N} \Re\left[\,\beta \; F\left(\frac{\beta}{t}\right) \exp\left(\frac{\beta}{t}\right)\right]
$$

where $\beta$ is a scaling constant (set to $1.0$ in the implementation).

### KAM‑VSG Dynamics
The Kinetic Energy (KE) balance for a Virtual Synchronous Generator (VSG) is modeled as:

$$
J\,\dot{\omega} = T_{m} - T_{e} - D\,(\omega-\omega_{0})
$$

with inertia $J$, mechanical torque $T_{m}$, electrical torque $T_{e}$, damping $D$, and nominal frequency $\omega_{0}$.

---

All equations are implemented in the Rust core and exposed to Python via the PyO3 bindings.
