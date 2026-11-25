# Roadmap & Architecture Overview

## Version History

| Version | Highlights |
|--------|------------|
| **v0.1.x – v0.2.x** | Initial wind‑statistics library, pure‑Rust core with `#![no_std]` compatibility. First Python (`pyo3`) and R (`extendr`) bindings introduced. |
| **v0.3.0 – v0.3.1** | Added ECO‑WIND D3.3 thermal models, tribology utilities, peridynamics, SCADA‑based health‑index, wake‑superposition, and quasi‑Monte‑Carlo sampling. Expanded test suite and documentation. |
| **v0.3.2** | New **heat diffusion** graph‑Laplacian implementation, corrected **Arrhenius oil‑life** sign, added **ZDDP additive depletion** model, updated documentation, and released Python ≥ 3.9 ABI3 wheel. |
| **v0.4.x+** (future) | Planned development workflow: modular plug‑in system, dual‑horizon RL derating interface (PPO/SAC) coupled with P4/CTD, advanced wake models with analytic gradients, IEC 61400‑15‑1 AEP banking, Monte‑Carlo risk analysis, and grid‑control (`kam_vsg`) enhancements. |

## Architecture Summary

- **Core (`crates/anemocore/src/`)** – Pure‑Rust, `#![no_std]`‑compatible where required. Public API stable across versions.
- **Python bindings** – `pyo3` with ABI3 support (wheel `anemocore‑0.3.2‑cp39‑abi3‑macosx_10_12_x86_64.whl`). Facade modules expose high‑level functions (`heat_diffuse`, `arrhenius_oil_life`, `zddp_depletion`).
- **R bindings** – `extendr` providing analogous entry points for the same core kernels.
- **Documentation** – MkDocs site with `MATHEMATICAL_FORMULATIONS.md` containing LaTeX equations, SI units, and constant definitions. `API_USAGE_MANUAL.md` reflects current release version.

The roadmap is maintained in this file and will be updated as new capabilities are merged.
