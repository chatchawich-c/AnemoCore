# AnemoCore Documentation

## Overview

**AnemoCore** v0.1.0 – a low‑footprint Rust, PyO3 Python, and R library for wind‑energy condition‑based maintenance, microsite/wind‑resource modelling, and grid‑stability workflows.

- **Wheel size:** < 5 MiB
- **Peak RSS budget:** ≤ 150 MiB on an 8 GiB system
- **Zero‑duplication data policy** – benchmark data are external and not stored in the repository.
- **Funding attribution**: European Union Erasmus+ CBHE ECO‑WIND project (Grant No. `ERASMUS‑EDU‑2025‑CBHE‑101177220`).

---
## Installation

```bash
# Python installation via PyPI
pip install anemocore

# Python installation via GitHub repository
python3 -m pip install "git+https://github.com/chatchawich-c/AnemoCore.git#subdirectory=bindings/python"

# R installation via remotes
R -q -e 'remotes::install_github("chatchawich-c/AnemoCore", subdir = "bindings/r")'

# R-universe (optional)
R -q -e 'install.packages("anemocore", repos = c("https://chatchawich-c.r-universe.dev", "https://cloud.r-project.org"))'
```

---

## Getting Started

*See the **Usage** page for installation and quick‑start examples.*
