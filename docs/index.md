# AnemoCore Documentation

## Overview

**AnemoCore v0.2.0** – a high-performance, zero-duplication wind energy resource assessment (WRA), wake modeling, operational assessment (OA), drivetrain reliability, and grid-stability engine written in pure Rust with native Python (`PyO3` / `cp39-abi3`) and R (`extendr`) bindings.

* **Wheel size:** `485 KiB` stripped universal `cp39-abi3` wheel (`< 5 MiB` budget)
* **Peak RSS budget:** `67.8 MiB` base import (`≤ 150 MiB` on an 8 GiB system, with lazy `anemocore.viz`)
* **Zero‑duplication data policy:** Benchmark datasets are external and never duplicated in the repository.
* **Funding attribution:** Supported by the **European Union Erasmus+ Capacity Building in Higher Education (CBHE)** project **ECO‑WIND** (*Enhancing Capacity in Condition-based Maintenance of Wind Energy*, **Grant No. `ERASMUS‑EDU‑2025‑CBHE‑101177220`**).

---

## Installation

### Python (Pre-built Wheel — Recommended)
Download `anemocore-0.2.0-cp39-abi3-macosx_10_12_x86_64.whl` from [GitHub Releases](https://github.com/chatchawich-c/AnemoCore/releases/tag/v0.2.0) and install via `pip`:

```bash
pip install anemocore-0.2.0-cp39-abi3-macosx_10_12_x86_64.whl
```

### Build from Source (Python & R)

```bash
# Python installation via GitHub repository
python3 -m pip install "git+[https://github.com/chatchawich-c/AnemoCore.git#subdirectory=bindings/python](https://github.com/chatchawich-c/AnemoCore.git#subdirectory=bindings/python)"

# R installation via remotes
R -q -e 'remotes::install_github("chatchawich-c/AnemoCore", subdir = "bindings/r")'

# R-universe (optional)
R -q -e 'install.packages("anemocore", repos = c("[https://chatchawich-c.r-universe.dev](https://chatchawich-c.r-universe.dev)", "[https://cloud.r-project.org](https://cloud.r-project.org)"))'
```

---

## Getting Started

*See the **[Usage](usage/)** page for quick-start examples using `WindClimate`, `WindFarm`, and `Drivetrain`, or browse the **[API Manual](API_USAGE_MANUAL/)** for full Rust, Python, and R function references.*
