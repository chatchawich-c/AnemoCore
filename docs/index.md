# AnemoCore Documentation

## Overview

**AnemoCore v0.3.5** – a high-performance, zero-duplication wind energy resource assessment (WRA), wake modeling, operational assessment (OA), drivetrain reliability, and grid‑stability engine written in pure Rust with native Python (`PyO3` / `cp39-abi3`) and R (`extendr`) bindings.

* **Wheel size:** `4.87 MiB` stripped universal `cp39-abi3` wheel (≤ 5 MiB budget)
* **Peak RSS budget:** `71 MiB` base import (≤ 150 MiB on an 8 GiB system, with lazy `anemocore.viz`)
* **Zero‑duplication data policy:** Benchmark datasets are external and never duplicated in the repository.
* **Funding attribution:** Supported by the **European Union Erasmus+ Capacity Building in Higher Education (CBHE)** project **ECO‑WIND** (*Enhancing Capacity in Condition-based Maintenance of Wind Energy*, **Grant No. `ERASMUS‑EDU‑2025‑CBHE‑101177220`**).

---

## Installation

### Python (Pre-built Wheel — Recommended)
Download `anemocore-0.3.5-cp39-abi3-macosx_10_12_x86_64.whl` from [GitHub Releases](https://github.com/chatchawich-c/AnemoCore/releases/tag/v0.3.5) and install via `pip`:

```bash
pip install anemocore-0.3.5-cp39-abi3-macosx_10_12_x86_64.whl
```

### Build from Source (Python & R)

```bash
# Python installation via GitHub repository
python3 -m pip install "git+https://github.com/chatchawich-c/AnemoCore.git#subdirectory=bindings/python"

# R installation via remotes
R -q -e 'remotes::install_github("chatchawich-c/AnemoCore", subdir = "bindings/r")'
```

---

## Getting Started

*See the **[Usage](usage.md)** page for quick‑start examples using `WindClimate`, `WindFarm`, and `Drivetrain`, or browse the **[API Manual](API_USAGE_MANUAL.md)** for full Rust, Python, and R function references.*

---
**Funding Attribution:** EU Erasmus+ CBHE ECO-WIND (Grant No. `ERASMUS-EDU-2025-CBHE-101177220`).
