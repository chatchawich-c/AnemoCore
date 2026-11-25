# ECO‑WIND Benchmarks Overview

This section documents the ECO‑WIND benchmark suite evaluating wind‑energy resource assessment (WRA), wake modeling, operational assessment (OA), and condition‑based maintenance algorithms.

* **Architecture**: The benchmark suite lives in the external [`ecowind-benchmarks`](https://github.com/chatchawich-c/AnemoCore) repository and invokes the compiled **AnemoCore v0.3.0** Rust‑backed Python wheel (`anemocore` / `anemocore._core` via `PyO3` `cp39-abi3`) for core numerical execution.
* **Zero‑duplication policy**: No raw datasets or benchmark scripts are duplicated into this repository. All benchmark data are streamed directly from the external benchmark repository at runtime.
* **Citation**: Please cite the benchmark suite via its Zenodo archive DOI `10.5281/zenodo.24496449` (placeholder).
* **Funding attribution**: This work is supported by the **European Union Erasmus+ Capacity Building in Higher Education (CBHE)** project **ECO‑WIND** (*Enhancing Capacity in Condition-based Maintenance of Wind Energy*, **Grant No. `ERASMUS‑EDU‑2025‑CBHE‑101177220`**).

---

## Release v0.3.0 Runtime & Memory Footprint

AnemoCore v0.3.0 maintains a lightweight deployment profile suitable for edge devices and typical workstations:

| Metric | Target Budget | AnemoCore v0.3.0 (`cp39-abi3` macOS x86_64) | Status |
| :--- | :--- | :--- | :--- |
| **Stripped Wheel Size** | `< 5.0 MiB` | **`485 KiB`** (`0.47 MiB`) | Pass |
| **Base Import Peak RSS** | `< 150.0 MiB` | **`67.8 MiB`** (lazy `anemocore.viz`) | Pass |
| **Wake Superposition Parity** | `<= 1e-4` rel. tol. | Verified vs. `windIO` (`Jensen`, `Bastankhah2014/2016`) | Pass |
| **Supported Scope** | – | WRA I/O, Shear, Wake/OA, Drivetrain Tribology, KAM-VSG | Pass |

---

## How to run the benchmarks

Install the **AnemoCore v0.2.0** wheel first, then run the validation suite against selected papers:

```bash
pip install anemocore-0.2.0-cp39-abi3-macosx_10_12_x86_64.whl

cd ../ecowind-benchmarks && \
python3 scripts/run_all_benchmarks.py \
  --only P01 P05 P07 P18 P31 P33 \
  --out results/bench_release_check.json
```

The runner invokes the native `anemocore._core` Rust backend for each selected paper and exports execution latency and peak RSS memory statistics to JSON. See [Results](results/) for detailed per-paper parity and runtime tables.

---

## Version History

| Version | Wheel Size | Base Peak RSS | Key Milestones |
| :--- | :--- | :--- | :--- |
| **`v0.2.0`** *(Latest)* | `485 KiB` | `67.8 MiB` | Added 5-format WRA I/O, Vertical Wind Shear suite, `windIO`-verified Wake/OA, Drivetrain Tribology & `kam_vsg` |
| **`v0.1.0`** | Initial build | Baseline | Core Weibull parameter estimation (`MLE`, `MMLE`, `MoM`, `EPFM`), QMC integration & initial PyO3 bindings (`anemocore_pyo3`) |
