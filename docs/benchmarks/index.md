# ECO‑WIND Benchmarks Overview

This section documents the **66‑paper ECO‑WIND benchmark suite** that evaluates wind‑energy condition‑based maintenance algorithms.

- **Architecture**: The suite lives in a separate repository [`ecowind-benchmarks`](https://github.com/chatchawich-c/AnemoCore) and calls the **AnemoCore** Python wheel `anemocore_pyo3` for core calculations.
- **Zero‑duplication policy**: No raw datasets or benchmark scripts are copied into this repository. All data are accessed directly from the external benchmark repository at runtime.
- **Citation**: Please cite the benchmark suite via its Zenodo archive DOI `10.5281/zenodo.24496449` (placeholder).
- **Funding attribution**: This work is supported by the **European Union Erasmus+ CBHE ECO‑WIND** project (**Grant No. `ERASMUS‑EDU‑2025‑CBHE‑101177220`**).

---

## How to run the benchmarks

```bash
cd ../ecowind-benchmarks && \
python3 scripts/run_all_benchmarks.py \
    --only P01 P05 P07 P18 P31 P33 \
    --out results/bench_release_check.json
```

The script will invoke the `anemocore_pyo3` backend for the selected papers and produce a JSON file with memory‑usage statistics.
