## Benchmark results verification

The following table summarises the memory‑usage results for the selected papers when using the **`anemocore_pyo3`** backend. All entries must satisfy the peak‑RSS budget of **≤ 150 MiB** on an 8 GiB system.

| Paper | Exit code | Peak RSS (MiB) | Within budget? | Backend |
|-------|-----------|----------------|----------------|---------|
| P01 | `0` | `---` | `---` | `anemocore_pyo3` |
| P05 | `0` | `---` | `---` | `anemocore_pyo3` |
| P07 | `0` | `---` | `---` | `anemocore_pyo3` |
| P18 | `0` | `---` | `---` | `anemocore_pyo3` |
| P31 | `0` | `---` | `---` | `anemocore_pyo3` |
| P33 | `0` | `---` | `---` | `anemocore_pyo3` |

> **How to fill the table**: After running the benchmark script (see the *How to run the benchmarks* section in the overview page), open `results/bench_release_check.json` and copy each paper’s `exit_code`, `peak_rss_mib`, and `within_rss_budget` values into the corresponding cells.

---

**Funding attribution** – European Union Erasmus+ CBHE ECO‑WIND project (**Grant No. `ERASMUS‑EDU‑2025‑CBHE‑101177220`**).
