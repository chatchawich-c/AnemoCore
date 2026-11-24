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


## Quickstart

```python
from anemocore import fit_weibull, qmc_weibull_cubic, py_fft2d, py_load_wtg, py_load_windio, py_kam_vsg_step

# Fit Weibull distribution
k, c = fit_weibull([1.0, 2.5, 3.0, 4.0])

# QMC Weibull cubic expectation
exp = qmc_weibull_cubic(k, c, 10000)

# Simple 2‑D FFT
grid = [1.0, 2.0, 3.0, 4.0]
out, nx, ny = py_fft2d(grid, nx=2, ny=2)

# Load placeholder data
ok_wtg = py_load_wtg(b"\x00\x01")
ok_windio = py_load_windio("""\nwind_speed: 12.5\n""")

# Run a KAM‑VSG step (example spec & state dictionaries)
spec = {"j_r": 0.5, "j_g": 0.3, "k0": 0.0, "eps_k1": 0.0, "d_s": 0.1, "k_sync": 0.0, "omega_0": 1.0, "p_base_mw": 5.0, "p_min_mw": 0.0, "p_max_mw": 5.0, "t_aero_base_nm": 0.0, "t_coolant_c": 20.0, "delta_t_hs_c": 0.0}
state = {"theta_s": 0.0, "dtheta_s": 0.0}
result = py_kam_vsg_step(spec, state, "grid_following", rocof=0.0, delta_f=0.0, t_aero_disturb_nm=0.0, dt=0.01)
print(result)
```

## Benchmark Execution

```bash
cd ../ecowind-benchmarks && \
python3 scripts/run_all_benchmarks.py \
    --only P01 P05 P07 P18 P31 P33 \
    --out results/bench_release_check.json
```

See the **Equations** page for the underlying mathematical models.
