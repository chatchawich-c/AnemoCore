import numpy as np
from typing import Optional

# Import low‑level functions
from ._core import (
    fit_weibull,
    qmc_weibull_cubic,
    py_moist_air_density_cipm2007,
    py_parse_lib,
    py_parse_gwc,
    py_parse_rsf,
    py_parse_dem,

)

class WindClimate:
    """High‑level wrapper for Weibull fitting and atmospheric properties.

    The class provides a convenient API that mirrors the original benchmark
    scripts while delegating the heavy numeric work to the compiled Rust
    bindings.
    """

    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=float)
        self.data = data
        # Fit Weibull distribution immediately for convenience
        self.k, self.c = fit_weibull(self.data.tolist())

    @classmethod
    def fit(cls, data: np.ndarray) -> "WindClimate":
        """Create a :class:`WindClimate` instance from raw wind speed data."""
        return cls(data)

    def cubic_expectation(self, n: int = 100_000) -> float:
        """Return the QMC estimate of the Weibull cubic expectation.

        Parameters
        ----------
        n: int, default 100 000
            Number of quasi‑Monte‑Carlo samples.
        """
        return qmc_weibull_cubic(self.k, self.c, n)

    def moist_air_density(self, pressure_pa: float, temperature_k: float, rh: float) -> float:
        """Return moist air density using the CIPM‑2007 standard.

        Parameters are passed directly to the low‑level Rust function.
        """
        return py_moist_air_density_cipm2007(pressure_pa, temperature_k, rh)

    def arrhenius_ctd(self, temps_k: np.ndarray, t_ref_k: float, ea_j_mol: float, dt_sec: float) -> float:
        """Cumulative thermal dose (CTD P6) for a temperature series.
        """
        return py_arrhenius_ctd_p6(temps_k, t_ref_k, ea_j_mol, dt_sec)
    @classmethod
    def from_tab(cls, path: str) -> "WindClimate":
        """Load Tabular WRA file via low‑level parser and create a ``WindClimate``.
        """
        data = py_parse_lib(path)
        return cls(np.asarray(data, dtype=float))

    @classmethod
    def from_gwc(cls, path: str) -> "WindClimate":
        """Load GWC file (placeholder) – uses the same parser as ``from_tab`` for now.
        """
        data = py_parse_gwc(path)
        return cls(np.asarray(data, dtype=float))

    @classmethod
    def from_rsf(cls, path: str) -> "WindClimate":
        """Load RSF file via low‑level parser.
        """
        data = py_parse_rsf(path)
        return cls(np.asarray(data, dtype=float))

    @classmethod
    def from_task43(cls, path: str) -> "WindClimate":
        """Alias for ``from_rsf`` – historic naming.
        """
        return cls.from_rsf(path)

    @classmethod
    def from_dem(cls, path: str) -> "WindClimate":
        """Load DEM data.
        Supports GeoTIFF via optional ``xarray``/``rasterio``. Falls back to ESRI ASCII ``.asc`` parser.
        """
        if path.lower().endswith(('.tif', '.tiff')):
            try:
                import xarray as xr  # optional
                arr = xr.open_rasterio(path).values
                return cls(arr)
            except Exception:
                # fallback to Rust parser for ASCII DEM
                data = py_parse_dem(path)
                return cls(np.asarray(data, dtype=float))
        else:
            data = py_parse_dem(path)
            return cls(np.asarray(data, dtype=float))
