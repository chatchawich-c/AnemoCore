import numpy as np
from typing import Any

# Low‑level imports
from ._core import (
    py_fft2d,
    py_load_wtg,
    py_load_windio,
    py_compute_aep_full,

)

class WindFarm:
    """High‑level wrapper for wind‑farm data handling.

    The class loads wind‑speed / direction fields and provides a convenient
    ``step_vsg`` method that forwards to the low‑level ``py_kam_vsg_step``.
    """

    def __init__(self, data: Any):
        # ``data`` may be a NumPy array, a file path, or a custom object.
        self.data = data

    @classmethod
    def from_windio(cls, source: str) -> "WindFarm":
        """Load a wind‑io formatted file.

        Parameters
        ----------
        source: str
            Path to a wind‑io JSON / CSV representation.
        """
        arr = py_load_windio(source)
        return cls(arr)

    @classmethod
    def from_wtg(cls, source: str) -> "WindFarm":
        """Load a WTG (wind turbine generator) dataset.
        """
        arr = py_load_wtg(source)
        return cls(arr)

    def fft2d(self) -> Any:
        """Return a 2‑D FFT of the stored field using the low‑level binding."""
        return py_fft2d(self.data)

    def step_vsg(self, *args, **kwargs) -> Any:
        """Perform a single VSG (virtual shear‑gradient) step.

        This method forwards all arguments to the underlying ``py_kam_vsg_step``
        function. The signature matches the original Rust binding.
        """
        return py_kam_vsg_step(*args, **kwargs)

    @classmethod
    def simulate(cls, wind_speeds: np.ndarray, power_curve: str = "PowerLaw", shear_model: str = "PowerLaw") -> "WindFarm":
        """Simulate a wind farm given wind speed series.
        This stub wraps the low‑level ``py_compute_aep_full`` function.
        ``power_curve`` and ``shear_model`` are currently ignored and present for API compatibility.
        """
        # In a full implementation, map strings to Rust enum constructors.
        # Here we simply compute AEP using the existing binding.
        aep = py_compute_aep_full(wind_speeds.tolist())  # type: ignore[arg-type]
        # Store result in a simple dict for later use.
        data = {"aep": aep, "speeds": wind_speeds}
        return cls(data)

    def bankable_aep(self, discount_rate: float = 0.05) -> float:
        """Return a bankable AEP estimate.
        This placeholder returns the stored AEP discounted over a 20‑year horizon.
        """
        aep = self.data.get("aep", 0.0)
        years = 20
        # Simple discounted cash flow approximation.
        factor = (1 - (1 + discount_rate) ** -years) / discount_rate
        return aep * factor
