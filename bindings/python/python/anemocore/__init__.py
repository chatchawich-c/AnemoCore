"""High‑level Python façade for AnemoCore.

The package re‑exports the low‑level Rust bindings from the compiled
module ``anemocore._core`` and provides user‑friendly classes that wrap the
core functionality. Existing symbols (e.g. ``fit_weibull``) are also exposed
so that legacy benchmark scripts continue to work unchanged.
"""

# Re‑export low‑level symbols for backward compatibility
from ._core import (
    fit_weibull,
    qmc_weibull_cubic,
    py_fft2d,
    py_load_wtg,
    py_load_windio,
    py_geostrophic_from_atlas,
    py_atlas_from_geostrophic,
    py_kam_vsg_step,
    py_thermal_stress_indicators,
    py_arrhenius_ctd_p6,
    py_moist_air_density_cipm2007,
    py_gearbox_power_loss,
    py_step_gearbox_temperature,
    py_ehl_two_stage,
    py_iso281_l10m,
    py_iso16281_tilt_factor,
    py_caputo_peridynamics_damage,
)

# High‑level façade classes
from .climate import WindClimate
from .farm import WindFarm
from .drivetrain import Drivetrain

__all__ = [
    # low‑level API
    "fit_weibull",
    "qmc_weibull_cubic",
    "py_fft2d",
    "py_load_wtg",
    "py_load_windio",
    "py_geostrophic_from_atlas",
    "py_atlas_from_geostrophic",
    "py_kam_vsg_step",
    "py_thermal_stress_indicators",
    "py_arrhenius_ctd_p6",
    "py_moist_air_density_cipm2007",
    "py_gearbox_power_loss",
    "py_step_gearbox_temperature",
    "py_ehl_two_stage",
    "py_iso281_l10m",
    "py_iso16281_tilt_factor",
    "py_caputo_peridynamics_damage",
    # high‑level API
    "WindClimate",
    "WindFarm",
    "Drivetrain",
]
