"""High-level Python facade for AnemoCore."""

from __future__ import annotations

import math

PHI: float = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV: float = (math.sqrt(5.0) - 1.0) / 2.0
FRAME_BOUND_A: float = 2.482
FRAME_BOUND_B: float = 2.492

from ._core import (
    fit_weibull,
    qmc_weibull_cubic,
    py_fft2d,
    py_load_wtg,
    py_load_windio,
    py_geostrophic_from_atlas,
    py_atlas_from_geostrophic,
    py_kam_vsg_step,
    py_kam_vsg_step3,
    py_kam_vsg_step_compat,
    py_kam_vsg_step_full,
    py_jacobi_horizon_guard,
    py_jacobi_horizon_guard3,
    py_thermal_stress_indicators,
    py_arrhenius_ctd_p6,
    py_moist_air_density_cipm2007,
    py_gearbox_power_loss,
    py_step_gearbox_temperature,
    py_ehl_two_stage,
    py_iso281_l10m,
    py_iso16281_tilt_factor,
    py_caputo_peridynamics_damage,
    py_parse_lib,
    py_parse_gwc,
    py_parse_rsf,
    py_parse_dem,
    py_extrapolate_wind_speed,
    py_extrapolate_weibull_justus_mikhail,
    py_normalize_wind_speed_iec,
    py_compute_aep_full,
    py_compute_aep_full_with_model,
    py_prandtl_tip_hub_loss,
    py_bem_steady_rotor,
    py_beddoes_leishman_stall,
    py_aero_torque_thrust_curve,
    py_mixed_ehl_asperity_load,
    py_micropitting_fatigue_index,
    py_bearing_spall_progression,
    py_lubricant_degradation_viscosity,
    py_sarimax_fit,
    py_sarimax_forecast_residuals,
    py_fleet_cross_turbine_residual,
    py_multi_domain_health_index,
    py_scada_missingness_diagnostic,
    py_compute_lcoe,
    py_compute_lcoh,
    py_om_unscheduled_cost_model,
    py_coupled_monte_carlo_risk,
    py_var_cvar_project_npv,
    py_wpt_db4_indicators,
    py_quasicrystal_shift_add_energies,
    py_quasicrystal_shift_add_split,
    py_fib_qmc_limit_state,
    py_fib_golden_stopping,
    py_fib_qmc_sweep_o1,
    py_market_roll_spread,
    py_market_lob_density,
    py_phyllotaxis_aerothermal_snapshot,
    py_pi_emeans_alpha,
    py_pi_emeans_cyber_alpha,
    py_bartlett_factor_project_vec,
    py_stealth_fdia_wind_check,
    py_weyl_convergent_lags,
    py_rank1_lattice_qmc,
)

# Non-prefixed aliases for backward compatibility and test parity
kam_vsg_step = py_kam_vsg_step3
jacobi_horizon_guard = py_jacobi_horizon_guard3
wpt_db4_indicators = py_wpt_db4_indicators
quasicrystal_shift_add_energies = py_quasicrystal_shift_add_energies
quasicrystal_shift_add_split = py_quasicrystal_shift_add_split
fib_qmc_limit_state = py_fib_qmc_limit_state
fib_golden_stopping = py_fib_golden_stopping
fib_qmc_sweep_o1 = py_fib_qmc_sweep_o1
market_roll_spread = py_market_roll_spread
market_lob_density = py_market_lob_density
phyllotaxis_aerothermal_snapshot = py_phyllotaxis_aerothermal_snapshot
pi_emeans_alpha = py_pi_emeans_alpha
pi_emeans_cyber_alpha = py_pi_emeans_cyber_alpha
bartlett_factor_project_vec = py_bartlett_factor_project_vec
stealth_fdia_wind_check = py_stealth_fdia_wind_check
weyl_convergent_lags = py_weyl_convergent_lags
rank1_lattice_qmc = py_rank1_lattice_qmc

# High-level facade classes (viz remains lazy so matplotlib is not imported eagerly)
from .climate import WindClimate
from .farm import WindFarm
from .drivetrain import Drivetrain
from .aero import AeroRotor
from .fleet_scada import FleetSCADA
from .wind_economics import WindEconomics


def __getattr__(name: str):
    if name == "viz":
        from . import viz as _viz
        return _viz
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PHI",
    "PHI_INV",
    "FRAME_BOUND_A",
    "FRAME_BOUND_B",
    "fit_weibull",
    "qmc_weibull_cubic",
    "py_fft2d",
    "py_load_wtg",
    "py_load_windio",
    "py_geostrophic_from_atlas",
    "py_atlas_from_geostrophic",
    "py_kam_vsg_step",
    "py_kam_vsg_step_compat",
    "py_kam_vsg_step_full",
    "py_jacobi_horizon_guard",
    "py_thermal_stress_indicators",
    "py_arrhenius_ctd_p6",
    "py_moist_air_density_cipm2007",
    "py_gearbox_power_loss",
    "py_step_gearbox_temperature",
    "py_ehl_two_stage",
    "py_iso281_l10m",
    "py_iso16281_tilt_factor",
    "py_caputo_peridynamics_damage",
    "py_parse_lib",
    "py_parse_gwc",
    "py_parse_rsf",
    "py_parse_dem",
    "py_extrapolate_wind_speed",
    "py_extrapolate_weibull_justus_mikhail",
    "py_normalize_wind_speed_iec",
    "py_compute_aep_full",
    "py_compute_aep_full_with_model",
    "py_prandtl_tip_hub_loss",
    "py_bem_steady_rotor",
    "py_beddoes_leishman_stall",
    "py_aero_torque_thrust_curve",
    "py_mixed_ehl_asperity_load",
    "py_micropitting_fatigue_index",
    "py_bearing_spall_progression",
    "py_lubricant_degradation_viscosity",
    "py_sarimax_fit",
    "py_sarimax_forecast_residuals",
    "py_fleet_cross_turbine_residual",
    "py_multi_domain_health_index",
    "py_scada_missingness_diagnostic",
    "py_compute_lcoe",
    "py_compute_lcoh",
    "py_om_unscheduled_cost_model",
    "py_coupled_monte_carlo_risk",
    "py_var_cvar_project_npv",
    "py_wpt_db4_indicators",
    "py_quasicrystal_shift_add_energies",
    "py_quasicrystal_shift_add_split",
    "py_fib_qmc_limit_state",
    "py_fib_golden_stopping",
    "py_fib_qmc_sweep_o1",
    "py_market_roll_spread",
    "py_market_lob_density",
    "py_phyllotaxis_aerothermal_snapshot",
    "py_pi_emeans_alpha",
    "py_pi_emeans_cyber_alpha",
    "py_bartlett_factor_project_vec",
    "py_stealth_fdia_wind_check",
    "py_weyl_convergent_lags",
    "py_rank1_lattice_qmc",
    "kam_vsg_step",
    "jacobi_horizon_guard",
    "wpt_db4_indicators",
    "quasicrystal_shift_add_energies",
    "quasicrystal_shift_add_split",
    "fib_qmc_limit_state",
    "fib_golden_stopping",
    "fib_qmc_sweep_o1",
    "market_roll_spread",
    "market_lob_density",
    "phyllotaxis_aerothermal_snapshot",
    "pi_emeans_alpha",
    "pi_emeans_cyber_alpha",
    "bartlett_factor_project_vec",
    "stealth_fdia_wind_check",
    "weyl_convergent_lags",
    "rank1_lattice_qmc",
    "WindClimate",
    "WindFarm",
    "Drivetrain",
    "AeroRotor",
    "FleetSCADA",
    "WindEconomics",
    "viz",
]
