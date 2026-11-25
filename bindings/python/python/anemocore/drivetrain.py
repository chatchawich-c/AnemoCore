import numpy as np
from typing import Any

# Low‑level imports
from ._core import (
    py_thermal_stress_indicators,
    py_arrhenius_ctd_p6,
    py_moist_air_density_cipm2007,
    py_gearbox_power_loss,
    py_step_gearbox_temperature,
    py_ehl_two_stage,
    py_iso281_l10m,
    py_iso16281_tilt_factor,
    py_caputo_peridynamics_damage,
    # New tribology functions
    py_mixed_ehl_asperity_load,
    py_micropitting_fatigue_index,
    py_bearing_spall_progression,
    py_lubricant_degradation_viscosity,
)

class Drivetrain:
    """High‑level façade for drivetrain thermal‑tribology analysis.

    The class provides a simple ``run`` method that demonstrates a typical
    workflow: compute thermal stress indicators, cumulative dose, gearbox
    losses, temperature evolution, EHL film thickness, bearing life, tilt
    factor and Caputo damage. All heavy‑lifting is delegated to the compiled
    Rust bindings.
    """

    def __init__(self):
        pass

    def run(
        self,
        temps_k: np.ndarray,
        t_ref_k: float,
        ea_j_mol: float,
        dt_sec: float,
        pressure_pa: float,
        rh: float,
        gearbox_params: dict,
        ehl_params: dict,
        bearing_params: dict,
        tilt_params: dict,
        caputo_params: dict,
    ) -> dict:
        """Execute the full analysis pipeline.

        Parameters
        ----------
        temps_k: np.ndarray
            Temperature series (K).
        t_ref_k: float
            Reference temperature for CTD.
        ea_j_mol: float
            Activation energy.
        dt_sec: float
            Time step.
        pressure_pa: float
            Atmospheric pressure for moist air density.
        rh: float
            Relative humidity (0‑1).
        gearbox_params: dict
            Keys required by ``py_gearbox_power_loss``.
        ehl_params: dict
            Keys required by ``py_ehl_two_stage``.
        bearing_params: dict
            Keys for ``py_iso281_l10m``.
        tilt_params: dict
            Keys for ``py_iso16281_tilt_factor``.
        caputo_params: dict
            Keys for ``py_caputo_perdynamics_damage``.

        Returns
        -------
        dict
            Mapping of step name to result.
        """
        # Thermal stress indicators (P1‑P5)
        indicators = py_thermal_stress_indicators(temps_k)

        # Arrhenius CTD (P6)
        ctd = py_arrhenius_ctd_p6(temps_k, t_ref_k, ea_j_mol, dt_sec)

        # Moist air density (optional auxiliary output)
        density = py_moist_air_density_cipm2007(pressure_pa, temps_k[0], rh)

        # Gearbox loss (total power loss W)
        gearbox_loss = py_gearbox_power_loss(**gearbox_params)

        # Temperature step (example: start temperature 300 K)
        t_next = py_step_gearbox_temperature(
            t_curr_k=temps_k[0],
            t_amb_k=temps_k[0],
            p_loss_w=gearbox_loss,
            g_th_w_per_k=gearbox_params.get("g_th_w_per_k", 0.0),
            c_th_j_per_k=gearbox_params.get("c_th_j_per_k", 0.0),
            dt_s=dt_sec,
        )

        # EHL two‑stage film thickness
        ehl = py_ehl_two_stage(**ehl_params)

        # ISO‑281 bearing life (L10m)
        bearing_l10m = py_iso281_l10m(**bearing_params)

        # Tilt derating factor
        tilt_factor = py_iso16281_tilt_factor(**tilt_params)

        # Caputo peridynamics damage
        caputo_damage = py_caputo_peridynamics_damage(**caputo_params)

        return {
            "thermal_indicators": indicators,
            "ctd": ctd,
            "moist_air_density": density,
            "gearbox_loss_w": gearbox_loss,
            "next_gearbox_temp_k": t_next,
            "ehl": ehl,
            "bearing_l10m": bearing_l10m,
            "tilt_factor": tilt_factor,
            "caputo_damage": caputo_damage,
        }

    # New tribology wrapper methods
    def mixed_ehl_asperity_load(self, l_we: float, l_sd: float, sigma: float) -> float:
        """Wrapper for mixed EHL asperity load.

        Parameters correspond to ``l_we``, ``l_sd``, and ``sigma``.
        """
        return py_mixed_ehl_asperity_load(l_we, l_sd, sigma)

    def micropitting_fatigue_index(self, stress: float, cycles: int) -> float:
        """Wrapper for micropitting fatigue index.

        ``cycles`` is an integer number of cycles.
        """
        return py_micropitting_fatigue_index(stress, cycles)

    def bearing_spall_progression(self, spall_size: float, time_hours: float) -> float:
        """Wrapper for bearing spall progression.

        ``spall_size`` and ``time_hours`` as floats.
        """
        return py_bearing_spall_progression(spall_size, time_hours)

    def lubricant_degradation_viscosity(self, temp_c: float, shear_rate: float) -> float:
        """Wrapper for lubricant degradation viscosity.

        ``temp_c`` temperature in Celsius, ``shear_rate`` shear rate.
        """
        return py_lubricant_degradation_viscosity(temp_c, shear_rate)
