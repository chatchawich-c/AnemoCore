from ._core import (
    py_compute_lcoe,
    py_compute_lcoh,
    py_om_unscheduled_cost_model,
    py_coupled_monte_carlo_risk,
    py_var_cvar_project_npv,
)

class WindEconomics:
    """Thin façade exposing economic risk functions.

    All methods forward directly to the compiled Rust bindings.
    """

    @staticmethod
    def compute_lcoe(capex: float, fixed_opex: float, variable_opex: float, discount_rate: float,
                    lifetime_years: float, annual_energy_mwh: float, efficiency_h2: float) -> float:
        """Levelized Cost Of Energy calculation.
        """
        return py_compute_lcoe(capex, fixed_opex, variable_opex, discount_rate,
                               lifetime_years, annual_energy_mwh, efficiency_h2)

    @staticmethod
    def compute_lcoh(capex: float, fixed_opex: float, variable_opex: float, discount_rate: float,
                    lifetime_years: float, annual_energy_mwh: float, efficiency_h2: float) -> float:
        """Levelized Cost Of Hydrogen calculation.
        """
        return py_compute_lcoh(capex, fixed_opex, variable_opex, discount_rate,
                               lifetime_years, annual_energy_mwh, efficiency_h2)

    @staticmethod
    def om_unscheduled_cost_model(failure_rate_per_year: float, repair_duration_hours: float,
                                 mobilization_cost: float, downtime_revenue_loss_per_hour: float) -> float:
        """O&M unscheduled cost model.
        """
        return py_om_unscheduled_cost_model(failure_rate_per_year, repair_duration_hours,
                                            mobilization_cost, downtime_revenue_loss_per_hour)

    @staticmethod
    def coupled_monte_carlo_risk(seed: int, n_samples: int, wind_stddev: float,
                                 drivetrain_failure_rate: float, price_volatility: float) -> list[float]:
        """Monte‑Carlo risk simulation returning a vector of results.
        """
        return py_coupled_monte_carlo_risk(seed, n_samples, wind_stddev,
                                           drivetrain_failure_rate, price_volatility)

    @staticmethod
    def var_cvar_project_npv(samples: list[float], confidence: float) -> tuple[float, float]:
        """Value‑at‑Risk and Conditional‑VaR for a project NPV.
        """
        return py_var_cvar_project_npv(samples, confidence)
