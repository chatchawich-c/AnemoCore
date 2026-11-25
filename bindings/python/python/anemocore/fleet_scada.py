from ._core import (
    py_sarimax_fit,
    py_sarimax_forecast_residuals,
    py_fleet_cross_turbine_residual,
    py_multi_domain_health_index,
    py_scada_missingness_diagnostic,
)

class FleetSCADA:
    """Thin façade exposing fleet‑SCADA functions.

    Each static method forwards arguments to the corresponding Rust
    implementation.
    """

    @staticmethod
    def sarimax_fit(data: list[float], order: tuple[int, int, int]) -> dict:
        """Fit a SARIMAX model to ``data``.

        ``order`` is ``(p, d, q)``.
        """
        return py_sarimax_fit(data, order)

    @staticmethod
    def sarimax_forecast_residuals(model: dict, steps: int) -> list[float]:
        """Forecast residuals using a fitted SARIMAX ``model``.
        """
        return py_sarimax_forecast_residuals(model, steps)

    @staticmethod
    def fleet_cross_turbine_residual(matrix: list[list[float]]) -> list[float]:
        """Compute cross‑turbine residuals for a fleet.
        """
        return py_fleet_cross_turbine_residual(matrix)

    @staticmethod
    def multi_domain_health_index(data: list[float]) -> float:
        """Calculate a health index across multiple domains.
        """
        return py_multi_domain_health_index(data)

    @staticmethod
    def scada_missingness_diagnostic(df: any) -> dict:
        """Diagnose missing data patterns in a SCADA dataframe.
        """
        return py_scada_missingness_diagnostic(df)
