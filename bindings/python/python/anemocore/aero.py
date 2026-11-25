from ._core import (
    py_prandtl_tip_hub_loss,
    py_bem_steady_rotor,
    py_beddoes_leishman_stall,
    py_aero_torque_thrust_curve,
)

class AeroRotor:
    """Thin façade exposing aero‑related Rust functions.

    The methods directly forward arguments to the underlying Rust
    implementation via the compiled ``_core`` module.
    """

    @staticmethod
    def prandtl_tip_hub_loss(u_tip: float, radius: float, hub_radius: float) -> float:
        """Calculate tip‑hub loss using the Prandtl‑Tip correction.

        Parameters are forwarded to ``py_prandtl_tip_hub_loss``.
        """
        return py_prandtl_tip_hub_loss(u_tip, radius, hub_radius)

    @staticmethod
    def bem_steady_rotor(blade_count: int, u_tip: float, radius: float) -> float:
        """Calculate the steady‑state BEM power for a rotor.
        """
        return py_bem_steady_rotor(blade_count, u_tip, radius)

    @staticmethod
    def beddoes_leishman_stall(alpha: float, beta: float) -> float:
        """Beddoes‑Leishman aerodynamic stall model.
        """
        return py_beddoes_leishman_stall(alpha, beta)

    @staticmethod
    def aero_torque_thrust_curve(u: float, theta: float) -> tuple[float, float]:
        """Return (torque, thrust) for a given wind speed and blade pitch.
        """
        return py_aero_torque_thrust_curve(u, theta)
