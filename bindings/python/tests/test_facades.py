import sys
import importlib
import inspect
import anemocore

def test_lazy_import():
    import importlib
    import sys
    import anemocore as ac
    assert "matplotlib" not in sys.modules

def _assert_method(cls, name, expected_params):
    method = getattr(cls, name, None)
    assert method is not None, f"{name} not found on {cls.__name__}"
    sig = inspect.signature(method)
    # subtract 'self' for instance methods
    params = list(sig.parameters.values())
    if params and params[0].name == 'self':
        params = params[1:]
    assert len(params) == expected_params, f"{name} expects {expected_params} params, got {len(params)}"

def test_facade_methods():
    # AeroRotor methods
    _assert_method(anemocore.AeroRotor, 'prandtl_tip_hub_loss', 3)
    _assert_method(anemocore.AeroRotor, 'bem_steady_rotor', 3)
    _assert_method(anemocore.AeroRotor, 'beddoes_leishman_stall', 2)
    _assert_method(anemocore.AeroRotor, 'aero_torque_thrust_curve', 2)
    # Drivetrain methods (including new tribology wrappers)
    _assert_method(anemocore.Drivetrain, 'mixed_ehl_asperity_load', 3)
    _assert_method(anemocore.Drivetrain, 'micropitting_fatigue_index', 2)
    _assert_method(anemocore.Drivetrain, 'bearing_spall_progression', 2)
    _assert_method(anemocore.Drivetrain, 'lubricant_degradation_viscosity', 2)
    # FleetSCADA and WindEconomics existence (no method checks here)
    assert hasattr(anemocore, 'FleetSCADA'), "FleetSCADA class missing"
    assert hasattr(anemocore, 'WindEconomics'), "WindEconomics class missing"
