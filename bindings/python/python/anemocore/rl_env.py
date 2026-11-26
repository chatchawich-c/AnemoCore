from __future__ import annotations
import numpy as np
from ._core import py_rl_derating_step

class ThermalDeratingEnv:
    def __init__(self, wind_speeds_ms, ambient_temps_c, dt_s=600.0, rated_power_mw=5.0, temp_crit_c=115.0):
        self.ws = np.asarray(wind_speeds_ms, dtype=np.float64)
        self.amb = np.asarray(ambient_temps_c, dtype=np.float64)
        self.dt_s, self.rated_power_mw, self.temp_crit_c = float(dt_s), float(rated_power_mw), float(temp_crit_c)
        self._i = 0; self._tf = 0.0; self._ts = 0.0; self._ctd = 0.0
    def reset(self, seed=None):
        self._i = 0; a0 = float(self.amb[0]); self._tf = a0 + 15.0; self._ts = a0 + 12.0; self._ctd = 0.0
        return np.array([float(self.ws[0]), a0, self._tf, self._ts, self._ctd], dtype=np.float64), {"step": 0}
    def step(self, action):
        act = float(np.asarray(action).reshape(-1)[0])
        p, tf, ts, p4, ci, ca, rew, viol = py_rl_derating_step(self._tf, self._ts, self._ctd, float(self.ws[self._i]), float(self.amb[self._i]), act, self.dt_s, self.rated_power_mw, self.temp_crit_c)
        self._tf, self._ts, self._ctd = tf, ts, ca; self._i += 1
        done = self._i >= len(self.ws); ni = min(self._i, len(self.ws) - 1)
        obs = np.array([float(self.ws[ni]), float(self.amb[ni]), tf, ts, ca], dtype=np.float64)
        return obs, float(rew), done, False, {"power_mw": p, "p4_margin_c": p4, "ctd_increment": ci, "ctd_accum": ca, "thermal_violation": viol}
