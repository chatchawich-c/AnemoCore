use crate::thermo::{ctd_p6, p4_hotspot_margin};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RlDeratingConfig {
    pub rated_power_mw: f64, pub cut_in_ms: f64, pub rated_wind_ms: f64, pub cut_out_ms: f64,
    pub temp_crit_c: f64, pub temp_ref_k: f64, pub ea_j_mol: f64,
    pub tau_fast_s: f64, pub tau_slow_s: f64, pub thermal_gain_k_per_mw: f64, pub w_ctd: f64, pub w_soak: f64,
}
impl Default for RlDeratingConfig {
    fn default() -> Self {
        Self { rated_power_mw: 5.0, cut_in_ms: 3.0, rated_wind_ms: 11.5, cut_out_ms: 25.0,
               temp_crit_c: 115.0, temp_ref_k: 363.15, ea_j_mol: 80_000.0,
               tau_fast_s: 600.0, tau_slow_s: 7200.0, thermal_gain_k_per_mw: 16.0, w_ctd: 2.0, w_soak: 5.0 }
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RlDeratingState { pub temp_fast_c: f64, pub temp_slow_soak_c: f64, pub ctd_accum: f64 }
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RlDeratingStepOutput {
    pub power_mw: f64, pub temp_fast_c: f64, pub temp_slow_soak_c: f64,
    pub p4_margin_c: f64, pub ctd_increment: f64, pub ctd_accum: f64, pub reward: f64, pub thermal_violation: bool,
}
pub fn step_rl_derating(cfg: &RlDeratingConfig, st: &mut RlDeratingState, ws: f64, amb: f64, act: f64, dt_s: f64) -> RlDeratingStepOutput {
    let dt = dt_s.max(1.0);
    let p_av = if ws < cfg.cut_in_ms || ws > cfg.cut_out_ms { 0.0 }
               else if ws >= cfg.rated_wind_ms { cfg.rated_power_mw }
               else { cfg.rated_power_mw * ((ws - cfg.cut_in_ms) / (cfg.rated_wind_ms - cfg.cut_in_ms).max(1e-6)).powi(3) };
    let p_mw = p_av * act.clamp(0.0, 1.0);
    let t_eq = amb + (cfg.thermal_gain_k_per_mw * p_mw) / (1.0 + 0.03 * ws.clamp(0.0, 25.0));
    st.temp_fast_c += (1.0 - (-dt / cfg.tau_fast_s.max(1.0)).exp()) * (t_eq - st.temp_fast_c);
    st.temp_slow_soak_c += (1.0 - (-dt / cfg.tau_slow_s.max(1.0)).exp()) * (0.75 * st.temp_fast_c + 0.25 * amb - st.temp_slow_soak_c);
    let hot = st.temp_fast_c.max(st.temp_slow_soak_c);
    let p4 = p4_hotspot_margin(hot, cfg.temp_crit_c);
    let ctd_inc = ctd_p6(&[hot + 273.15], cfg.temp_ref_k, cfg.ea_j_mol, dt) / 3600.0;
    st.ctd_accum += ctd_inc;
    let viol = p4 < 0.0;
    let soak = (st.temp_slow_soak_c - (cfg.temp_crit_c - 15.0)).max(0.0);
    let rew = (p_mw / cfg.rated_power_mw.max(1e-6)) - cfg.w_ctd * ctd_inc - cfg.w_soak * (0.05 * soak + if viol { -p4 * 2.0 } else { 0.0 });
    RlDeratingStepOutput { power_mw: p_mw, temp_fast_c: st.temp_fast_c, temp_slow_soak_c: st.temp_slow_soak_c, p4_margin_c: p4, ctd_increment: ctd_inc, ctd_accum: st.ctd_accum, reward: rew, thermal_violation: viol }
}
pub fn evaluate_rl_derating_rollout(cfg: &RlDeratingConfig, ws: &[f64], amb: &[f64], acts: &[f64], dt_s: f64) -> (f64, f64, f64, usize) {
    let n = ws.len().min(amb.len()).min(acts.len());
    if n == 0 { return (0.0, 0.0, 0.0, 0); }
    let mut st = RlDeratingState { temp_fast_c: amb[0] + 15.0, temp_slow_soak_c: amb[0] + 12.0, ctd_accum: 0.0 };
    let (mut rew, mut en, mut v) = (0.0, 0.0, 0);
    for i in 0..n {
        let o = step_rl_derating(cfg, &mut st, ws[i], amb[i], acts[i], dt_s);
        rew += o.reward; en += o.power_mw * (dt_s / 3600.0); if o.thermal_violation { v += 1; }
    }
    (rew, en, st.ctd_accum, v)
}
