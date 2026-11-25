use anemocore::fib_liquidity::{lob_density, roll_effective_spread, FibonacciWallConfig};
use anemocore::fib_qmc::{
    estimate_failure_prob_qmc_o1, evaluate_limit_state_6d, golden_section_stopping_search,
};
use anemocore::kam_vsg::{step_kam_vsg, BiquadNotchFilter, ControlMode, KamVsgSpec, KamVsgState};
use anemocore::phyllotaxis_wake::{
    compute_ctd_hours, golden_yaw_staggering, step_aerothermal_wake, structure_factor_2d,
    vogel_phyllotaxis_layout, AeroThermalSpec,
};
use anemocore::quasicrystal_wpt::{
    shift_add_forward_step, shift_add_inverse_step, shift_add_wpt_3level_band_energy,
};
use anemocore::signal_wpt::{extract_wpt_indicators_vec, HertzianBearingSpec};

#[test]
fn round1_wave1_end_to_end_pipeline_parity() {
    // 1. P42 + P19: KAM-VSG step & Biquad notch filter stability
    let vsg_spec = KamVsgSpec::p42_default();
    let mut vsg_state = KamVsgState::zero();
    let mut notch = BiquadNotchFilter::new(40.75, 0.05, 0.707, 0.001);
    for _ in 0..50 {
        let out = step_kam_vsg(
            &vsg_spec,
            &mut vsg_state,
            ControlMode::KamGoldenRatioVsg,
            -0.05,
            -0.02,
            0.0,
            0.001,
        );
        let filtered = notch.step(out.p_wt_mw);
        assert!(filtered.is_finite());
        assert!(out.t_junction_c > vsg_spec.t_coolant_c);
    }

    // 2. P44: Zero-FPU integer shift-add wavelet lifting lossless reconstruction + 8-band WPT
    let mut frame = [0i32; 32];
    for (i, v) in frame.iter_mut().enumerate() {
        *v = ((i as i32 * 37) % 500) - 250;
    }
    let mut lo = [0i32; 16];
    let mut hi = [0i32; 16];
    let mut recon = [0i32; 32];
    shift_add_forward_step(&frame, &mut lo, &mut hi);
    shift_add_inverse_step(&lo, &hi, &mut recon);
    assert_eq!(frame, recon);
    let mut sa = [0i32; 32];
    let mut sb = [0i32; 32];
    let mut energies = [0u64; 8];
    shift_add_wpt_3level_band_energy(&frame, &mut sa, &mut sb, &mut energies);
    assert!(energies.iter().sum::<u64>() > 0);

    // 3. P46: Rank-1 6D QMC sweep -> Golden-Section stopping S*(mean_dtf)
    let (_pf, mean_dtf) = estimate_failure_prob_qmc_o1(&[0.11, 0.23, 0.37, 0.53, 0.71, 0.89]);
    let stop = golden_section_stopping_search(mean_dtf, 5.0, 160.0, 1e-5, 35);
    assert!(stop.s_star.is_finite() && stop.final_width < 1e-5);
    let sample_eval = evaluate_limit_state_6d(&[0.5; 6]);
    assert!(sample_eval.dtf > 0.0 && sample_eval.dtf <= 1.0);

    // 4. P33 + P10: WPT indicators & Hertzian mechanics under heavy load
    let vib: Vec<f64> = (0..128).map(|i| (0.25 * (i as f64)).sin()).collect();
    let wpt = extract_wpt_indicators_vec(&vib).unwrap();
    assert!(wpt.h3_renyi >= 0.0 && wpt.p_high >= 0.0);
    let (_kh, fc_heavy, _tau, scale_heavy) =
        HertzianBearingSpec::p10_default().instantaneous_mechanics(30_000.0, 1.0);
    assert!(fc_heavy > 11_779.0 && scale_heavy < 1.0);

    // 5. P43 + P08: Fibonacci LOB wall density + Roll spread
    let lob_cfg = FibonacciWallConfig::p43_default();
    assert!(lob_density(0.618, &lob_cfg) > lob_density(0.125, &lob_cfg));
    let spread = roll_effective_spread(&[62.0, 65.0, 61.0, 66.0, 60.0, 67.0, 59.0, 68.0], 8.0);
    assert!(spread[7] > 0.0);

    // 6. P45: Phyllotaxis layout + Structure Factor + Aero-Thermal Wake + Golden Yaw + CTD
    let aero_spec = AeroThermalSpec::p45_default();
    let (vx, vy) = vogel_phyllotaxis_layout(25, 4.5, aero_spec.rotor_diam_m);
    let sq = structure_factor_2d(&vx, &vy, &[[0.015, 0.0], [0.0, 0.015]]);
    assert_eq!(sq.len(), 2);
    let yaw = golden_yaw_staggering(&vx, &vy, aero_spec.rotor_diam_m, 250.0, 16.0);
    let mut t_oil = vec![288.15 + 45.0; 25];
    let step = step_aerothermal_wake(
        &vx, &vy, &aero_spec, 9.5, 250.0, 288.15, &yaw, &mut t_oil, 600.0,
    );
    assert_eq!(step.t_oil_k.len(), 25);
    assert!(compute_ctd_hours(&step.t_oil_k, &aero_spec, 600.0) > 0.0);
}
