# Firmware Target (Round 2 — Month 03 Hardware-in-the-Loop)

This directory is reserved for the bare-metal `#![no_std]` firmware binary crate targeting
**ARM Cortex-M4F (STM32F401RE Nucleo @ 84 MHz)** (`thumbv7em-none-eabihf`) and **TI C2000** HIL validation:

- `P42` (`kam_vsg`): `O(1)` KAM Golden-Ratio Detuned VSG + 4-stage Foster RC thermal guard
- `P44` (`quasicrystal_wpt`): Zero-FPU integer-only Shift-Add Fibonacci frame filter bank
- `P46` (`fib_qmc`): Array-less rank-1 Fibonacci-Korobov QMC + Golden-Section stopping `S*`

Verify `#![no_std]` ARM compilation of `anemocore` right now with:

```bash
rustup target add thumbv7em-none-eabihf
cargo check -p anemocore --no-default-features --target thumbv7em-none-eabihf
```
