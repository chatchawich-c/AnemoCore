# End-to-end R parity tests against `ecowind-core` compiled shared library.

find_ecowindr_lib <- function() {
  candidates <- c(
    "target/release/libecowindr.dylib",
    "target/release/libecowindr.so",
    "../../target/release/libecowindr.dylib",
    "../../target/release/libecowindr.so"
  )
  for (p in candidates) {
    if (file.exists(p)) return(normalizePath(p))
  }
  stop("Compiled `ecowindr` shared library not found. Run `cargo build --release -p ecowindr` first.")
}

lib_path <- find_ecowindr_lib()
dyn.load(lib_path)

assert_close <- function(actual, expected, tol = 1e-6, label = "value") {
  if (any(is.na(actual)) || max(abs(actual - expected)) > tol) {
    stop(sprintf(
      "Assertion failed for %s: actual=%s expected=%s",
      label,
      paste(actual, collapse = ","),
      paste(expected, collapse = ",")
    ))
  }
}

# 1. Paper 33 DB4 WPT exact parity
t_grid <- seq(0, 63) / 64.0
sig_p33 <- sin(8.0 * pi * t_grid) + 0.3 * cos(24.0 * pi * t_grid)
wpt <- .Call("wrap__r_wpt_db4_indicators", sig_p33)
assert_close(wpt[1], 0.2966570345, tol = 1e-8, label = "wpt$p_low_mid")
assert_close(wpt[2], 0.0007641162, tol = 1e-8, label = "wpt$p_high")
assert_close(wpt[4], 0.7740334671, tol = 1e-8, label = "wpt$h3_renyi")
stopifnot(wpt[3] > 0.0)

# 2. Paper 46 Golden-Section stopping & 6D Limit State
stop_res <- .Call("wrap__r_fib_golden_stopping", 0.35)
assert_close(stop_res[1], 73.3189833311, tol = 1e-6, label = "fib_golden_stopping$s_star")
assert_close(stop_res[2], 36.0, tol = 1e-9, label = "fib_golden_stopping$iterations")

lim <- .Call("wrap__r_fib_limit_state_6d", c(0.1, 0.8, 0.3, 0.6, 0.9, 0.2))
assert_close(
  lim,
  c(68.6107392823, 30.7512613229, 25.0594779594, 0.8588422274),
  tol = 1e-6,
  label = "fib_limit_state_6d"
)

# 3. Paper 42 KAM Golden-Ratio VSG step (freq_ratio == phi^-1)
kam <- .Call("wrap__r_kam_vsg_step", 0.005, -0.02, -0.01, 0.001)
phi_inv <- (sqrt(5.0) - 1.0) / 2.0
stopifnot(is.finite(kam[1]), kam[1] > 0.0)
assert_close(kam[2], phi_inv, tol = 1e-9, label = "kam_vsg$freq_ratio")
stopifnot(is.finite(kam[3]), kam[3] > 45.0)

# 4. Paper 43 + Paper 08 Market Microstructure
rho <- .Call("wrap__r_lob_density", 0.618)
stopifnot(is.finite(rho), rho > 0.0)

set.seed(42)
prices <- 65.0 + cumsum(rnorm(120, mean = 0.0, sd = 0.6))
spread <- .Call("wrap__r_market_roll_spread", prices, 20L)
stopifnot(length(spread) == 120L, all(is.finite(spread)), all(spread >= 0.0))

cat("ALL 4 R BINDING PARITY GROUPS PASSED cleanly via:", lib_path, "\n")
