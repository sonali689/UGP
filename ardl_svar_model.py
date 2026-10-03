"""
Phase 4: ARDL Model & Structural VAR
=====================================
Produces:
  - ARDL bounds test for cointegration
  - Long-run and short-run coefficient estimates
  - Error correction model
  - VAR/SVAR with IRFs and FEVD
  - Granger causality tests
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.ardl import UECM, ardl_select_order
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'merged_monthly.csv'),
                     index_col='date', parse_dates=True)
    return df

# ═══════════════════════════════════════════════════════════════
# PART A: ARDL MODEL
# ═══════════════════════════════════════════════════════════════

def run_ardl_model(df, dep_var='LOG_NEER', label='NEER'):
    """Estimate UECM (Unrestricted ECM) and run the Pesaran-Shin-Smith bounds test.

    statsmodels implements bounds_test() on the UECM results class, not on
    the plain ARDL results class. Using UECM directly ensures the bounds test
    actually executes and the output file is written.
    """
    print(f"\n  --- UECM / ARDL Bounds Test: {label} ---")

    df_sample = df.loc['2001-04':'2026-06'].copy()

    # Independent variables
    exog_cols = ['LOG_OIL', 'CPI_YOY', 'LOG_EPU', 'INT_DIFF', 'LOG_RESERVES']

    # Prepare data
    y = df_sample[dep_var].dropna()
    X = df_sample[exog_cols].dropna()

    # Align
    common_idx = y.index.intersection(X.index)
    y = y.loc[common_idx]
    X = X.loc[common_idx]

    # Drop any remaining NaN
    mask = y.notna() & X.notna().all(axis=1)
    y = y[mask]
    X = X[mask]

    print(f"  Sample: {y.index.min().strftime('%Y-%m')} to {y.index.max().strftime('%Y-%m')} ({len(y)} obs)")
    print(f"  Dependent: {dep_var}")
    print(f"  Independent: {', '.join(exog_cols)}")

    # Select optimal lag order via AIC (used for both ARDL and UECM)
    print("\n  Selecting optimal lag order...")
    try:
        sel = ardl_select_order(y, maxlag=6, exog=X, maxorder=4, ic='aic', trend='c')
        ar_order = max(sel.ar_lags) if sel.ar_lags else 1
        dl_order = {col: max(max(sel.dl_lags.get(col, [1])), 1) for col in exog_cols}
        print(f"  AR order: {ar_order}, DL orders: {dl_order}")
    except Exception as e:
        print(f"  Auto-selection failed ({e}), using default UECM(2,1,1,1,1,1)")
        ar_order = 2
        dl_order = {col: 1 for col in exog_cols}

    # -- Estimate UECM (needed for bounds_test()) --
    # UECM is the Unrestricted Error Correction form of the ARDL model.
    # It gives identical coefficient estimates to ARDL but exposes the
    # bounds_test() method required for Pesaran, Shin & Smith (2001).
    try:
        uecm_model  = UECM(y, lags=ar_order, exog=X, order=dl_order, trend='c')
        uecm_result = uecm_model.fit()

        print("\n  === UECM Estimation Results ===")
        print(uecm_result.summary().as_text())

        # Save coefficient table
        with open(os.path.join(OUTPUT_DIR, f'ardl_results_{label}.txt'), 'w', encoding='utf-8') as f:
            f.write(uecm_result.summary().as_text())

        # -- ARDL Bounds Test (Pesaran, Shin & Smith, 2001) --
        # Case III: unrestricted constant, no trend (most common for macro levels)
        print("\n  === ARDL Bounds Test (Pesaran et al., 2001) ===")
        bounds = uecm_result.bounds_test(case=3)
        print(f"  F-statistic: {bounds.stat:.4f}")
        print(f"  {'Significance':<15s} {'I(0) Bound':>12s} {'I(1) Bound':>12s} {'Decision':>15s}")
        print("  " + "-" * 60)
        cv = bounds.crit_vals
        for row in cv.itertuples():
            sig = f"{100 - row.Index:.1f}%"
            i0  = row.lower
            i1  = row.upper
            decision = ('Cointegration'  if bounds.stat > i1 else
                        'Inconclusive'   if bounds.stat > i0 else
                        'No cointegration')
            print(f"  {sig:<15s} {i0:>12.4f} {i1:>12.4f} {decision:>15s}")

        # Save bounds test output
        with open(os.path.join(OUTPUT_DIR, f'bounds_test_{label}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"ARDL Bounds Test Results - {label}\n")
            f.write(f"F-statistic: {bounds.stat:.4f}\n")
            f.write(f"P-values: lower={bounds.p_values[0]:.4f}, upper={bounds.p_values[1]:.4f}\n\n")
            f.write(bounds.crit_vals.to_string())
        print(f"  Saved: bounds_test_{label}.txt")

        # ── Diagnostics ──
        print("\n  === Diagnostic Tests ===")
        residuals = uecm_result.resid

        # Breusch-Godfrey serial correlation
        try:
            bg_stat, bg_pval, _, _ = acorr_breusch_godfrey(uecm_result, nlags=4)
            print(f"  Breusch-Godfrey (serial correlation): stat={bg_stat:.4f}, p={bg_pval:.4f} "
                  f"{'[PASS: No serial correlation]' if bg_pval > 0.05 else '[FAIL: Serial correlation detected]'}")
        except Exception as e:
            print(f"  Breusch-Godfrey: Error - {e}")

        # Jarque-Bera normality
        jb_stat, jb_pval, _, _ = jarque_bera(residuals)
        print(f"  Jarque-Bera (normality): stat={jb_stat:.4f}, p={jb_pval:.4f} "
              f"{'[PASS: Normal]' if jb_pval > 0.05 else '[FAIL: Non-normal (common in macro)]'}")

        if hasattr(uecm_result, 'rsquared'):
            print(f"  R-squared:     {uecm_result.rsquared:.4f}")
        if hasattr(uecm_result, 'rsquared_adj'):
            print(f"  Adj R-squared: {uecm_result.rsquared_adj:.4f}")
        print(f"  AIC:           {uecm_result.aic:.4f}")
        print(f"  BIC:           {uecm_result.bic:.4f}")
        print(f"  Log-Likelihood:{uecm_result.llf:.4f}")

        return uecm_result

    except Exception as e:
        print(f"  UECM estimation error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════
# PART B: VECTOR AUTOREGRESSION (VAR / SVAR)
# ═══════════════════════════════════════════════════════════════

def run_var_model(df):
    """Estimate VAR, compute IRFs and FEVD."""
    print("\n" + "=" * 70)
    print("PART B: VECTOR AUTOREGRESSION (VAR)")
    print("=" * 70)
    
    df_sample = df.loc['2001-04':'2026-06'].copy()
    
    # VAR variables (all in first differences / stationary form)
    var_cols = ['D_LOG_OIL', 'D_CPI', 'D_INT_DIFF', 'D_LOG_RESERVES', 'D_LOG_NEER']
    var_labels = ['Oil Returns', 'Delta CPI', 'Delta Int. Diff', 'Delta Reserves', 'NEER Returns']
    
    data = df_sample[var_cols].dropna()
    print(f"\n  Sample: {data.index.min().strftime('%Y-%m')} to {data.index.max().strftime('%Y-%m')} ({len(data)} obs)")
    print(f"  Variables: {', '.join(var_cols)}")
    
    # ── Lag Selection ──
    print("\n  --- VAR Lag Selection ---")
    model = VAR(data)
    lag_results = model.select_order(maxlags=12)
    print(lag_results.summary())
    
    # Use AIC-selected lag
    aic_lag = lag_results.aic
    bic_lag = lag_results.bic
    print(f"\n  AIC optimal lag: {aic_lag}")
    print(f"  BIC optimal lag: {bic_lag}")
    
    # Use AIC lag (but at least 2)
    selected_lag = max(aic_lag, 2)
    print(f"  Selected lag: {selected_lag}")
    
    # ── Estimate VAR ──
    print(f"\n  --- Estimating VAR({selected_lag}) ---")
    var_result = model.fit(selected_lag)
    print(var_result.summary())
    
    # Save VAR summary
    with open(os.path.join(OUTPUT_DIR, 'var_results.txt'), 'w', encoding='utf-8') as f:
        f.write(str(var_result.summary()))
    
    # ── Impulse Response Functions ──
    print("\n  --- Impulse Response Functions ---")
    irf = var_result.irf(periods=24)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Impulse Response Functions: Response of NEER Returns to Shocks\n'
                 f'(VAR({selected_lag}), Cholesky Ordering)', fontsize=14, fontweight='bold')
    
    neer_idx = var_cols.index('D_LOG_NEER')
    
    for i, (col, label) in enumerate(zip(var_cols, var_labels)):
        if i >= 5:
            break
        ax = axes[i // 3, i % 3]
        shock_idx = var_cols.index(col)
        
        # IRF values
        irf_vals = irf.irfs[:, neer_idx, shock_idx]
        # Use stderr-based CI (irf_err_bands_mc may not be available)
        try:
            lower, upper = irf.err_band_mc()  # Monte Carlo error bands
            lower_vals = lower[:, neer_idx, shock_idx]
            upper_vals = upper[:, neer_idx, shock_idx]
        except Exception:
            # Fallback: no confidence bands
            lower_vals = irf_vals
            upper_vals = irf_vals
        
        periods = range(len(irf_vals))
        ax.plot(periods, irf_vals, 'b-', linewidth=2)
        if not np.array_equal(lower_vals, upper_vals):
            ax.fill_between(periods, lower_vals, upper_vals, alpha=0.15, color='blue')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(f'Shock: {label}', fontweight='bold')
        ax.set_xlabel('Months')
        ax.set_ylabel('Response')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(var_cols) < 6:
        axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'irf_neer.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also plot response of USD/INR
    # (Run separate VAR with USD/INR)
    var_cols_usd = ['D_LOG_OIL', 'D_CPI', 'D_INT_DIFF', 'D_LOG_RESERVES', 'D_LOG_USDINR']
    var_labels_usd = ['Oil Returns', 'Delta CPI', 'Delta Int. Diff', 'Delta Reserves', 'USD/INR Returns']
    
    data_usd = df_sample[var_cols_usd].dropna()
    model_usd = VAR(data_usd)
    var_result_usd = model_usd.fit(selected_lag)
    irf_usd = var_result_usd.irf(periods=24)
    
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('Impulse Response Functions: Response of USD/INR Returns to Shocks\n'
                  f'(VAR({selected_lag}), Cholesky Ordering)', fontsize=14, fontweight='bold')
    
    usdinr_idx = var_cols_usd.index('D_LOG_USDINR')
    
    for i, (col, label) in enumerate(zip(var_cols_usd, var_labels_usd)):
        if i >= 5:
            break
        ax = axes2[i // 3, i % 3]
        shock_idx = var_cols_usd.index(col)
        
        irf_vals = irf_usd.irfs[:, usdinr_idx, shock_idx]
        try:
            lower, upper = irf_usd.err_band_mc()
            lower_vals = lower[:, usdinr_idx, shock_idx]
            upper_vals = upper[:, usdinr_idx, shock_idx]
        except Exception:
            lower_vals = irf_vals
            upper_vals = irf_vals
        
        periods = range(len(irf_vals))
        ax.plot(periods, irf_vals, 'r-', linewidth=2)
        if not np.array_equal(lower_vals, upper_vals):
            ax.fill_between(periods, lower_vals, upper_vals, alpha=0.15, color='red')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(f'Shock: {label}', fontweight='bold')
        ax.set_xlabel('Months')
        ax.set_ylabel('Response')
        ax.grid(True, alpha=0.3)
    
    if len(var_cols_usd) < 6:
        axes2[1, 2].set_visible(False)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'irf_usdinr.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  Saved: irf_neer.png, irf_usdinr.png")
    
    # ── Forecast Error Variance Decomposition ──
    print("\n  --- Forecast Error Variance Decomposition (FEVD) ---")
    fevd = var_result.fevd(periods=24)
    # Get FEVD data - fevd.decomp shape is (n_vars, periods, n_vars)
    # where decomp[response_var, horizon, shock_var]
    fevd_data = fevd.decomp
    print(f"  FEVD decomp shape: {fevd_data.shape}")
    
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
    
    # NEER FEVD - decomp[neer_idx, :, :] gives (periods, n_shocks)
    ax = axes3[0]
    fevd_neer = fevd_data[neer_idx, :, :]  # (24, 5)
    n_periods = fevd_neer.shape[0]
    
    colors = ['#f39c12', '#e67e22', '#2980b9', '#27ae60', '#2c3e50']
    bottom = np.zeros(n_periods)
    for j, (label, color) in enumerate(zip(var_labels, colors)):
        vals = fevd_neer[:, j]
        ax.bar(range(1, n_periods + 1), vals, bottom=bottom, label=label, color=color, alpha=0.8)
        bottom += vals
    ax.set_title('FEVD: NEER Returns', fontweight='bold')
    ax.set_xlabel('Horizon (months)')
    ax.set_ylabel('Proportion')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)
    
    # USDINR FEVD
    fevd_usd = var_result_usd.fevd(periods=24)
    ax = axes3[1]
    fevd_usd_data = fevd_usd.decomp
    fevd_usdinr = fevd_usd_data[usdinr_idx, :, :]  # (24, 5)
    
    bottom = np.zeros(n_periods)
    for j, (label, color) in enumerate(zip(var_labels_usd, colors)):
        vals = fevd_usdinr[:, j]
        ax.bar(range(1, n_periods + 1), vals, bottom=bottom, label=label, color=color, alpha=0.8)
        bottom += vals
    ax.set_title('FEVD: USD/INR Returns', fontweight='bold')
    ax.set_xlabel('Horizon (months)')
    ax.set_ylabel('Proportion')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, 'fevd.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print FEVD table at key horizons
    print("\n  FEVD for NEER Returns (% explained by each shock):")
    print(f"  {'Horizon':<10s}", end='')
    for label in var_labels:
        print(f"  {label:<18s}", end='')
    print()
    for h in [1, 3, 6, 12, 24]:
        if h <= n_periods:
            print(f"  {h:<10d}", end='')
            for j in range(len(var_labels)):
                print(f"  {fevd_neer[h-1, j]*100:>16.2f}%", end='')
            print()
    
    print("\n  FEVD for USD/INR Returns:")
    print(f"  {'Horizon':<10s}", end='')
    for label in var_labels_usd:
        print(f"  {label:<18s}", end='')
    print()
    for h in [1, 3, 6, 12, 24]:
        if h <= n_periods:
            print(f"  {h:<10d}", end='')
            for j in range(len(var_labels_usd)):
                print(f"  {fevd_usdinr[h-1, j]*100:>16.2f}%", end='')
            print()
    
    print("\n  Saved: fevd.png")
    
    return var_result, var_result_usd


# ═══════════════════════════════════════════════════════════════
# PART C: GRANGER CAUSALITY TESTS
# ═══════════════════════════════════════════════════════════════

def run_granger_causality(df, fixed_lag=None):
    """Test Granger causality between key variable pairs.

    Parameters
    ----------
    fixed_lag : int or None
        The lag to use for every Granger test. When None, the function
        determines the lag from VAR AIC selection.  Using a single
        pre-specified lag avoids the multiple-testing inflation that
        results from reporting the best p-value across several lags.
    """
    print("\n" + "=" * 70)
    print("PART C: GRANGER CAUSALITY TESTS")
    print("=" * 70)

    df_sample = df.loc['2001-04':'2026-06'].copy()

    # ── Determine lag from VAR AIC (if not supplied) ──
    if fixed_lag is None:
        var_cols_ref = ['D_LOG_OIL', 'D_CPI', 'D_INT_DIFF', 'D_LOG_RESERVES', 'D_LOG_NEER']
        ref_data = df_sample[var_cols_ref].dropna()
        aic_lag = VAR(ref_data).select_order(maxlags=12).aic
        fixed_lag = max(aic_lag, 1)   # at least 1
        print(f"\n  Lag fixed at {fixed_lag} (VAR AIC-optimal, used for ALL pairs).")
        print(f"  Using a single pre-specified lag avoids multiple-testing inflation")
        print(f"  that arises from reporting the minimum p-value across several lags.")
    else:
        print(f"\n  Lag fixed at {fixed_lag} (pre-specified).")

    # Pairs to test (cause, effect)
    pairs = [
        ('D_LOG_OIL',       'D_LOG_NEER',     'Oil Returns -> NEER Returns'),
        ('D_LOG_OIL',       'D_LOG_USDINR',   'Oil Returns -> USD/INR Returns'),
        ('D_LOG_EPU',       'D_LOG_NEER',     'EPU Changes -> NEER Returns'),
        ('D_LOG_EPU',       'D_LOG_USDINR',   'EPU Changes -> USD/INR Returns'),
        ('D_CPI',           'D_LOG_NEER',     'CPI Changes -> NEER Returns'),
        ('D_INT_DIFF',      'D_LOG_NEER',     'Int. Diff Changes -> NEER Returns'),
        ('D_LOG_RESERVES',  'D_LOG_USDINR',   'Reserve Changes -> USD/INR Returns'),
        ('D_LOG_USDINR',    'D_LOG_RESERVES', 'USD/INR Returns -> Reserve Changes'),
        ('D_LOG_OIL',       'D_CPI',          'Oil Returns -> CPI Changes'),
        ('D_LOG_OIL',       'D_INT_DIFF',     'Oil Returns -> Int. Diff Changes'),
    ]

    results = []

    print(f"\n  {'Hypothesis':<45s} {'Lag':>4s} {'F-stat':>10s} {'p-value':>10s} {'Conclusion':>20s}")
    print("  " + "-" * 95)

    for cause, effect, label in pairs:
        data = df_sample[[effect, cause]].dropna()
        if len(data) < 30:
            print(f"  {label:<45s} {'--':>4s} {'--':>10s} {'--':>10s} {'Insufficient data':>20s}")
            continue

        try:
            gc_result = grangercausalitytests(data, maxlag=fixed_lag, verbose=False)

            # Use only the pre-specified lag (no search across lags)
            f_stat = gc_result[fixed_lag][0]['ssr_ftest'][0]
            p_val  = gc_result[fixed_lag][0]['ssr_ftest'][1]

            conclusion = 'Granger-causes' if p_val < 0.05 else 'No causality'
            print(f"  {label:<45s} {fixed_lag:>4d} {f_stat:>10.4f} {p_val:>10.4f} {conclusion:>20s}")

            results.append({
                'Hypothesis':  label,
                'Lag':         fixed_lag,
                'F_statistic': f_stat,
                'p_value':     p_val,
                'Conclusion':  conclusion
            })

        except Exception as e:
            print(f"  {label:<45s}  Error: {e}")

    # Save
    gc_df = pd.DataFrame(results)
    gc_df.to_csv(os.path.join(OUTPUT_DIR, 'granger_causality.csv'), index=False, float_format='%.4f')
    print("\n  Saved: granger_causality.csv")

    return gc_df


def main():
    print("=" * 70)
    print("PHASE 4: ARDL MODEL & STRUCTURAL VAR")
    print("=" * 70)
    
    df = load_data()
    
    # PART A: ARDL
    print("\n" + "=" * 70)
    print("PART A: ARDL BOUNDS TEST & ESTIMATION")
    print("=" * 70)
    
    # Model 1: NEER as dependent
    ardl_neer = run_ardl_model(df, dep_var='LOG_NEER', label='NEER')
    
    # Model 2: USD/INR as dependent
    ardl_usdinr = run_ardl_model(df, dep_var='LOG_USDINR', label='USDINR')
    
    # PART B: VAR
    var_neer, var_usdinr = run_var_model(df)

    # PART C: Granger Causality
    # Pass the VAR AIC lag so all Granger tests use the same pre-specified lag
    aic_lag = VAR(df.loc['2001-04':'2026-06',
                         ['D_LOG_OIL', 'D_CPI', 'D_INT_DIFF',
                          'D_LOG_RESERVES', 'D_LOG_NEER']].dropna()
                  ).select_order(maxlags=12).aic
    gc_results = run_granger_causality(df, fixed_lag=max(aic_lag, 1))
    
    print("\n" + "=" * 70)
    print("Phase 4 COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
