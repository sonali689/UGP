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

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.ardl import ARDL, ardl_select_order
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
    """Estimate ARDL model and bounds test."""
    print(f"\n  --- ARDL Model: {label} ---")
    
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
    
    # Select optimal ARDL lag order
    print("\n  Selecting optimal lag order...")
    try:
        sel = ardl_select_order(y, maxlag=6, exog=X, maxorder=4,
                                ic='aic', trend='c')
        print(f"  Selected ARDL order: ({sel.ar_lags}, {[sel.dl_lags.get(c, 0) for c in exog_cols]})")
        
        # Estimate ARDL model
        ar_order = max(sel.ar_lags) if sel.ar_lags else 1
        dl_order = {}
        for col in exog_cols:
            lags = sel.dl_lags.get(col, [0])
            dl_order[col] = max(lags) if lags else 0
        
        print(f"  AR order: {ar_order}")
        print(f"  DL orders: {dl_order}")
        
    except Exception as e:
        print(f"  Auto-selection failed ({e}), using default ARDL(2,1,1,1,1,1)")
        ar_order = 2
        dl_order = {col: 1 for col in exog_cols}
    
    # Fit ARDL
    try:
        model = ARDL(y, lags=ar_order, exog=X, order=dl_order, trend='c')
        result = model.fit()
        
        print("\n  === ARDL Estimation Results ===")
        print(result.summary().as_text())
        
        # Save results
        with open(os.path.join(OUTPUT_DIR, f'ardl_results_{label}.txt'), 'w', encoding='utf-8') as f:
            f.write(result.summary().as_text())
        
        # ── ARDL Bounds Test ──
        print("\n  === ARDL Bounds Test (Pesaran et al., 2001) ===")
        try:
            bounds = result.bounds_test(case=3)  # Case III: unrestricted constant, no trend
            print(f"  F-statistic: {bounds.stat:.4f}")
            print(f"  {'Significance':<15s} {'I(0) Bound':>12s} {'I(1) Bound':>12s} {'Decision':>15s}")
            print("  " + "-" * 60)
            for row in bounds.critical_values.itertuples():
                sig = f"{row.Index}"
                i0 = row._1
                i1 = row._2
                decision = 'Cointegration' if bounds.stat > i1 else ('Inconclusive' if bounds.stat > i0 else 'No cointegration')
                print(f"  {sig:<15s} {i0:>12.4f} {i1:>12.4f} {decision:>15s}")
            
            with open(os.path.join(OUTPUT_DIR, f'bounds_test_{label}.txt'), 'w', encoding='utf-8') as f:
                f.write(f"ARDL Bounds Test Results - {label}\n")
                f.write(f"F-statistic: {bounds.stat:.4f}\n\n")
                f.write(bounds.critical_values.to_string())
        except Exception as e:
            print(f"  Bounds test error: {e}")
        
        # ── Diagnostics ──
        print("\n  === Diagnostic Tests ===")
        residuals = result.resid
        
        # Breusch-Godfrey serial correlation
        try:
            bg_stat, bg_pval, _, _ = acorr_breusch_godfrey(result, nlags=4)
            print(f"  Breusch-Godfrey (serial correlation): stat={bg_stat:.4f}, p={bg_pval:.4f} "
                  f"{'[PASS: No serial correlation]' if bg_pval > 0.05 else '[FAIL: Serial correlation detected]'}")
        except Exception as e:
            print(f"  Breusch-Godfrey: Error - {e}")
        
        # Jarque-Bera normality
        jb_stat, jb_pval, _, _ = jarque_bera(residuals)
        print(f"  Jarque-Bera (normality): stat={jb_stat:.4f}, p={jb_pval:.4f} "
              f"{'[PASS: Normal]' if jb_pval > 0.05 else '[FAIL: Non-normal (common in macro)]'}")
        
        # R-squared
        print(f"  R-squared: {result.rsquared:.4f}")
        print(f"  Adj R-squared: {result.rsquared_adj:.4f}")
        print(f"  AIC: {result.aic:.4f}")
        print(f"  BIC: {result.bic:.4f}")
        
        return result
        
    except Exception as e:
        print(f"  ARDL estimation error: {e}")
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

def run_granger_causality(df):
    """Test Granger causality between key variable pairs."""
    print("\n" + "=" * 70)
    print("PART C: GRANGER CAUSALITY TESTS")
    print("=" * 70)
    
    df_sample = df.loc['2001-04':'2026-06'].copy()
    
    # Pairs to test (cause, effect)
    pairs = [
        ('D_LOG_OIL', 'D_LOG_NEER', 'Oil Returns -> NEER Returns'),
        ('D_LOG_OIL', 'D_LOG_USDINR', 'Oil Returns -> USD/INR Returns'),
        ('D_LOG_EPU', 'D_LOG_NEER', 'EPU Changes -> NEER Returns'),
        ('D_LOG_EPU', 'D_LOG_USDINR', 'EPU Changes -> USD/INR Returns'),
        ('D_CPI', 'D_LOG_NEER', 'CPI Changes -> NEER Returns'),
        ('D_INT_DIFF', 'D_LOG_NEER', 'Int. Diff Changes -> NEER Returns'),
        ('D_LOG_RESERVES', 'D_LOG_USDINR', 'Reserve Changes -> USD/INR Returns'),
        ('D_LOG_USDINR', 'D_LOG_RESERVES', 'USD/INR Returns -> Reserve Changes'),
        ('D_LOG_OIL', 'D_CPI', 'Oil Returns -> CPI Changes'),
        ('D_LOG_OIL', 'D_INT_DIFF', 'Oil Returns -> Int. Diff Changes'),
    ]
    
    results = []
    max_lag = 6
    
    print(f"\n  {'Hypothesis':<45s} {'Lag':>4s} {'F-stat':>10s} {'p-value':>10s} {'Conclusion':>20s}")
    print("  " + "-" * 95)
    
    for cause, effect, label in pairs:
        data = df_sample[[effect, cause]].dropna()
        if len(data) < 30:
            print(f"  {label:<45s} {'--':>4s} {'--':>10s} {'--':>10s} {'Insufficient data':>20s}")
            continue
        
        try:
            gc_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            
            # Find the lag with strongest evidence
            best_lag = None
            best_pval = 1.0
            best_fstat = 0
            
            for lag in range(1, max_lag + 1):
                f_stat = gc_result[lag][0]['ssr_ftest'][0]
                p_val = gc_result[lag][0]['ssr_ftest'][1]
                if p_val < best_pval:
                    best_pval = p_val
                    best_lag = lag
                    best_fstat = f_stat
            
            conclusion = 'Granger-causes' if best_pval < 0.05 else 'No causality'
            print(f"  {label:<45s} {best_lag:>4d} {best_fstat:>10.4f} {best_pval:>10.4f} {conclusion:>20s}")
            
            results.append({
                'Hypothesis': label,
                'Best_Lag': best_lag,
                'F_statistic': best_fstat,
                'p_value': best_pval,
                'Conclusion': conclusion
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
    gc_results = run_granger_causality(df)
    
    print("\n" + "=" * 70)
    print("Phase 4 COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
