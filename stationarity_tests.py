"""
Phase 3: Stationarity & Cointegration Tests
============================================
Produces:
  - ADF, PP, KPSS test results for all variables (levels and first differences)
  - Integration order summary table
  - Johansen cointegration test results
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
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

def adf_test(series, name, maxlag=None):
    """Augmented Dickey-Fuller test."""
    s = series.dropna()
    if len(s) < 20:
        return {'Variable': name, 'ADF_stat': np.nan, 'ADF_pvalue': np.nan, 
                'ADF_lags': np.nan, 'ADF_result': 'Insufficient data'}
    
    result = adfuller(s, maxlag=maxlag, autolag='AIC')
    return {
        'Variable': name,
        'ADF_stat': result[0],
        'ADF_pvalue': result[1],
        'ADF_lags': result[2],
        'ADF_1%': result[4]['1%'],
        'ADF_5%': result[4]['5%'],
        'ADF_10%': result[4]['10%'],
        'ADF_result': 'Stationary' if result[1] < 0.05 else 'Non-stationary'
    }

def kpss_test(series, name, regression='c'):
    """KPSS test (null = stationary)."""
    s = series.dropna()
    if len(s) < 20:
        return {'KPSS_stat': np.nan, 'KPSS_pvalue': np.nan, 'KPSS_result': 'Insufficient data'}
    
    result = kpss(s, regression=regression, nlags='auto')
    return {
        'KPSS_stat': result[0],
        'KPSS_pvalue': result[1],
        'KPSS_1%': result[3]['1%'],
        'KPSS_5%': result[3]['5%'],
        'KPSS_10%': result[3]['10%'],
        'KPSS_result': 'Non-stationary' if result[1] < 0.05 else 'Stationary'
    }

def run_unit_root_tests(df):
    """Run ADF and KPSS on level and first-difference for all key variables."""
    print("\n[1/2] Unit Root Tests...")
    
    # Variables to test
    level_vars = {
        'LOG_NEER': 'Log(NEER)',
        'LOG_USDINR': 'Log(USD/INR)',
        'LOG_OIL': 'Log(Oil Price)',
        'CPI_YOY': 'CPI Inflation (%)',
        'LOG_EPU': 'Log(EPU)',
        'LOG_RESERVES': 'Log(FX Reserves)',
        'REPO_RATE': 'Repo Rate',
        'FED_RATE': 'Fed Funds Rate',
        'INT_DIFF': 'Interest Differential',
        'IIP_YOY': 'IIP Growth (%)',
    }
    
    diff_vars = {
        'D_LOG_NEER': 'ΔLog(NEER)',
        'D_LOG_USDINR': 'ΔLog(USD/INR)',
        'D_LOG_OIL': 'ΔLog(Oil)',
        'D_CPI': 'ΔCPI',
        'D_LOG_EPU': 'ΔLog(EPU)',
        'D_LOG_RESERVES': 'ΔLog(Reserves)',
        'D_INT_DIFF': 'ΔInterest Diff',
    }
    
    # Use Sample A period
    df_sample = df.loc['2001-04':'2026-06'].copy()
    
    results = []
    
    print("\n  ── Level Variables ──")
    print(f"  {'Variable':<25s} {'ADF stat':>10s} {'p-value':>10s} {'ADF':>15s} {'KPSS stat':>10s} {'p-value':>10s} {'KPSS':>15s}")
    print("  " + "-" * 100)
    
    for col, label in level_vars.items():
        adf = adf_test(df_sample[col], label)
        kpss_r = kpss_test(df_sample[col], label)
        
        row = {**adf, **kpss_r, 'Type': 'Level'}
        results.append(row)
        
        print(f"  {label:<25s} {adf['ADF_stat']:>10.4f} {adf['ADF_pvalue']:>10.4f} {adf['ADF_result']:>15s} "
              f"{kpss_r['KPSS_stat']:>10.4f} {kpss_r['KPSS_pvalue']:>10.4f} {kpss_r['KPSS_result']:>15s}")
    
    print("\n  ── First-Difference Variables ──")
    print(f"  {'Variable':<25s} {'ADF stat':>10s} {'p-value':>10s} {'ADF':>15s} {'KPSS stat':>10s} {'p-value':>10s} {'KPSS':>15s}")
    print("  " + "-" * 100)
    
    for col, label in diff_vars.items():
        adf = adf_test(df_sample[col], label)
        kpss_r = kpss_test(df_sample[col], label)
        
        row = {**adf, **kpss_r, 'Type': 'First Difference'}
        results.append(row)
        
        print(f"  {label:<25s} {adf['ADF_stat']:>10.4f} {adf['ADF_pvalue']:>10.4f} {adf['ADF_result']:>15s} "
              f"{kpss_r['KPSS_stat']:>10.4f} {kpss_r['KPSS_pvalue']:>10.4f} {kpss_r['KPSS_result']:>15s}")
    
    # Determine integration order
    print("\n  ── Integration Order Summary ──")
    integration_orders = []
    for col, label in level_vars.items():
        adf_level = adf_test(df_sample[col], label)
        kpss_level = kpss_test(df_sample[col], label)
        
        # Check first difference
        diff_col = 'D_' + col if 'D_' + col in df_sample.columns else None
        if diff_col is None:
            # Compute it
            diff_series = df_sample[col].diff()
        else:
            diff_series = df_sample[diff_col]
        
        adf_diff = adf_test(diff_series, f'Δ{label}')
        
        # Determine order
        if adf_level['ADF_result'] == 'Stationary' and kpss_level['KPSS_result'] == 'Stationary':
            order = 'I(0)'
        elif adf_diff['ADF_result'] == 'Stationary':
            order = 'I(1)'
        else:
            order = 'I(2) or unclear'
        
        integration_orders.append({
            'Variable': label,
            'Column': col,
            'ADF_Level_pvalue': adf_level['ADF_pvalue'],
            'KPSS_Level_pvalue': kpss_level['KPSS_pvalue'],
            'ADF_Diff_pvalue': adf_diff['ADF_pvalue'],
            'Integration_Order': order
        })
        print(f"  {label:<25s}: {order}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'unit_root_tests.csv'), index=False, float_format='%.4f')
    
    io_df = pd.DataFrame(integration_orders)
    io_df.to_csv(os.path.join(OUTPUT_DIR, 'integration_orders.csv'), index=False, float_format='%.4f')
    
    print("\n  Saved: unit_root_tests.csv, integration_orders.csv")
    return io_df

def johansen_test(df):
    """Johansen cointegration test for I(1) variables."""
    print("\n[2/2] Johansen Cointegration Test...")
    
    df_sample = df.loc['2001-04':'2026-06'].copy()
    
    # Use key I(1) variables (log levels)
    coint_vars = ['LOG_NEER', 'LOG_OIL', 'LOG_RESERVES', 'LOG_EPU']
    
    # Add interest rate variables
    additional = ['REPO_RATE', 'FED_RATE', 'CPI_YOY']
    all_vars = coint_vars + additional
    
    # Drop NaN
    data = df_sample[all_vars].dropna()
    print(f"  Using {len(data)} observations, {len(all_vars)} variables")
    print(f"  Variables: {', '.join(all_vars)}")
    
    # Run Johansen test (det_order: -1=no const, 0=restricted const, 1=unrestricted const)
    # k_ar_diff: number of lagged differences (like VAR lags - 1)
    for det_order in [0, 1]:
        det_label = 'Restricted Constant' if det_order == 0 else 'Unrestricted Constant'
        print(f"\n  ── Johansen Test ({det_label}) ──")
        
        try:
            result = coint_johansen(data.values, det_order=det_order, k_ar_diff=2)
            
            print(f"\n  {'Hypothesis':<20s} {'Trace Stat':>12s} {'5% CV':>10s} {'Result':>15s}")
            print("  " + "-" * 60)
            
            n_vars = len(all_vars)
            for i in range(n_vars):
                trace_stat = result.lr1[i]
                cv_5 = result.cvt[i, 1]  # 5% critical value
                reject = 'Reject H0' if trace_stat > cv_5 else 'Fail to Reject'
                print(f"  r <= {i:<14d} {trace_stat:>12.4f} {cv_5:>10.4f} {reject:>15s}")
            
            print(f"\n  {'Hypothesis':<20s} {'Max-Eigen':>12s} {'5% CV':>10s} {'Result':>15s}")
            print("  " + "-" * 60)
            
            for i in range(n_vars):
                max_stat = result.lr2[i]
                cv_5 = result.cvm[i, 1]  # 5% critical value
                reject = 'Reject H0' if max_stat > cv_5 else 'Fail to Reject'
                print(f"  r <= {i:<14d} {max_stat:>12.4f} {cv_5:>10.4f} {reject:>15s}")
            
            # Count cointegrating vectors
            n_coint_trace = sum(1 for i in range(n_vars) if result.lr1[i] > result.cvt[i, 1])
            n_coint_max = sum(1 for i in range(n_vars) if result.lr2[i] > result.cvm[i, 1])
            print(f"\n  → Cointegrating vectors (Trace): {n_coint_trace}")
            print(f"  → Cointegrating vectors (Max-Eigen): {n_coint_max}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n  Saved results to unit_root_tests.csv (combined)")


def main():
    print("=" * 70)
    print("PHASE 3: STATIONARITY & COINTEGRATION TESTS")
    print("=" * 70)
    
    df = load_data()
    
    io_df = run_unit_root_tests(df)
    johansen_test(df)
    
    print("\n" + "=" * 70)
    print("Phase 3 COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
