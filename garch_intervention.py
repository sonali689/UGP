"""
Phase 5: GARCH Volatility & RBI Intervention Analysis
=====================================================
Produces:
  - GARCH(1,1) on exchange rate returns -> conditional volatility
  - GARCH on oil returns -> oil price uncertainty
  - Intervention effectiveness regressions (level + volatility)
  - Granger causality: Reserves <-> Exchange Rate
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import grangercausalitytests
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


def garch_exchange_rate(df):
    """Estimate GARCH(1,1) on exchange rate returns."""
    print("\n[1/4] GARCH Model: Exchange Rate Volatility...")
    
    df_sample = df.loc['2001-04':'2026-06'].copy()
    
    results = {}
    
    for col, label in [('D_LOG_NEER', 'NEER'), ('D_LOG_USDINR', 'USD/INR')]:
        print(f"\n  --- GARCH(1,1) on {label} Returns ---")
        
        returns = df_sample[col].dropna() * 100  # Scale to percentage
        
        # GARCH(1,1) with constant mean
        model = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
        res = model.fit(disp='off')
        
        print(res.summary())
        
        # Save
        safe_label = label.replace('/', '_')
        with open(os.path.join(OUTPUT_DIR, f'garch_{safe_label.lower()}.txt'), 'w', encoding='utf-8') as f:
            f.write(str(res.summary()))
        
        # Extract conditional volatility
        cond_vol = res.conditional_volatility
        cond_var = cond_vol ** 2
        
        # Store for later use
        results[label] = {
            'model': res,
            'cond_vol': cond_vol,
            'cond_var': cond_var,
            'returns': returns
        }
        
        # Print key parameters
        params = res.params
        print(f"\n  Key Parameters:")
        print(f"    omega (constant):  {params.get('omega', np.nan):.6f}")
        print(f"    alpha (ARCH):      {params.get('alpha[1]', np.nan):.6f}")
        print(f"    beta  (GARCH):     {params.get('beta[1]', np.nan):.6f}")
        alpha_beta = params.get('alpha[1]', 0) + params.get('beta[1]', 0)
        print(f"    alpha + beta:      {alpha_beta:.6f} {'(persistent)' if alpha_beta > 0.95 else '(moderate persistence)'}")
        
        # EGARCH for asymmetry
        print(f"\n  --- EGARCH(1,1) on {label} Returns ---")
        model_e = arch_model(returns, vol='EGARCH', p=1, q=1, mean='Constant', dist='normal')
        res_e = model_e.fit(disp='off')
        
        print(f"    gamma (asymmetry): {res_e.params.get('gamma[1]', np.nan):.6f}")
        print(f"    Interpretation: {'Negative shocks increase volatility more' if res_e.params.get('gamma[1]', 0) < 0 else 'Positive shocks increase volatility more'}")
        
        with open(os.path.join(OUTPUT_DIR, f'egarch_{safe_label.lower()}.txt'), 'w', encoding='utf-8') as f:
            f.write(str(res_e.summary()))
    
    return results


def garch_oil(df):
    """Estimate GARCH(1,1) on oil price returns."""
    print("\n[2/4] GARCH Model: Oil Price Volatility...")
    
    df_sample = df.loc['2001-04':'2026-06'].copy()
    
    oil_returns = df_sample['D_LOG_OIL'].dropna() * 100
    
    model = arch_model(oil_returns, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
    res = model.fit(disp='off')
    
    print(res.summary())
    
    with open(os.path.join(OUTPUT_DIR, 'garch_oil.txt'), 'w', encoding='utf-8') as f:
        f.write(str(res.summary()))
    
    cond_vol_oil = res.conditional_volatility
    
    print(f"\n  Key Parameters:")
    params = res.params
    print(f"    omega:  {params.get('omega', np.nan):.6f}")
    print(f"    alpha:  {params.get('alpha[1]', np.nan):.6f}")
    print(f"    beta:   {params.get('beta[1]', np.nan):.6f}")
    print(f"    alpha + beta: {params.get('alpha[1]', 0) + params.get('beta[1]', 0):.6f}")
    
    return cond_vol_oil


def plot_volatilities(df, er_results, oil_vol):
    """Plot conditional volatilities together."""
    print("\n  Plotting conditional volatilities...")
    
    df_sample = df.loc['2001-04':'2026-06'].copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Conditional Volatility (GARCH(1,1) Estimates)', fontsize=14, fontweight='bold')
    
    # NEER volatility
    ax = axes[0]
    neer_vol = er_results['NEER']['cond_vol']
    ax.plot(neer_vol.index, neer_vol, color='#2c3e50', linewidth=1)
    ax.fill_between(neer_vol.index, 0, neer_vol, alpha=0.2, color='#2c3e50')
    ax.set_title('NEER Return Volatility', fontweight='bold')
    ax.set_ylabel('Cond. Std Dev (%)')
    ax.grid(True, alpha=0.3)
    # Mark crisis periods
    for start, end, label in [('2008-09', '2009-03', 'GFC'), ('2013-06', '2013-09', 'Taper'),
                                ('2020-03', '2020-06', 'COVID')]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.2, color='red')
    
    # USD/INR volatility
    ax = axes[1]
    usdinr_vol = er_results['USD/INR']['cond_vol']
    ax.plot(usdinr_vol.index, usdinr_vol, color='#e74c3c', linewidth=1)
    ax.fill_between(usdinr_vol.index, 0, usdinr_vol, alpha=0.2, color='#e74c3c')
    ax.set_title('USD/INR Return Volatility', fontweight='bold')
    ax.set_ylabel('Cond. Std Dev (%)')
    ax.grid(True, alpha=0.3)
    for start, end, label in [('2008-09', '2009-03', 'GFC'), ('2013-06', '2013-09', 'Taper'),
                                ('2020-03', '2020-06', 'COVID')]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.2, color='red')
    
    # Oil volatility
    ax = axes[2]
    ax.plot(oil_vol.index, oil_vol, color='#f39c12', linewidth=1)
    ax.fill_between(oil_vol.index, 0, oil_vol, alpha=0.2, color='#f39c12')
    ax.set_title('Oil Price Return Volatility', fontweight='bold')
    ax.set_ylabel('Cond. Std Dev (%)')
    ax.grid(True, alpha=0.3)
    for start, end, label in [('2008-06', '2009-03', 'GFC'), ('2014-06', '2016-02', 'Oil Crash'),
                                ('2020-03', '2020-06', 'COVID')]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.2, color='red')
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'conditional_volatilities.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: conditional_volatilities.png")


def intervention_analysis(df, er_results, oil_vol):
    """Test RBI intervention effectiveness."""
    print("\n[3/4] RBI Intervention Effectiveness Analysis...")
    
    df_sample = df.loc['2001-04':'2026-06'].copy()
    
    # ── Model 1: Level Effect ──
    # Does reserve intervention affect exchange rate level?
    # D_LOG_USDINR = a + b1*D_LOG_RESERVES + b2*D_LOG_OIL + b3*D_LOG_EPU + b4*D_CPI + e
    print("\n  --- Model 1: Level Effect of Intervention ---")
    
    dep = df_sample['D_LOG_USDINR']
    regressors = df_sample[['D_LOG_RESERVES', 'D_LOG_OIL', 'D_LOG_EPU', 'D_CPI', 'D_INT_DIFF']]
    
    data = pd.concat([dep, regressors], axis=1).dropna()
    y = data['D_LOG_USDINR']
    X = add_constant(data[['D_LOG_RESERVES', 'D_LOG_OIL', 'D_LOG_EPU', 'D_CPI', 'D_INT_DIFF']])
    
    ols_level = OLS(y, X).fit(cov_type='HC1')  # Heteroskedasticity-robust
    print(ols_level.summary())
    
    with open(os.path.join(OUTPUT_DIR, 'intervention_level_effect.txt'), 'w', encoding='utf-8') as f:
        f.write(str(ols_level.summary()))
    
    print(f"\n  Key Result:")
    print(f"    Coeff on D_LOG_RESERVES: {ols_level.params['D_LOG_RESERVES']:.6f}")
    print(f"    p-value:                 {ols_level.pvalues['D_LOG_RESERVES']:.6f}")
    sign = 'negative' if ols_level.params['D_LOG_RESERVES'] < 0 else 'positive'
    sig = 'significant' if ols_level.pvalues['D_LOG_RESERVES'] < 0.05 else 'not significant'
    print(f"    Interpretation: Reserve accumulation has a {sign} and {sig} effect on USD/INR depreciation")
    
    # ── Model 2: Volatility Effect ──
    # Does intervention reduce exchange rate volatility?
    print("\n  --- Model 2: Volatility Effect of Intervention ---")
    
    usdinr_vol_series = er_results['USD/INR']['cond_var']
    abs_reserves = df_sample['D_LOG_RESERVES'].abs()
    oil_vol_series = oil_vol ** 2
    epu_level = df_sample['LOG_EPU']
    
    vol_data = pd.DataFrame({
        'ER_VOLATILITY': usdinr_vol_series,
        'ABS_D_RESERVES': abs_reserves,
        'OIL_VOLATILITY': oil_vol_series,
        'LOG_EPU': epu_level,
        'INT_DIFF': df_sample['INT_DIFF']
    }).dropna()
    
    y_vol = vol_data['ER_VOLATILITY']
    X_vol = add_constant(vol_data[['ABS_D_RESERVES', 'OIL_VOLATILITY', 'LOG_EPU', 'INT_DIFF']])
    
    ols_vol = OLS(y_vol, X_vol).fit(cov_type='HC1')
    print(ols_vol.summary())
    
    with open(os.path.join(OUTPUT_DIR, 'intervention_volatility_effect.txt'), 'w', encoding='utf-8') as f:
        f.write(str(ols_vol.summary()))
    
    print(f"\n  Key Result:")
    print(f"    Coeff on ABS_D_RESERVES: {ols_vol.params['ABS_D_RESERVES']:.6f}")
    print(f"    p-value:                 {ols_vol.pvalues['ABS_D_RESERVES']:.6f}")
    effect = 'increases' if ols_vol.params['ABS_D_RESERVES'] > 0 else 'decreases'
    sig = 'significant' if ols_vol.pvalues['ABS_D_RESERVES'] < 0.05 else 'not significant'
    print(f"    Interpretation: FX intervention {effect} exchange rate volatility ({sig})")
    
    # ── Model 3: Asymmetric Effects ──
    print("\n  --- Model 3: Asymmetric Intervention Effects ---")
    
    # Split reserves changes into buying (positive) and selling (negative)
    df_asym = df_sample[['D_LOG_USDINR', 'D_LOG_RESERVES', 'D_LOG_OIL', 'D_CPI']].dropna().copy()
    df_asym['RESERVES_BUY'] = np.maximum(0, df_asym['D_LOG_RESERVES'])  # Accumulation
    df_asym['RESERVES_SELL'] = np.minimum(0, df_asym['D_LOG_RESERVES'])  # Decumulation
    
    y_asym = df_asym['D_LOG_USDINR']
    X_asym = add_constant(df_asym[['RESERVES_BUY', 'RESERVES_SELL', 'D_LOG_OIL', 'D_CPI']])
    
    ols_asym = OLS(y_asym, X_asym).fit(cov_type='HC1')
    print(ols_asym.summary())
    
    with open(os.path.join(OUTPUT_DIR, 'intervention_asymmetric.txt'), 'w', encoding='utf-8') as f:
        f.write(str(ols_asym.summary()))
    
    print(f"\n  Asymmetric Results:")
    print(f"    Reserve Buying  coeff: {ols_asym.params['RESERVES_BUY']:.6f} (p={ols_asym.pvalues['RESERVES_BUY']:.4f})")
    print(f"    Reserve Selling coeff: {ols_asym.params['RESERVES_SELL']:.6f} (p={ols_asym.pvalues['RESERVES_SELL']:.4f})")


def intervention_granger(df):
    """Granger causality between reserves and exchange rate."""
    print("\n[4/4] Granger Causality: Reserves <-> Exchange Rate...")
    
    df_sample = df.loc['2001-04':'2026-06'].copy()
    
    pairs = [
        ('D_LOG_RESERVES', 'D_LOG_USDINR', 'Reserve Changes -> USD/INR Returns'),
        ('D_LOG_USDINR', 'D_LOG_RESERVES', 'USD/INR Returns -> Reserve Changes'),
        ('D_LOG_RESERVES', 'D_LOG_NEER', 'Reserve Changes -> NEER Returns'),
        ('D_LOG_NEER', 'D_LOG_RESERVES', 'NEER Returns -> Reserve Changes'),
    ]
    
    print(f"\n  {'Hypothesis':<45s} {'Lag':>4s} {'F-stat':>10s} {'p-value':>10s} {'Result':>20s}")
    print("  " + "-" * 95)
    
    for cause, effect, label in pairs:
        data = df_sample[[effect, cause]].dropna()
        gc = grangercausalitytests(data, maxlag=6, verbose=False)
        
        best_lag = None
        best_pval = 1.0
        best_fstat = 0
        
        for lag in range(1, 7):
            f_stat = gc[lag][0]['ssr_ftest'][0]
            p_val = gc[lag][0]['ssr_ftest'][1]
            if p_val < best_pval:
                best_pval = p_val
                best_lag = lag
                best_fstat = f_stat
        
        result = 'YES - Granger-causes' if best_pval < 0.05 else 'NO'
        print(f"  {label:<45s} {best_lag:>4d} {best_fstat:>10.4f} {best_pval:>10.4f} {result:>20s}")


def main():
    print("=" * 70)
    print("PHASE 5: GARCH VOLATILITY & INTERVENTION ANALYSIS")
    print("=" * 70)
    
    df = load_data()
    
    er_results = garch_exchange_rate(df)
    oil_vol = garch_oil(df)
    plot_volatilities(df, er_results, oil_vol)
    intervention_analysis(df, er_results, oil_vol)
    intervention_granger(df)
    
    print("\n" + "=" * 70)
    print("Phase 5 COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
