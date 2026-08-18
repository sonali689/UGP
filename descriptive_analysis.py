"""
Phase 2: Descriptive Statistics & Visualization
================================================
Produces:
  - Summary statistics table
  - Correlation matrix heatmap
  - Time series plots of all key variables
  - Rolling correlations (oil vs NEER, oil vs USD/INR)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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

def summary_statistics(df):
    """Generate and save summary statistics table."""
    print("\n[1/4] Summary Statistics...")
    
    # Key variables for analysis
    cols = ['NEER', 'USDINR', 'OIL_BRENT', 'CPI_YOY', 'EPU', 
            'FX_RESERVES', 'REPO_RATE', 'FED_RATE', 'IIP_YOY',
            'INT_DIFF', 'D_LOG_NEER', 'D_LOG_USDINR', 'D_LOG_OIL']
    
    stats = df[cols].describe().T
    stats['skewness'] = df[cols].skew()
    stats['kurtosis'] = df[cols].kurtosis()
    stats['obs'] = df[cols].count()
    
    # Reorder columns
    stats = stats[['obs', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']]
    stats.columns = ['Obs', 'Mean', 'Std Dev', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Skewness', 'Kurtosis']
    
    # Save
    stats.to_csv(os.path.join(OUTPUT_DIR, 'summary_statistics.csv'), float_format='%.4f')
    
    # Print formatted
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    print(stats.to_string(float_format=lambda x: f'{x:.4f}'))
    print("=" * 120)
    
    return stats

def correlation_matrix(df):
    """Generate correlation matrix heatmap."""
    print("\n[2/4] Correlation Matrix...")
    
    # Level variables
    level_cols = ['NEER', 'USDINR', 'OIL_BRENT', 'CPI_YOY', 'EPU', 
                  'FX_RESERVES', 'REPO_RATE', 'FED_RATE']
    
    # First-difference / return variables
    diff_cols = ['D_LOG_NEER', 'D_LOG_USDINR', 'D_LOG_OIL', 'D_CPI', 
                 'D_LOG_EPU', 'D_LOG_RESERVES', 'D_INT_DIFF']
    
    # --- Level correlations ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    corr_level = df[level_cols].corr()
    mask = np.triu(np.ones_like(corr_level, dtype=bool))
    sns.heatmap(corr_level, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=axes[0],
                square=True, linewidths=0.5)
    axes[0].set_title('Correlation Matrix: Level Variables', fontsize=14, fontweight='bold')
    
    # --- First-difference correlations ---
    corr_diff = df[diff_cols].corr()
    mask2 = np.triu(np.ones_like(corr_diff, dtype=bool))
    sns.heatmap(corr_diff, mask=mask2, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=axes[1],
                square=True, linewidths=0.5)
    axes[1].set_title('Correlation Matrix: First-Difference Variables', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save numeric tables
    corr_level.to_csv(os.path.join(OUTPUT_DIR, 'correlation_levels.csv'), float_format='%.4f')
    corr_diff.to_csv(os.path.join(OUTPUT_DIR, 'correlation_differences.csv'), float_format='%.4f')
    
    print("  Saved: correlation_matrices.png, correlation_levels.csv, correlation_differences.csv")

def time_series_plots(df):
    """Create comprehensive time series plots."""
    print("\n[3/4] Time Series Plots...")
    
    # Use Sample A period (2001-04 onward)
    df_plot = df.loc['2001-04':].copy()
    
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    fig.suptitle('Key Macroeconomic Variables — India (2001–2026)', 
                 fontsize=16, fontweight='bold', y=1.01)
    
    # (1) NEER
    ax = axes[0, 0]
    ax.plot(df_plot.index, df_plot['NEER'], color='#2c3e50', linewidth=1.2)
    ax.set_title('Nominal Effective Exchange Rate (NEER)', fontweight='bold')
    ax.set_ylabel('Index (2020=100)')
    ax.grid(True, alpha=0.3)
    ax.axhline(100, color='red', linestyle='--', alpha=0.5, label='Base (2020)')
    ax.legend(fontsize=8)
    
    # (2) USD/INR
    ax = axes[0, 1]
    ax.plot(df_plot.index, df_plot['USDINR'], color='#e74c3c', linewidth=1.2)
    ax.set_title('USD/INR Bilateral Exchange Rate', fontweight='bold')
    ax.set_ylabel('INR per USD')
    ax.grid(True, alpha=0.3)
    
    # (3) Brent Crude Oil
    ax = axes[1, 0]
    ax.plot(df_plot.index, df_plot['OIL_BRENT'], color='#f39c12', linewidth=1.2)
    ax.set_title('Brent Crude Oil Price', fontweight='bold')
    ax.set_ylabel('USD/Barrel')
    ax.grid(True, alpha=0.3)
    # Shade crisis periods
    for start, end, label in [('2008-06', '2009-02', 'GFC'), 
                                ('2020-02', '2020-06', 'COVID'),
                                ('2014-06', '2016-01', 'Oil Crash')]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color='red')
    
    # (4) CPI Inflation
    ax = axes[1, 1]
    ax.plot(df_plot.index, df_plot['CPI_YOY'], color='#e67e22', linewidth=1.2)
    ax.axhline(4, color='green', linestyle='--', alpha=0.6, label='RBI Target (4%)')
    ax.axhline(6, color='red', linestyle='--', alpha=0.6, label='Upper Band (6%)')
    ax.set_title('CPI Inflation (YoY %)', fontweight='bold')
    ax.set_ylabel('% YoY')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (5) Economic Policy Uncertainty
    ax = axes[2, 0]
    ax.plot(df_plot.index, df_plot['EPU'], color='#8e44ad', linewidth=1.2)
    ax.set_title('Economic Policy Uncertainty Index', fontweight='bold')
    ax.set_ylabel('Index')
    ax.grid(True, alpha=0.3)
    
    # (6) FX Reserves
    ax = axes[2, 1]
    ax.plot(df_plot.index, df_plot['FX_RESERVES'] / 1000, color='#27ae60', linewidth=1.2)
    ax.set_title('Foreign Exchange Reserves', fontweight='bold')
    ax.set_ylabel('USD Billion')
    ax.grid(True, alpha=0.3)
    
    # (7) Interest Rates
    ax = axes[3, 0]
    ax.plot(df_plot.index, df_plot['REPO_RATE'], color='#2980b9', linewidth=1.2, label='India Repo Rate')
    ax.plot(df_plot.index, df_plot['FED_RATE'], color='#c0392b', linewidth=1.2, label='US Fed Funds Rate')
    ax.set_title('Policy Interest Rates', fontweight='bold')
    ax.set_ylabel('% p.a.')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # (8) Interest Rate Differential
    ax = axes[3, 1]
    ax.plot(df_plot.index, df_plot['INT_DIFF'], color='#16a085', linewidth=1.2)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_title('Interest Rate Differential (India − US)', fontweight='bold')
    ax.set_ylabel('Percentage Points')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'time_series_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Second figure: Returns / first differences ---
    fig2, axes2 = plt.subplots(3, 2, figsize=(18, 14))
    fig2.suptitle('Monthly Returns & Changes (2001–2026)', fontsize=16, fontweight='bold', y=1.01)
    
    plots = [
        ('D_LOG_NEER', 'NEER Monthly Return', '#2c3e50'),
        ('D_LOG_USDINR', 'USD/INR Monthly Return', '#e74c3c'),
        ('D_LOG_OIL', 'Oil Price Monthly Return', '#f39c12'),
        ('D_CPI', 'Change in CPI (Δ%)', '#e67e22'),
        ('D_LOG_EPU', 'EPU Monthly Change', '#8e44ad'),
        ('DELTA_RESERVES', 'ΔFX Reserves (USD mn)', '#27ae60'),
    ]
    
    for idx, (col, title, color) in enumerate(plots):
        ax = axes2[idx // 2, idx % 2]
        ax.bar(df_plot.index, df_plot[col], color=color, alpha=0.7, width=25)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'returns_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  Saved: time_series_overview.png, returns_overview.png")

def rolling_correlations(df):
    """Compute and plot rolling correlations."""
    print("\n[4/4] Rolling Correlations...")
    
    df_sample = df.loc['2001-04':].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Rolling 24-Month Correlations (2001–2026)', fontsize=14, fontweight='bold')
    
    window = 24
    
    pairs = [
        ('D_LOG_OIL', 'D_LOG_NEER', 'Oil Returns vs NEER Returns'),
        ('D_LOG_OIL', 'D_LOG_USDINR', 'Oil Returns vs USD/INR Returns'),
        ('D_LOG_EPU', 'D_LOG_NEER', 'EPU Changes vs NEER Returns'),
        ('DELTA_RESERVES', 'D_LOG_USDINR', 'ΔReserves vs USD/INR Returns'),
    ]
    
    for idx, (x, y, title) in enumerate(pairs):
        ax = axes[idx // 2, idx % 2]
        rolling_corr = df_sample[x].rolling(window).corr(df_sample[y])
        ax.plot(df_sample.index, rolling_corr, color='#2c3e50', linewidth=1.2)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.fill_between(df_sample.index, rolling_corr, 0, alpha=0.15, color='steelblue')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Correlation')
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'rolling_correlations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  Saved: rolling_correlations.png")


def main():
    print("=" * 70)
    print("PHASE 2: DESCRIPTIVE STATISTICS & VISUALIZATION")
    print("=" * 70)
    
    df = load_data()
    
    stats = summary_statistics(df)
    correlation_matrix(df)
    time_series_plots(df)
    rolling_correlations(df)
    
    print("\n" + "=" * 70)
    print("Phase 2 COMPLETE! All outputs saved to:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == '__main__':
    main()
