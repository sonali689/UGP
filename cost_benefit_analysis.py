"""
Phase 6: Cost-Benefit Analysis of FX Intervention
==================================================
Produces:
  - Quasi-fiscal (carrying) cost of reserves
  - Cumulative cost over time
  - Crisis episode analysis (2008 GFC, 2013 Taper Tantrum, 2020 COVID)
  - Cost vs. benefit comparison table
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def compute_carrying_cost(df):
    """Compute quasi-fiscal carrying cost of FX reserves."""
    print("\n[1/3] Quasi-Fiscal Carrying Cost of Reserves...")
    
    df_sample = df.loc['2001-04':'2026-06'].copy().dropna(subset=['FX_RESERVES', 'REPO_RATE', 'FED_RATE'])
    
    # Carrying cost = Reserves * (Sterilization cost - Return on reserves) / 12
    # Sterilization cost proxy: India Repo Rate
    # Return on reserves proxy: US Fed Funds Rate (since reserves are mostly in US Treasuries)
    
    df_sample['CARRYING_COST_RATE'] = (df_sample['REPO_RATE'] - df_sample['FED_RATE']) / 100  # Convert to decimal
    
    # Monthly carrying cost (in USD million)
    df_sample['MONTHLY_COST'] = df_sample['FX_RESERVES'] * df_sample['CARRYING_COST_RATE'] / 12
    
    # Cumulative cost
    df_sample['CUMULATIVE_COST'] = df_sample['MONTHLY_COST'].cumsum()
    
    # Annual cost
    annual_cost = df_sample['MONTHLY_COST'].resample('YE').sum()
    
    print(f"\n  Sample: {df_sample.index.min().strftime('%Y-%m')} to {df_sample.index.max().strftime('%Y-%m')}")
    print(f"\n  Average monthly carrying cost: USD {df_sample['MONTHLY_COST'].mean():.1f} million")
    print(f"  Total cumulative cost:         USD {df_sample['CUMULATIVE_COST'].iloc[-1] / 1000:.1f} billion")
    print(f"  Average cost rate spread:      {df_sample['CARRYING_COST_RATE'].mean() * 100:.2f}%")
    
    print(f"\n  Annual Carrying Costs (USD billion):")
    print(f"  {'Year':<8s} {'Cost (USD bn)':>15s} {'Avg Reserves (USD bn)':>25s} {'Spread (%)':>12s}")
    print("  " + "-" * 65)
    
    for year in range(2002, 2027):
        yr_data = df_sample[df_sample.index.year == year]
        if len(yr_data) == 0:
            continue
        yr_cost = yr_data['MONTHLY_COST'].sum() / 1000
        yr_reserves = yr_data['FX_RESERVES'].mean() / 1000
        yr_spread = yr_data['CARRYING_COST_RATE'].mean() * 100
        print(f"  {year:<8d} {yr_cost:>15.2f} {yr_reserves:>25.1f} {yr_spread:>12.2f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Cost-Benefit Analysis: FX Intervention in India', fontsize=14, fontweight='bold')
    
    # (1) Reserves and Cost Rate
    ax = axes[0, 0]
    ax.plot(df_sample.index, df_sample['FX_RESERVES'] / 1000, color='#27ae60', linewidth=1.2, label='FX Reserves')
    ax.set_ylabel('FX Reserves (USD Billion)', color='#27ae60')
    ax.tick_params(axis='y', labelcolor='#27ae60')
    ax2 = ax.twinx()
    ax2.plot(df_sample.index, df_sample['CARRYING_COST_RATE'] * 100, color='#e74c3c', linewidth=1, alpha=0.7, label='Cost Spread')
    ax2.set_ylabel('Cost Spread: Repo - Fed (%)', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax.set_title('FX Reserves & Interest Rate Spread', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # (2) Monthly carrying cost
    ax = axes[0, 1]
    ax.bar(df_sample.index, df_sample['MONTHLY_COST'], color='#e74c3c', alpha=0.6, width=25)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('Monthly Carrying Cost', fontweight='bold')
    ax.set_ylabel('USD Million')
    ax.grid(True, alpha=0.3)
    
    # (3) Cumulative cost
    ax = axes[1, 0]
    ax.fill_between(df_sample.index, 0, df_sample['CUMULATIVE_COST'] / 1000, 
                     color='#e74c3c', alpha=0.3)
    ax.plot(df_sample.index, df_sample['CUMULATIVE_COST'] / 1000, color='#e74c3c', linewidth=1.5)
    ax.set_title('Cumulative Carrying Cost', fontweight='bold')
    ax.set_ylabel('USD Billion')
    ax.grid(True, alpha=0.3)
    
    # (4) Annual cost bars
    ax = axes[1, 1]
    annual = df_sample.groupby(df_sample.index.year)['MONTHLY_COST'].sum() / 1000
    colors = ['#e74c3c' if v > 0 else '#27ae60' for v in annual.values]
    ax.bar(annual.index, annual.values, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('Annual Carrying Cost', fontweight='bold')
    ax.set_ylabel('USD Billion')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'cost_benefit_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n  Saved: cost_benefit_analysis.png")
    
    # Save data
    cost_df = df_sample[['FX_RESERVES', 'REPO_RATE', 'FED_RATE', 'CARRYING_COST_RATE', 
                          'MONTHLY_COST', 'CUMULATIVE_COST']].copy()
    cost_df.to_csv(os.path.join(OUTPUT_DIR, 'carrying_cost_data.csv'), float_format='%.4f')
    
    return df_sample


def crisis_episode_analysis(df):
    """Analyze RBI intervention during major crisis episodes."""
    print("\n[2/3] Crisis Episode Analysis...")
    
    episodes = [
        ('2008 Global Financial Crisis', '2008-06', '2009-06'),
        ('2013 Taper Tantrum', '2013-05', '2013-12'),
        ('2014-16 Oil Price Crash', '2014-06', '2016-03'),
        ('2018 EM Crisis', '2018-04', '2018-10'),
        ('2020 COVID-19 Pandemic', '2020-02', '2020-09'),
        ('2022 Russia-Ukraine War', '2022-02', '2022-10'),
    ]
    
    print(f"\n  {'Episode':<35s} {'USDINR Start':>13s} {'USDINR End':>11s} {'Chg (%)':>9s} "
          f"{'Reserves Start':>16s} {'Reserves End':>14s} {'Chg (USD bn)':>13s}")
    print("  " + "-" * 120)
    
    episode_results = []
    
    for name, start, end in episodes:
        ep_data = df.loc[start:end]
        if len(ep_data) == 0 or ep_data['USDINR'].isna().all():
            continue
        
        usdinr_start = ep_data['USDINR'].dropna().iloc[0]
        usdinr_end = ep_data['USDINR'].dropna().iloc[-1]
        usdinr_chg = (usdinr_end / usdinr_start - 1) * 100
        
        res_start = ep_data['FX_RESERVES'].dropna().iloc[0]
        res_end = ep_data['FX_RESERVES'].dropna().iloc[-1]
        res_chg = (res_end - res_start) / 1000
        
        neer_start = ep_data['NEER'].dropna().iloc[0] if ep_data['NEER'].notna().any() else np.nan
        neer_end = ep_data['NEER'].dropna().iloc[-1] if ep_data['NEER'].notna().any() else np.nan
        neer_chg = (neer_end / neer_start - 1) * 100 if not np.isnan(neer_start) else np.nan
        
        # Volatility during episode
        vol = ep_data['D_LOG_USDINR'].std() * 100 if ep_data['D_LOG_USDINR'].notna().any() else np.nan
        
        print(f"  {name:<35s} {usdinr_start:>13.2f} {usdinr_end:>11.2f} {usdinr_chg:>8.1f}% "
              f"{res_start/1000:>14.1f}bn {res_end/1000:>12.1f}bn {res_chg:>11.1f}bn")
        
        episode_results.append({
            'Episode': name,
            'Start': start,
            'End': end,
            'USDINR_Start': usdinr_start,
            'USDINR_End': usdinr_end,
            'USDINR_Change_Pct': usdinr_chg,
            'Reserves_Start_Bn': res_start / 1000,
            'Reserves_End_Bn': res_end / 1000,
            'Reserves_Change_Bn': res_chg,
            'NEER_Change_Pct': neer_chg,
            'Monthly_Volatility_Pct': vol
        })
    
    ep_df = pd.DataFrame(episode_results)
    ep_df.to_csv(os.path.join(OUTPUT_DIR, 'crisis_episodes.csv'), index=False, float_format='%.2f')
    
    # Plot crisis episodes
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle('RBI Intervention During Crisis Episodes', fontsize=14, fontweight='bold')
    
    for idx, (name, start, end) in enumerate(episodes):
        if idx >= 6:
            break
        ax = axes[idx // 2, idx % 2]
        
        # Extend window
        start_dt = pd.Timestamp(start) - pd.DateOffset(months=3)
        end_dt = pd.Timestamp(end) + pd.DateOffset(months=3)
        
        ep_data = df.loc[start_dt:end_dt]
        
        # Plot USDINR
        ax.plot(ep_data.index, ep_data['USDINR'], color='#e74c3c', linewidth=1.5, label='USD/INR')
        ax.set_ylabel('USD/INR', color='#e74c3c')
        ax.tick_params(axis='y', labelcolor='#e74c3c')
        
        # Plot reserves on secondary axis
        ax2 = ax.twinx()
        ax2.plot(ep_data.index, ep_data['FX_RESERVES'] / 1000, color='#27ae60', 
                 linewidth=1.5, linestyle='--', label='FX Reserves')
        ax2.set_ylabel('Reserves (USD bn)', color='#27ae60')
        ax2.tick_params(axis='y', labelcolor='#27ae60')
        
        # Shade crisis period
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color='red')
        
        ax.set_title(name, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'crisis_episodes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n  Saved: crisis_episodes.csv, crisis_episodes.png")
    return ep_df


def cost_benefit_summary(df, cost_data):
    """Summarize costs vs benefits of intervention."""
    print("\n[3/3] Cost-Benefit Summary...")
    
    df_sample = df.loc['2001-04':'2026-06'].copy()
    
    # Total carrying cost
    total_cost = cost_data['CUMULATIVE_COST'].iloc[-1] / 1000  # USD billion
    
    # Benefits: Reduced volatility (compare periods of high vs low intervention)
    # Define high intervention: |delta_reserves| > median
    median_intervention = df_sample['DELTA_RESERVES'].abs().median()
    high_intervention = df_sample['DELTA_RESERVES'].abs() > median_intervention
    
    vol_high_int = df_sample.loc[high_intervention, 'D_LOG_USDINR'].std() * 100
    vol_low_int = df_sample.loc[~high_intervention, 'D_LOG_USDINR'].std() * 100
    
    # Reserve adequacy metrics
    latest = df_sample.dropna(subset=['FX_RESERVES']).iloc[-1]
    reserves_bn = latest['FX_RESERVES'] / 1000
    
    print(f"\n  === COST-BENEFIT SUMMARY ===")
    print(f"\n  COSTS:")
    print(f"    Total cumulative carrying cost (2001-2026):  USD {total_cost:.1f} billion")
    print(f"    Average annual carrying cost:                USD {total_cost / 25:.1f} billion")
    print(f"    Average cost spread (Repo - Fed):            {cost_data['CARRYING_COST_RATE'].mean() * 100:.2f}%")
    
    print(f"\n  BENEFITS:")
    print(f"    Volatility (high intervention months):       {vol_high_int:.4f}%")
    print(f"    Volatility (low intervention months):        {vol_low_int:.4f}%")
    vol_reduction = (1 - vol_high_int / vol_low_int) * 100
    print(f"    Volatility difference:                       {vol_reduction:.1f}%")
    
    print(f"\n  RESERVE ADEQUACY:")
    print(f"    Current reserves:                            USD {reserves_bn:.1f} billion")
    
    # Import cover (rough: India imports ~$600bn annually)
    import_cover = reserves_bn / (600 / 12)  # months of import cover
    print(f"    Estimated import cover:                      ~{import_cover:.1f} months")
    
    print(f"\n  OVERALL ASSESSMENT:")
    if vol_high_int < vol_low_int:
        print(f"    -> Active intervention is ASSOCIATED with LOWER volatility")
        print(f"    -> However, this is correlation, not necessarily causation")
        print(f"       (RBI intervenes more when markets are calm/manageable)")
    else:
        print(f"    -> Active intervention periods show HIGHER volatility")
        print(f"    -> This is consistent with 'leaning against the wind' behavior")
        print(f"       (RBI intervenes precisely when volatility is high)")
    
    print(f"\n    -> The paper (Aktug & Rezghi, 2026) argues FXI is welfare-improving")
    print(f"       when combined with optimal monetary policy, as it breaks the")
    print(f"       link between commodity shocks and financial risk.")
    print(f"    -> India's reserve accumulation provides crisis insurance value")
    print(f"       that is difficult to quantify but evidently useful during")
    print(f"       GFC (2008), Taper Tantrum (2013), and COVID (2020).")
    
    # Save summary
    summary = {
        'Total Carrying Cost (USD bn)': total_cost,
        'Average Annual Cost (USD bn)': total_cost / 25,
        'Average Cost Spread (%)': cost_data['CARRYING_COST_RATE'].mean() * 100,
        'Volatility High Intervention (%)': vol_high_int,
        'Volatility Low Intervention (%)': vol_low_int,
        'Current Reserves (USD bn)': reserves_bn,
        'Import Cover (months)': import_cover,
    }
    
    summary_df = pd.DataFrame([summary]).T
    summary_df.columns = ['Value']
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'cost_benefit_summary.csv'), float_format='%.4f')
    print("\n  Saved: cost_benefit_summary.csv")


def main():
    print("=" * 70)
    print("PHASE 6: COST-BENEFIT ANALYSIS OF FX INTERVENTION")
    print("=" * 70)
    
    df = load_data()
    
    cost_data = compute_carrying_cost(df)
    episode_results = crisis_episode_analysis(df)
    cost_benefit_summary(df, cost_data)
    
    print("\n" + "=" * 70)
    print("Phase 6 COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
