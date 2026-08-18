"""
Phase 1: Data Preparation — Parse, Clean, Merge all CEIC data
=============================================================
Outputs:
  - Data/merged_monthly.csv  (analysis-ready panel)
  - Data/data_summary.txt    (quick overview of what was parsed)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data')
OUTPUT_CSV = os.path.join(DATA_DIR, 'merged_monthly.csv')

# ─────────────────────────────────────────────────────────────
# Helper: Parse CEIC-format CSV
# CEIC CSVs have ~29 header rows of metadata, then date,value pairs
# ─────────────────────────────────────────────────────────────

def parse_ceic_csv(filepath, date_format='%m/%Y'):
    """
    Parse a CEIC-exported CSV.
    Returns a Series indexed by datetime with the variable values.
    """
    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 2:
            continue
        date_str = parts[0].strip()
        val_str = parts[1].strip().strip('"')
        
        # Try parsing as date
        try:
            if date_format == 'daily':
                # Daily format: MM/DD/YYYY
                dt = pd.to_datetime(date_str, format='%m/%d/%Y')
            else:
                # Monthly format: MM/YYYY
                dt = pd.to_datetime(date_str, format='%m/%Y')
            val = float(val_str)
            rows.append((dt, val))
        except (ValueError, TypeError):
            continue
    
    if not rows:
        raise ValueError(f"No data parsed from {filepath}")
    
    df = pd.DataFrame(rows, columns=['date', 'value'])
    df = df.set_index('date').sort_index()
    # Remove duplicates (keep last)
    df = df[~df.index.duplicated(keep='last')]
    return df['value']


def main():
    print("=" * 70)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────
    # 1. Parse each file
    # ─────────────────────────────────────────────────────────
    
    print("\n[1/6] Parsing CEIC CSV files...")
    
    # (a) NEER — BIS Broad (monthly, 2020=100)
    neer = parse_ceic_csv(
        os.path.join(DATA_DIR, 'Nominal Effective Exchange Rate Index BIS 2020100 Broad.csv')
    )
    neer.name = 'NEER'
    print(f"  NEER:          {neer.index.min().strftime('%Y-%m')} to {neer.index.max().strftime('%Y-%m')} ({len(neer)} obs)")
    
    # (b) USD/INR — Daily → need to convert to monthly average
    usdinr_daily = parse_ceic_csv(
        os.path.join(DATA_DIR, 'Foreign Exchange Rate RBI Reference Rate US Dollar.csv'),
        date_format='daily'
    )
    usdinr_daily.name = 'USDINR_daily'
    print(f"  USD/INR daily: {usdinr_daily.index.min().strftime('%Y-%m-%d')} to {usdinr_daily.index.max().strftime('%Y-%m-%d')} ({len(usdinr_daily)} obs)")
    
    # (c) Brent Crude Oil (monthly, USD/barrel)
    oil = parse_ceic_csv(
        os.path.join(DATA_DIR, 'Commodity Price Nominal Energy Crude Oil Brent.csv')
    )
    oil.name = 'OIL_BRENT'
    print(f"  Oil Brent:     {oil.index.min().strftime('%Y-%m')} to {oil.index.max().strftime('%Y-%m')} ({len(oil)} obs)")
    
    # (d) CPI YoY (monthly, %)
    cpi = parse_ceic_csv(
        os.path.join(DATA_DIR, 'Consumer Price Index YoY Monthly India.csv')
    )
    cpi.name = 'CPI_YOY'
    print(f"  CPI YoY:       {cpi.index.min().strftime('%Y-%m')} to {cpi.index.max().strftime('%Y-%m')} ({len(cpi)} obs)")
    
    # (e) EPU Index (monthly)
    epu = parse_ceic_csv(
        os.path.join(DATA_DIR, 'Economic Policy Uncertainty Index India.csv')
    )
    epu.name = 'EPU'
    print(f"  EPU Index:     {epu.index.min().strftime('%Y-%m')} to {epu.index.max().strftime('%Y-%m')} ({len(epu)} obs)")
    
    # (f) FX Reserves (monthly, USD mn)
    reserves = parse_ceic_csv(
        os.path.join(DATA_DIR, 'Foreign Exchange Reserve USD Foreign Exchange.csv')
    )
    reserves.name = 'FX_RESERVES'
    print(f"  FX Reserves:   {reserves.index.min().strftime('%Y-%m')} to {reserves.index.max().strftime('%Y-%m')} ({len(reserves)} obs)")
    
    # (g) Repo Rate (monthly, % pa)
    repo = parse_ceic_csv(
        os.path.join(DATA_DIR, 'Policy Rate Month End India Repo Rate.csv')
    )
    repo.name = 'REPO_RATE'
    print(f"  Repo Rate:     {repo.index.min().strftime('%Y-%m')} to {repo.index.max().strftime('%Y-%m')} ({len(repo)} obs)")
    
    # (h) Fed Funds Rate (monthly, % pa)
    fedfunds = parse_ceic_csv(
        os.path.join(DATA_DIR, 'Policy Rate Month End Effective Federal Funds Rate.csv')
    )
    fedfunds.name = 'FED_RATE'
    print(f"  Fed Funds:     {fedfunds.index.min().strftime('%Y-%m')} to {fedfunds.index.max().strftime('%Y-%m')} ({len(fedfunds)} obs)")
    
    # (i) IIP YoY (monthly, %)
    iip = parse_ceic_csv(
        os.path.join(DATA_DIR, 'Industrial Production Index YoY Monthly India.csv')
    )
    iip.name = 'IIP_YOY'
    print(f"  IIP YoY:       {iip.index.min().strftime('%Y-%m')} to {iip.index.max().strftime('%Y-%m')} ({len(iip)} obs)")
    
    # (j) REER — BIS Broad (monthly, 2020=100)
    reer_bis = parse_ceic_csv(
        os.path.join(DATA_DIR, 'Real Effective Exchange Rate Index BIS 2020100 Broad.csv')
    )
    reer_bis.name = 'REER_BIS'
    print(f"  REER BIS:      {reer_bis.index.min().strftime('%Y-%m')} to {reer_bis.index.max().strftime('%Y-%m')} ({len(reer_bis)} obs)")
    
    # (k) REER — RBI (monthly, 2005=100)
    reer_rbi = parse_ceic_csv(
        os.path.join(DATA_DIR, 'REER 2005100 Month Avg India.csv')
    )
    reer_rbi.name = 'REER_RBI'
    print(f"  REER RBI:      {reer_rbi.index.min().strftime('%Y-%m')} to {reer_rbi.index.max().strftime('%Y-%m')} ({len(reer_rbi)} obs)")
    
    # (l) REER — CPI Based (monthly, 2015=100)
    reer_cpi = parse_ceic_csv(
        os.path.join(DATA_DIR, 'Real Effective Exchange Rate Index CPI Based.csv')
    )
    reer_cpi.name = 'REER_CPI'
    print(f"  REER CPI:      {reer_cpi.index.min().strftime('%Y-%m')} to {reer_cpi.index.max().strftime('%Y-%m')} ({len(reer_cpi)} obs)")
    
    # ─────────────────────────────────────────────────────────
    # 2. Convert daily USD/INR to monthly average
    # ─────────────────────────────────────────────────────────
    
    print("\n[2/6] Converting daily USD/INR to monthly average...")
    usdinr_monthly = usdinr_daily.resample('MS').mean()
    usdinr_monthly.name = 'USDINR'
    print(f"  USD/INR monthly: {usdinr_monthly.index.min().strftime('%Y-%m')} to {usdinr_monthly.index.max().strftime('%Y-%m')} ({len(usdinr_monthly)} obs)")
    
    # Also compute monthly std dev (intra-month volatility)
    usdinr_vol = usdinr_daily.resample('MS').std()
    usdinr_vol.name = 'USDINR_MONTHLY_VOL'
    
    # ─────────────────────────────────────────────────────────
    # 3. Merge all monthly series
    # ─────────────────────────────────────────────────────────
    
    print("\n[3/6] Merging all series to common monthly index...")
    
    # Ensure all series have month-start index for alignment
    all_series = {
        'NEER': neer,
        'USDINR': usdinr_monthly,
        'USDINR_MONTHLY_VOL': usdinr_vol,
        'OIL_BRENT': oil,
        'CPI_YOY': cpi,
        'EPU': epu,
        'FX_RESERVES': reserves,
        'REPO_RATE': repo,
        'FED_RATE': fedfunds,
        'IIP_YOY': iip,
        'REER_BIS': reer_bis,
        'REER_RBI': reer_rbi,
        'REER_CPI': reer_cpi,
    }
    
    # Normalize all indices to month-start
    normalized = {}
    for name, s in all_series.items():
        s_copy = s.copy()
        # Normalize to first day of each month
        s_copy.index = s_copy.index.to_period('M').to_timestamp()
        s_copy = s_copy[~s_copy.index.duplicated(keep='last')]
        normalized[name] = s_copy
    
    merged = pd.DataFrame(normalized)
    merged.index.name = 'date'
    merged = merged.sort_index()
    
    print(f"  Full merged shape: {merged.shape}")
    print(f"  Date range: {merged.index.min().strftime('%Y-%m')} to {merged.index.max().strftime('%Y-%m')}")
    print(f"  Missing values per column:")
    for col in merged.columns:
        valid = merged[col].notna().sum()
        total = len(merged)
        print(f"    {col:25s}: {valid:4d} / {total:4d} valid ({100*valid/total:.1f}%)")
    
    # ─────────────────────────────────────────────────────────
    # 4. Compute derived variables
    # ─────────────────────────────────────────────────────────
    
    print("\n[4/6] Computing derived variables...")
    
    # Log levels (for unit root tests and cointegration)
    merged['LOG_NEER'] = np.log(merged['NEER'])
    merged['LOG_USDINR'] = np.log(merged['USDINR'])
    merged['LOG_OIL'] = np.log(merged['OIL_BRENT'])
    merged['LOG_RESERVES'] = np.log(merged['FX_RESERVES'])
    merged['LOG_EPU'] = np.log(merged['EPU'])
    
    # Returns / first differences
    merged['D_LOG_NEER'] = merged['LOG_NEER'].diff()           # NEER monthly return
    merged['D_LOG_USDINR'] = merged['LOG_USDINR'].diff()       # USD/INR monthly return
    merged['D_LOG_OIL'] = merged['LOG_OIL'].diff()             # Oil price monthly return
    merged['D_LOG_RESERVES'] = merged['LOG_RESERVES'].diff()   # Reserve growth
    merged['D_LOG_EPU'] = merged['LOG_EPU'].diff()             # EPU change
    
    # Month-on-month change in reserves (level, USD mn)
    merged['DELTA_RESERVES'] = merged['FX_RESERVES'].diff()
    
    # Interest rate differential (India - US)
    merged['INT_DIFF'] = merged['REPO_RATE'] - merged['FED_RATE']
    
    # Change in interest rate differential
    merged['D_INT_DIFF'] = merged['INT_DIFF'].diff()
    
    # Change in CPI
    merged['D_CPI'] = merged['CPI_YOY'].diff()
    
    # Oil price shock: absolute change > 2 std devs
    oil_ret_std = merged['D_LOG_OIL'].std()
    merged['OIL_SHOCK'] = (merged['D_LOG_OIL'].abs() > 2 * oil_ret_std).astype(int)
    
    # Net Oil Price Increase (Hamilton, 2003): 
    # max(0, oil_t - max(oil_{t-1},...,oil_{t-12}))
    merged['OIL_MAX_12'] = merged['OIL_BRENT'].rolling(window=12).max().shift(1)
    merged['NOPI'] = np.maximum(0, merged['OIL_BRENT'] - merged['OIL_MAX_12'])
    
    print(f"  Added {len([c for c in merged.columns if c not in all_series])} derived columns")
    print(f"  Total columns: {len(merged.columns)}")
    
    # ─────────────────────────────────────────────────────────
    # 5. Define sample periods
    # ─────────────────────────────────────────────────────────
    
    print("\n[5/6] Defining sample periods...")
    
    # Core analysis columns
    core_cols = ['NEER', 'USDINR', 'OIL_BRENT', 'CPI_YOY', 'EPU', 
                 'FX_RESERVES', 'REPO_RATE', 'FED_RATE']
    
    # Sample A: Without IIP (longer)
    sample_a = merged[core_cols].dropna()
    print(f"  Sample A (without IIP): {sample_a.index.min().strftime('%Y-%m')} to {sample_a.index.max().strftime('%Y-%m')} ({len(sample_a)} obs)")
    
    # Sample B: With IIP (shorter)
    core_cols_b = core_cols + ['IIP_YOY']
    sample_b = merged[core_cols_b].dropna()
    print(f"  Sample B (with IIP):    {sample_b.index.min().strftime('%Y-%m')} to {sample_b.index.max().strftime('%Y-%m')} ({len(sample_b)} obs)")
    
    # ─────────────────────────────────────────────────────────
    # 6. Export
    # ─────────────────────────────────────────────────────────
    
    print(f"\n[6/6] Exporting merged data to {OUTPUT_CSV}...")
    merged.to_csv(OUTPUT_CSV)
    print(f"  Saved {merged.shape[0]} rows x {merged.shape[1]} columns")
    
    # Also save a quick summary
    summary_path = os.path.join(DATA_DIR, 'data_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("UGP DATA SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total date range: {merged.index.min().strftime('%Y-%m')} to {merged.index.max().strftime('%Y-%m')}\n")
        f.write(f"Total rows: {merged.shape[0]}\n")
        f.write(f"Total columns: {merged.shape[1]}\n\n")
        f.write("Sample A (without IIP):\n")
        f.write(f"  {sample_a.index.min().strftime('%Y-%m')} to {sample_a.index.max().strftime('%Y-%m')} ({len(sample_a)} obs)\n\n")
        f.write("Sample B (with IIP):\n")
        f.write(f"  {sample_b.index.min().strftime('%Y-%m')} to {sample_b.index.max().strftime('%Y-%m')} ({len(sample_b)} obs)\n\n")
        f.write("Columns:\n")
        for col in merged.columns:
            f.write(f"  {col}\n")
    
    print(f"  Summary saved to {summary_path}")
    print("\n" + "=" * 70)
    print("Phase 1 COMPLETE!")
    print("=" * 70)
    
    return merged


if __name__ == '__main__':
    merged = main()
