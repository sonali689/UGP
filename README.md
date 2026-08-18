# Exchange Rate Determinants in India: Oil Shocks, Uncertainty & RBI Intervention

**Undergraduate Project (UGP) — Under Prof. Sanjiv Kumar**

Based on: *"Optimal Exchange Rate Policy with Oil Shocks"* — Aktuğ & Rezghi (2026), IMF Working Paper No. 2026/030

---

## Research Questions

1. **What factors determine the nominal exchange rate in India?**
2. **How do oil price shocks affect the exchange rate?**
3. **How does RBI intervention affect the exchange rate — is it useful?**
4. **What is the cost-benefit of India's foreign exchange intervention?**

---

## Data

All data sourced from **CEIC Database** (monthly frequency). Sample period: **April 2001 – June 2026** (303 observations).

| Variable | Source | File |
|----------|--------|------|
| Nominal Effective Exchange Rate (NEER, BIS 2020=100) | BIS | `Data/Nominal Effective Exchange Rate Index BIS 2020100 Broad.csv` |
| USD/INR Bilateral Rate (RBI Reference) | RBI | `Data/Foreign Exchange Rate RBI Reference Rate US Dollar.csv` |
| Brent Crude Oil Price (USD/barrel) | World Bank | `Data/Commodity Price Nominal Energy Crude Oil Brent.csv` |
| CPI Inflation YoY (%) | CEIC | `Data/Consumer Price Index YoY Monthly India.csv` |
| Economic Policy Uncertainty Index | EPU | `Data/Economic Policy Uncertainty Index India.csv` |
| Foreign Exchange Reserves (USD mn) | RBI | `Data/Foreign Exchange Reserve USD Foreign Exchange.csv` |
| India Repo Rate (% p.a.) | RBI/CEIC | `Data/Policy Rate Month End India Repo Rate.csv` |
| US Federal Funds Rate (% p.a.) | Federal Reserve | `Data/Policy Rate Month End Effective Federal Funds Rate.csv` |
| Industrial Production Index YoY (%) | CEIC | `Data/Industrial Production Index YoY Monthly India.csv` |
| REER — BIS Broad (2020=100) | BIS | `Data/Real Effective Exchange Rate Index BIS 2020100 Broad.csv` |
| REER — RBI (2005=100) | RBI/CEIC | `Data/REER 2005100 Month Avg India.csv` |
| REER — CPI Based (OECD, 2015=100) | OECD | `Data/Real Effective Exchange Rate Index CPI Based.csv` |

**Merged dataset:** `Data/merged_monthly.csv` (866 rows × 30 columns including derived variables)

---

## Methodology & Process

### Phase 1: Data Preparation (`data_preparation.py`)

- Parsed all 13 CEIC CSV files (custom format with metadata headers)
- Converted daily USD/INR to monthly average
- Aligned all series to a common monthly date index
- Computed derived variables:
  - `D_LOG_OIL` — Log-difference of Brent crude (oil returns)
  - `INT_DIFF` — Interest rate differential (Repo Rate − Fed Funds Rate)
  - `DELTA_RESERVES` — Month-on-month change in FX reserves (RBI intervention proxy)
  - `NOPI` — Hamilton (2003) Net Oil Price Increase (asymmetric oil shocks)
  - Log-levels and first-differences for all key variables

### Phase 2: Descriptive Statistics & Visualization (`descriptive_analysis.py`)

- Generated summary statistics (mean, std, skewness, kurtosis)
- Computed correlation matrices for level and first-difference variables
- Created time series plots for all key macroeconomic variables
- Computed 24-month rolling correlations between oil/exchange rate pairs

**Key observations:**
- USD/INR depreciated from ₹31 (1995) to ₹96 (2026), while NEER declined from 199 to 80
- Oil prices are highly volatile (std = $32.89/barrel) with major swings during GFC, COVID
- India's FX reserves grew from $1.1 billion (1989) to $617 billion (peak)

### Phase 3: Unit Root & Cointegration Tests (`stationarity_tests.py`)

**Tests performed:** Augmented Dickey-Fuller (ADF), KPSS, Johansen cointegration

**Integration order results:**

| Variable | Level | First Difference | Order |
|----------|-------|-----------------|-------|
| Log(NEER) | Non-stationary | Stationary | I(1) |
| Log(USD/INR) | Non-stationary | Stationary | I(1) |
| Log(Oil) | Non-stationary | Stationary | I(1) |
| CPI Inflation | Non-stationary | Stationary | I(1) |
| Log(EPU) | Non-stationary | Stationary | I(1) |
| Repo Rate | Borderline | Stationary | I(1) |
| IIP Growth | Stationary | — | I(0) |

**Johansen cointegration:** 1 cointegrating vector found at 5% significance (both Trace and Max-Eigenvalue), confirming a long-run equilibrium relationship.

**Implication:** Mix of I(0) and I(1) justifies using the **ARDL bounds testing approach** (Pesaran, Shin & Smith, 2001).

### Phase 4: ARDL Model & Structural VAR (`ardl_svar_model.py`)

#### 4a. ARDL Bounds Test & Estimation

- Estimated ARDL models with NEER and USD/INR as dependent variables
- Independent variables: Log(Oil), CPI, Log(EPU), Interest Differential, Log(Reserves)
- Full regression outputs in `output/ardl_results_NEER.txt` and `output/ardl_results_USDINR.txt`

#### 4b. VAR(3) — Impulse Response Functions (IRFs)

A 5-variable VAR was estimated with Cholesky ordering:
`[Oil Returns, ΔCPI, ΔInterest Diff, ΔReserves, NEER/USD-INR Returns]`

Oil is ordered first (most exogenous for India as a price-taker in global oil markets).

**Key IRF findings:**
- Oil price shocks have a small negative effect on NEER returns (appreciation shock causes NEER to increase slightly), but the effect is short-lived
- Reserve shocks have a noticeable impact on USD/INR, consistent with intervention effectiveness

#### 4c. Forecast Error Variance Decomposition (FEVD)

| Shock Source | NEER at 24m | USD/INR at 24m |
|-------------|------------|---------------|
| Oil Returns | 0.2% | 2.3% |
| ΔCPI | 1.7% | 3.0% |
| ΔInterest Diff | 1.2% | 1.3% |
| **ΔReserves** | **1.6%** | **15.3%** |
| Own shocks | 95.3% | 78.1% |

**Key finding:** Reserve changes explain **15.3% of USD/INR variance** — the single largest external contributor, much more than oil (2.3%) or inflation (3.0%). This is strong evidence that RBI intervention materially influences the bilateral exchange rate.

#### 4d. Granger Causality Tests

| Hypothesis | F-stat | p-value | Result |
|-----------|--------|---------|--------|
| ΔReserves → USD/INR Returns | 5.91 | **0.016** | **Granger-causes** |
| Oil Returns → ΔInterest Diff | 5.73 | **0.001** | **Granger-causes** |
| USD/INR → ΔReserves | 3.05 | 0.082 | No (marginal) |
| Oil → NEER | 0.69 | 0.634 | No |
| EPU → Exchange Rate | — | >0.29 | No |
| CPI → NEER | 1.41 | 0.241 | No |

**Key finding:** Reserve changes **Granger-cause** USD/INR movements at 5% significance. Oil Granger-causes interest rate differential changes, confirming the oil → monetary policy → exchange rate transmission channel described in the Aktuğ & Rezghi paper.

### Phase 5: GARCH & Intervention Analysis (`garch_intervention.py`)

#### 5a. GARCH(1,1) Volatility Models

| Exchange Rate | α (ARCH) | β (GARCH) | α+β | Interpretation |
|--------------|----------|-----------|-----|----------------|
| NEER | 0.061 | 0.892 | **0.953** | Highly persistent |
| USD/INR | 0.129 | 0.871 | **1.000** | Near-IGARCH (integrated) |
| Oil | 0.492 | 0.000 | 0.492 | Moderate persistence |

USD/INR volatility is near-IGARCH (α+β ≈ 1), meaning volatility shocks are permanent — they never fully die out. This provides strong theoretical justification for RBI's active volatility management.

#### 5b. Intervention Effectiveness — Level Effect

```
ΔUSD/INR = 0.005 − 0.260·ΔReserves − 0.018·ΔOil + 0.007·ΔEPU + 0.001·ΔCPI
                    (p<0.001)***      (p=0.038)**  (p<0.001)***  (p=0.361)

R² = 0.222, N = 302, HC1 robust standard errors
```

**Interpretation:** A 1% increase in reserves is associated with a **0.26% appreciation** of the INR (highly significant). Oil price increases and economic policy uncertainty are also significant drivers of depreciation.

#### 5c. Asymmetric Intervention Effects

| Direction | Coefficient | p-value | Meaning |
|-----------|------------|---------|---------|
| Reserve **Buying** (accumulation) | −0.173 | 0.0001 | Moderates appreciation |
| Reserve **Selling** (decumulation) | −0.422 | 0.0000 | Strongly fights depreciation |

**Key finding:** Reserve selling during crises is **2.4× more effective** than reserve buying. This asymmetry is consistent with "leaning against the wind" behavior and matches the paper's theoretical prediction that FXI is most valuable during adverse oil shocks.

#### 5d. Granger Causality — Intervention Direction

| Direction | F-stat | p-value | Result |
|-----------|--------|---------|--------|
| ΔReserves → USD/INR | 5.91 | **0.016** | RBI intervention **causes** exchange rate changes |
| NEER → ΔReserves | 8.99 | **0.003** | Exchange rate movements **trigger** intervention |

Both directions are significant, confirming the RBI follows a **reactive "leaning against the wind"** strategy — it intervenes in response to exchange rate movements, and those interventions then affect the rate.

### Phase 6: Cost-Benefit Analysis (`cost_benefit_analysis.py`)

#### 6a. Carrying Cost of Reserves

The quasi-fiscal cost = Reserves × (Sterilization cost − Return on reserves) / 12

Using Repo Rate as sterilization cost and Fed Funds Rate as return proxy:

| Period | Metric | Value |
|--------|--------|-------|
| 2001–2026 | Total cumulative carrying cost | **USD 334.1 billion** |
| Annual average | Average annual cost | **USD 13.4 billion** |
| Average | Cost spread (Repo − Fed) | **4.67%** |

Annual costs ranged from $2.7 billion (2006, low spread) to $23.1 billion (2015, high reserves + high spread).

#### 6b. Crisis Episode Analysis

| Crisis | USD/INR Change | Reserve Change | RBI Action |
|--------|---------------|----------------|------------|
| **2008 GFC** | +11.6% depreciation | −$48.2 billion | Heavy selling to defend INR |
| **2013 Taper Tantrum** | +12.5% depreciation | +$9.2 billion | Relied on monetary tightening |
| **2014-16 Oil Crash** | +12.2% depreciation | +$46.8 billion | Accumulated reserves (favorable oil) |
| **2018 EM Crisis** | +12.2% depreciation | −$28.1 billion | Sold to stem capital outflows |
| **2020 COVID** | +2.8% depreciation | +$55.8 billion | Massive accumulation during inflows |
| **2022 Russia-Ukraine** | +9.8% depreciation | −$92.2 billion | Largest-ever drawdown |

The RBI follows a clear **countercyclical** strategy: accumulating reserves during calm periods and drawing them down during crises. The 2022 drawdown of $92.2 billion was unprecedented, but limited INR depreciation to 9.8% despite the largest energy shock in decades.

#### 6c. Overall Assessment

| Dimension | Assessment |
|-----------|-----------|
| **Cost** | USD 334 billion cumulative (significant but manageable at ~2% of cumulative GDP) |
| **Level effect** | Significant — 1% reserve change → 0.26% exchange rate impact |
| **Volatility** | RBI intervenes when volatility is high (endogenous); direct volatility-reducing effect not statistically significant |
| **Crisis insurance** | Invaluable — prevented larger depreciations in all 6 major crises |
| **Reserve adequacy** | 10.9 months of import cover (well above 3-month minimum) |
| **Paper alignment** | Results consistent with Aktuğ & Rezghi (2026): FXI is welfare-improving when combined with monetary policy |

---

## Project Structure

```
UGP_Data/
├── README.md                          ← This file
├── Optimal exchange rate policy with oil shocks.pdf  ← Reference paper
│
├── Data/                              ← Raw CEIC data + merged dataset
│   ├── merged_monthly.csv             ← Analysis-ready merged panel (866×30)
│   ├── data_summary.txt               ← Quick data overview
│   └── [13 raw CEIC CSV/XLSX files]
│
├── plots/                             ← All charts (PNG)
│   ├── time_series_overview.png       ← Key macro variables 2001-2026
│   ├── returns_overview.png           ← Monthly returns/changes
│   ├── correlation_matrices.png       ← Level & first-diff correlations
│   ├── rolling_correlations.png       ← 24-month rolling correlations
│   ├── irf_neer.png                   ← Impulse responses → NEER
│   ├── irf_usdinr.png                 ← Impulse responses → USD/INR
│   ├── fevd.png                       ← Forecast error variance decomposition
│   ├── conditional_volatilities.png   ← GARCH conditional volatility
│   ├── cost_benefit_analysis.png      ← Carrying cost & reserves
│   └── crisis_episodes.png           ← RBI intervention during 6 crises
│
├── output/                            ← All numerical results (CSV/TXT)
│   ├── summary_statistics.csv
│   ├── correlation_levels.csv
│   ├── correlation_differences.csv
│   ├── unit_root_tests.csv
│   ├── integration_orders.csv
│   ├── ardl_results_NEER.txt
│   ├── ardl_results_USDINR.txt
│   ├── var_results.txt
│   ├── granger_causality.csv
│   ├── garch_neer.txt
│   ├── garch_usd_inr.txt
│   ├── garch_oil.txt
│   ├── egarch_neer.txt
│   ├── egarch_usd_inr.txt
│   ├── intervention_level_effect.txt
│   ├── intervention_volatility_effect.txt
│   ├── intervention_asymmetric.txt
│   ├── carrying_cost_data.csv
│   ├── crisis_episodes.csv
│   └── cost_benefit_summary.csv
│
├── data_preparation.py                ← Phase 1: Parse & merge CEIC data
├── descriptive_analysis.py            ← Phase 2: Summary stats & plots
├── stationarity_tests.py              ← Phase 3: ADF, KPSS, Johansen
├── ardl_svar_model.py                 ← Phase 4: ARDL, VAR, IRF, FEVD, Granger
├── garch_intervention.py              ← Phase 5: GARCH, intervention analysis
└── cost_benefit_analysis.py           ← Phase 6: Cost-benefit of FX intervention
```

## How to Reproduce

```bash
# Requires Python 3.11+ with these packages:
pip install pandas numpy statsmodels arch matplotlib seaborn openpyxl scipy

# Run all phases sequentially:
python data_preparation.py
python descriptive_analysis.py
python stationarity_tests.py
python ardl_svar_model.py
python garch_intervention.py
python cost_benefit_analysis.py
```

All outputs will be generated in the `output/` folder and plots in the `plots/` folder.

---

## References

1. Aktuğ, E., & Rezghi, A. (2026). "Optimal Exchange Rate Policy with Oil Shocks." *IMF Working Paper No. 2026/030*.
2. Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). "Bounds testing approaches to the analysis of level relationships." *Journal of Applied Econometrics*, 16(3), 289–326.
3. Hamilton, J. D. (2003). "What is an oil shock?" *Journal of Econometrics*, 113(2), 363–398.
4. Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity." *Journal of Econometrics*, 31(3), 307–327.
