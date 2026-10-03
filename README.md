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
- USD/INR depreciated from ₹44 (2001) to ₹96 (2026), while NEER declined from ~157 to ~80 (summary statistics restricted to the 2001–2026 analysis sample)
- Oil prices are highly volatile (std ≈ $33/barrel) with major swings during GFC, COVID
- India's FX reserves grew from ~$42 billion (2001) to $617 billion (peak)

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

**Johansen cointegration:** Results saved to `output/johansen_cointegration.txt`. Both the Trace and Max-Eigenvalue tests indicate at least one cointegrating vector at 5% significance, consistent with a long-run equilibrium relationship among the I(1) variables.

**ADF test specification:** Level variables are tested with a constant-and-trend specification (`regression='ct'`) because macro series such as log(reserves) and log(NEER) exhibit clear deterministic trends; omitting the trend biases the test toward non-rejection of a unit root.

**Implication:** Mix of I(0) and I(1) justifies using the **ARDL bounds testing approach** (Pesaran, Shin & Smith, 2001).

### Phase 4: ARDL Model & Structural VAR (`ardl_svar_model.py`)

#### 4a. UECM / ARDL Bounds Test & Estimation

- Estimated Unrestricted Error Correction Models (UECM) with NEER and USD/INR as dependent variables
- The UECM is the error-correction reparametrisation of the ARDL model; statsmodels' `bounds_test()` method is available on the UECM results class
- Independent variables: Log(Oil), CPI, Log(EPU), Interest Differential, Log(Reserves)
- Full regression outputs in `output/ardl_results_NEER.txt` and `output/ardl_results_USDINR.txt`
- Bounds test results in `output/bounds_test_NEER.txt` and `output/bounds_test_USDINR.txt`

#### 4b. VAR(3) — Impulse Response Functions (IRFs)

A 5-variable VAR was estimated with Cholesky ordering:
`[Oil Returns, ΔCPI, ΔInterest Diff, ΔReserves, NEER/USD-INR Returns]`

Oil is ordered first (most exogenous for India as a price-taker in global oil markets).

**Key IRF findings:**
- Oil price shocks have a small, short-lived negative effect on USD/INR returns (higher oil → small INR appreciation in the short run), likely reflecting the global dollar effect (oil and the dollar tend to move in opposite directions)
- Reserve shocks have a noticeable contemporaneous impact on USD/INR, consistent with intervention having some effect on the exchange rate

#### 4c. Forecast Error Variance Decomposition (FEVD)

| Shock Source | NEER at 24m | USD/INR at 24m |
|-------------|------------|---------------|
| Oil Returns | 0.2% | 2.3% |
| ΔCPI | 1.7% | 3.0% |
| ΔInterest Diff | 1.2% | 1.3% |
| **ΔReserves** | **1.6%** | **15.3%** |
| Own shocks | 95.3% | 78.1% |

**Key finding:** Reserve changes explain **15.3% of USD/INR variance** — the single largest external contributor, much more than oil (2.3%) or inflation (3.0%). 

> **Note on Cholesky ordering:** The VAR uses the ordering `[Oil, ΔCPI, ΔInt.Diff, ΔReserves, USD/INR]`. With reserves ordered before USD/INR, any contemporaneous correlation between the two — including the mechanical valuation effect (dollar-reported reserves fall when the dollar strengthens) — is attributed to "reserve shocks." The 15.3% figure should therefore be interpreted as an upper bound on the contribution of genuine RBI intervention.

#### 4d. Granger Causality Tests

| Hypothesis | F-stat | p-value | Result |
|-----------|--------|---------|--------|
| ΔReserves → USD/INR Returns | 5.91 | **0.016** | **Granger-causes** |
| Oil Returns → ΔInterest Diff | 5.73 | **0.001** | **Granger-causes** |
| USD/INR → ΔReserves | 3.05 | 0.082 | No (marginal) |
| Oil → NEER | 0.69 | 0.634 | No |
| EPU → Exchange Rate | — | >0.29 | No |
| CPI → NEER | 1.41 | 0.241 | No |

**Key finding:** Using a single pre-specified lag (VAR AIC-optimal) for all tests to avoid multiple-testing inflation. Reserve changes Granger-cause USD/INR movements, and Oil Granger-causes interest rate differential changes, consistent with an oil → monetary policy → exchange rate transmission channel.

### Phase 5: GARCH & Intervention Analysis (`garch_intervention.py`)

#### 5a. GARCH(1,1) Volatility Models

| Exchange Rate | α (ARCH) | β (GARCH) | α+β | Interpretation |
|--------------|----------|-----------|-----|----------------|
| NEER | 0.061 | 0.892 | **0.953** | Highly persistent |
| USD/INR | 0.129 | 0.871 | **1.000** | Near-IGARCH (integrated) |
| Oil | 0.492 | 0.000 | 0.492 | Close to ARCH(1); β ≈ 0, limited GARCH persistence |

USD/INR volatility is near-IGARCH (α+β ≈ 1), meaning volatility shocks are permanent — they never fully die out. This provides strong theoretical justification for RBI's active volatility management.

#### 5b. Intervention Effectiveness — Level Effect

```
ΔUSD/INR = 0.005 − 0.260·ΔReserves − 0.018·ΔOil + 0.007·ΔEPU + 0.001·ΔCPI
                    (p<0.001)***      (p=0.038)**  (p<0.001)***  (p=0.361)

R² = 0.222, N = 302, HC1 robust standard errors
```

**Interpretation:** A 1% increase in reserves is associated with a **0.26% appreciation** of the INR (USD/INR falls). 

> **Caveats on the reserve coefficient:** This negative coefficient has at least two non-causal interpretations. First, *valuation effect*: India's reserves are reported in USD but partly held in euros, yen, and gold; when the dollar strengthens globally, those assets lose dollar value, so reported reserves fall at the same moment the rupee weakens — creating a mechanical negative correlation with no RBI action involved. Second, *reverse causality*: the RBI buys dollars when the rupee is *already* appreciating. Both effects push the coefficient negative regardless of whether intervention actually works. The coefficient should therefore not be read as a clean causal estimate.

> **Oil sign:** The oil coefficient (−0.018) is **negative**. Since the dependent variable is ΔUSD/INR (positive = depreciation), a negative sign means higher oil prices are associated with rupee *appreciation*. This is a global dollar effect — oil and the dollar tend to move in opposite directions — and does **not** mean India benefits from oil price increases.

#### 5c. Asymmetric Intervention Effects

| Direction | Coefficient | p-value | Meaning |
|-----------|------------|---------|---------|
| Reserve **Buying** (accumulation) | −0.173 | 0.0001 | Moderates appreciation |
| Reserve **Selling** (decumulation) | −0.422 | 0.0000 | Strongly fights depreciation |

**Key finding:** Reserve selling during crises is **2.4× more effective** than reserve buying in the raw regression. However, since the RBI intervenes precisely when the exchange rate is moving ("leaning against the wind"), the negative coefficient on buying and selling are endogenous to the direction of RBI action — not a clean measure of its causal effect.

#### 5d. Granger Causality — Intervention Direction

| Direction | F-stat | p-value | Result |
|-----------|--------|---------|--------|
| ΔReserves → USD/INR | 5.91 | **0.016** | RBI intervention **causes** exchange rate changes |
| NEER → ΔReserves | 8.99 | **0.003** | Exchange rate movements **trigger** intervention |

Both directions are significant, consistent with the RBI following a **reactive "leaning against the wind"** strategy — it intervenes in response to exchange rate movements, and those interventions then Granger-cause subsequent rate changes. Granger causality tests use a single VAR AIC-selected lag for all pairs.

### Phase 6: Cost-Benefit Analysis (`cost_benefit_analysis.py`)

#### 6a. Carrying Cost of Reserves

The quasi-fiscal cost = Reserves × (Sterilization cost − Return on reserves) / 12

Using Repo Rate as sterilization cost and Fed Funds Rate as return proxy.

> **Limitations of this formula:** (1) The cost is applied to the *entire* reserve stock, not just the portion built up through active intervention. (2) The Fed Funds rate proxies for return on all reserves, but reserves include longer-dated bonds and gold, which typically yield more. (3) The formula ignores currency valuation gains: the rupee depreciated roughly 3%/year over the sample, meaning dollar assets held as reserves gained in rupee terms, partially offsetting the interest spread. These factors together imply the \~USD 334 billion figure overstates the true net carrying cost.

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
| **2013 Taper Tantrum** | +12.5% depreciation | +$9.2 billion | Relied on monetary tightening and FCNR(B) dollar swap window |
| **2014-16 Oil Crash** | +12.2% depreciation | +$46.8 billion | Accumulated reserves (INR benefited from lower import bill) |
| **2018 EM Crisis** | +12.2% depreciation | −$28.1 billion | Sold to stem capital outflows |
| **2020 COVID** | +2.8% depreciation | +$55.8 billion | Massive accumulation during inflows |
| **2022 Russia-Ukraine** | +9.8% depreciation | −$92.2 billion | Largest-ever drawdown |

The RBI follows a broadly **countercyclical** pattern: accumulating reserves during calm periods and drawing them down during crises. The 2022 drawdown of $92.2 billion was unprecedented; note that a significant portion reflects **valuation losses** (non-dollar assets and gold losing dollar value as the USD strengthened globally), not all of it active reserve sales.

#### 6c. Overall Assessment

| Dimension | Assessment |
|-----------|-----------|
| **Cost** | USD 334 billion cumulative (upper-bound estimate; formula overstates cost by ignoring valuation gains and applying to the full stock) |
| **Level effect** | Negative and significant association between reserve changes and USD/INR — but causal interpretation is confounded by reverse causality and valuation effects |
| **Volatility** | Volatility regression uses stationary (first-differenced) EPU and interest differential; direct volatility-reducing effect is not statistically significant |
| **Crisis insurance** | The table documents what happened during 6 episodes; a formal counterfactual (what would have happened without intervention) is outside the scope of this analysis. Note that in 3 of the 6 episodes reserves *rose*, and the 2022 drawdown includes substantial valuation losses |
| **Reserve adequacy** | 10.9 months of import cover (well above 3-month minimum) |
| **Paper alignment** | Results are broadly consistent with Aktuğ & Rezghi (2026). However, that paper models an *oil-exporting* economy where oil windfalls build net foreign assets and reduce the risk premium. For India — a large oil *importer* — the mechanism runs in the opposite direction: oil shocks drain FX, raise import costs, and typically weaken the rupee. The NEER and FEVD results should be interpreted in this context |

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
│   ├── bounds_test_NEER.txt
│   ├── bounds_test_USDINR.txt
│   ├── johansen_cointegration.txt
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
