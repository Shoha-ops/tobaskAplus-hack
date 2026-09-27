# AI Financial Alert Risk Scoring — Fintech Track

> **Team:** tobaskA+  
> **Metric:** ROC-AUC (Single final submission — evaluated with honest out-of-fold validation and zero test leakage)

---


### Model Leaderboard (Honest CV)

| Model Architecture | Honest CV ROC-AUC | Description / Strategic Role |
|---|:---:|---|
| CatBoost (depth 4) | **0.6492** | Shallow regularized trees avoiding scale overfitting |
| Hybrid Booster (LightGBM) | **0.6477** | LightGBM initialized with linear model log-odds (`init_score`) |
| CatBoost (depth 6) | **0.6474** | Captures non-linear instrument volume interactions |
| LightGBM (tuned) | **0.6447** | Fast leaf-wise tree baseline |
| Logistic Regression (8 instruments) | **0.6413** | Direct linear plane on normalized instrument means |
| **Final Ensemble (Equal-weight Rank)** | **`0.6533`** | **Blends linear hyperplanes and tree partitions** |

*Ensemble Selection Rule:* Fixed prior to evaluating test outcomes: simple rank average of all diverse model families achieving honest CV ROC-AUC > 0.640.

---

## Key Technical Discovery: Customer Scale vs. Instrument Contrast

The primary challenge in the dataset is separating raw customer activity scale from true escalation risk.

1. **`miqdor_indeksi` Nature:** The amount index behaves as a standardized log-amount with distinct baselines across payment instruments (`karta_kirim` ≈ -0.54 up to `xalqaro` ≈ +2.17).
2. **The "Scale Trap" (62% of variance, AUC 0.535):** Customer mean amounts across all 8 instrument types (`type × direction`) have strong mutual correlations (0.50 – 0.88). Wealthier or higher-volume clients simply move larger sums across all channels. However, overall transaction volume/scale has almost zero predictive power for alert escalation.
3. **The Pure Signal Contrast (4% of variance, AUC 0.617 – 0.638):**
   The entire predictive signal lies in a subtle structural contrast:
   $$\text{Escalation Risk} \uparrow \iff \frac{\text{Cash Deposits (naqd-kirim)} + \text{Card Spending (karta-chiqim)}}{\text{Bank Transfers In (bank-kirim)} + \text{Bank Transfers Out (bank-chiqim)}} \uparrow$$
   - A simple logistic regression trained **solely on the 8 instrument means** scores **0.633** AUC.
   - The contrast formulated directly as a single non-parametric ratio (80th percentile) achieves **0.638** AUC without training.
   - Standard decision trees struggle to isolate this diagonal boundary because splits are dominated by the useless customer-scale axis. Combining gradient boosting with linear priors (our **Hybrid Booster**) bridges this gap.

---

## Theoretical Limit & Signal Ceiling Analysis

To understand whether further feature engineering could yield improvements, we modeled performance as a function of transaction history fraction $f \in [1/8, 1]$ per alert.

Using the binormal projection formula:
$$\text{AUC}(f) = \Phi\left(z_\infty \cdot \sqrt{\frac{f}{f + k}}\right)$$
The empirical curve fits the model with error $\le 0.003$, yielding an asymptotic upper bound at infinite history of:
$$\mathbf{AUC_\infty \approx 0.652 - 0.657}$$
Our ensemble score of **0.6533** operates directly at this theoretical boundary. Residual analysis confirmed that calibrated ensemble residuals exhibit no statistically significant correlation with any of 371 candidate temporal, sequence, or structural features.

---

## Leakage Prevention & Audit Trail

All checks are documented in [`experiments/AUDIT.md`](file:///d:/hack/experiments/AUDIT.md) and [`experiments/results.csv`](file:///d:/hack/experiments/results.csv):

1. **Permutation Baseline:** Shuffling target labels in the pipeline collapses CV to **0.48 – 0.51** (pure noise floor), confirming no label leakage in feature extraction.
2. **Fold Discipline:** Disjoint 5-fold Stratified CV. Early stopping is performed on a reserved 20% validation split inside each training fold (preventing the +0.005 optimistic bias seen when stopping on test folds).
3. **Pre-processing Isolation:** Instrument normalization parameters, imputation medians, and linear model scalers are computed strictly on training folds and frozen for validation/test.
4. **Held-out Generalization:** Retuning features and hyperparameters on 10,000 alerts and evaluating on 4,000 isolated alerts yielded **0.639** (95% CI: [0.617, 0.662]), indicating an overfitting gap $\le 0.004$ over 35 experiments.
5. **Pre-Alert Burst Handling (`signal_sanasi`):**
   - 840 train transactions (21 alerts) and 51 test transactions (2 alerts) occur on the date of `signal_sanasi`.
   - `signal_sanasi` provides only a calendar date (`YYYY-MM-DD` at 00:00:00) without exact alert timestamp.
   - For 99% of alerts, the 180-day history ends with a ~3-minute trigger burst immediately before 00:00. For the remaining 21 train / 2 test alerts, their entire history is shifted by 1 day and ends with an identical 3-minute burst before 24:00 of `signal_sanasi`.
   - These transactions represent legitimate pre-alert triggers, not future data. Processing is identical between train and test.

---

## Negative Results (What Did NOT Work)

Systematic experimentation confirmed that temporal dynamics and sequence order in this dataset contain no signal:

| Tested Hypothesis | CV Impact ($\Delta$) | Technical Reason |
|---|:---:|---|
| Temporal velocity windows (7 / 30 / 90 days) | **-0.0006** | Activity distribution across time is identical between classes |
| Day of week / hour / night transaction share | **-0.0017** | Timestamps follow uniform synthetic distribution (1/7, 1/24) |
| Sequence transitions & Markov chains | **-0.0009** | Instrument ordering is statistically independent (entropy 0.993 vs 1.006) |
| Float amount precision micro-clusters | **+0.0003** | High variance across random seeds; artifact of float32 rounding |
| Multiple Instance Learning (MIL) on raw txs | **0.5560** | Fails to capture global customer balance ratios |
| Feature selection via GBDT feature importance | **-0.0060** | Overfits training fold split points |

---

---

## Repository Structure

```
.
├── README.md                      # Comprehensive project documentation (this file)
├── requirements.txt               # Pinned dependencies for model and web app
├── train_final.py                 # Self-contained training script (reproduces final submission)
├── src/                           # Production source modules
│   ├── config.py                  # Paths, constants, seed list, evaluation metrics
│   ├── data.py                    # Optimized memory-mapped parquet loaders
│   ├── features.py                # Base customer summary aggregates
│   ├── features_v4.py             # Payment instrument level statistics (primary signal)
│   ├── features_v7.py             # Customer scale vs. balance decomposition
│   ├── features_v8.py             # Explicit instrument contrast features
│   ├── hybrid.py                  # Hybrid booster (LightGBM with linear model init_score)
│   └── validation.py              # Leak-safe 5-fold Stratified CV runner
├── notebooks/
│   ├── solution.ipynb             # Fully executed, reproducible Jupyter Notebook
│   └── build_notebook.py          # Script compiling source modules into the single notebook
├── submissions/
│   └── team_3B832E89.csv          # Final submitted predictions (md5 a0d1cf0afc9be6c039ad19590cfd26c9)
├── experiments/
│   ├── AUDIT.md                   # Formal validation and audit verification checklist
│   ├── results.csv                # Historical experiment tracking log
│   ├── hypotheses.csv             # Structured hypothesis test results
│   └── stability_nofilter.json    # Multi-seed stability analysis
└── eda_site/                      # Streamlit EDA web application
    ├── app.py                     # Main web application entry point
    ├── site_data/                 # Precomputed compact datasets (~450 KB)
    ├── style.css                  # UI design tokens and responsive styles
    └── requirements.txt           # Minimal web app dependencies
```

---

