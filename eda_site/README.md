# Interactive EDA Web Application — AI Financial Alert Risk Scoring

Streamlit web application presenting the comprehensive exploratory data analysis, payment instrument dynamics, and feature engineering insights behind the **tobaskA+** model (final honest CV ROC-AUC **0.6533**).

## Quick Start (Run Locally)

From the project root:
```bash
pip install -r requirements.txt
streamlit run eda_site/app.py
```

Or from the `eda_site/` directory:
```bash
cd eda_site
pip install -r requirements.txt
streamlit run app.py
```

## Structure & Key Files

```
eda_site/
├── app.py                  Main Streamlit application (7 investigation sections)
├── style.css               Design system tokens, responsive layout, dark theme UI
├── client.js               Client-side interactions, timeline navigation, risk simulation
├── site_data/              Compact precomputed CSV & JSON (~450 KB) — app reads only this
├── assets/                 Team visual assets, logos, and photos
├── prepare_data.py         Pipeline computing summaries from raw train transactions
├── prepare_inspector.py    Precomputes alert inspector profiles and gains curves
├── requirements.txt        Streamlit (pinned), pandas, plotly
└── .streamlit/config.toml   Theme and server configuration
```

> **Note:** The heavy raw competition datasets (135+ MB) are **not** needed to run or deploy the site. All charts and interactive tools consume compact precomputed artifacts from `site_data/`.

## UI & Interactive Features: The Case File

The site is styled as a financial intelligence investigation case file:
- **7-Step Narrative:** Each section poses a key question, provides an immediate one-line conclusion, and displays interactive supporting evidence.
- **Adaptive Timeline Navigation:** Vertical case timeline on desktop; responsive thumb-friendly bottom dock on mobile devices.
- **Score Journey:** Interactive progression tracing the model's performance gains from baseline to the 0.6533 ceiling.
- **Payment Instrument Contrast Explorer:** Visualizing how the balance between cash deposits, card spending, and bank transfers drives escalation risk.
- **Real-Life Alert Inspector:** Deep-dive inspection tool covering 30 real alerts with transaction timeline scrubbing, dynamic quantiles, and risk breakdown.
- **Analyst Budget & Gains Curve:** Interactive simulation allowing security analysts to evaluate alert triage workload vs. captured escalated cases.
- **Theoretical Ceiling Calculator:** Demonstrating the mathematical upper bound of the transaction signal using the binormal model.
