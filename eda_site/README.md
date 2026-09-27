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