# EDA website — AI Financial Alert Risk Scoring

Streamlit app presenting our exploratory data analysis and how it shaped the model
(final honest CV ROC-AUC 0.6533).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
```
eda_site/
├── app.py              the website (7 sections)
├── style.css           design tokens, navigation, components, motion
├── client.js           instant switching, timeline + dock, journey replay, simulator, SVG insertion
├── prepare_data.py     computes every number/chart input from the RAW train data
├── prepare_inspector.py samples 30 alerts & precomputes gains curve -> inspector.json, gains.json
├── assets/           team logo: logo.svg (mark), logo-lockup.svg, logo-512.png, favicon.png
│                     (make_logo_png.py renders the PNGs from the same geometry)
├── site_data/          small precomputed CSV/JSON (~450 KB) — the only data the app reads
├── requirements.txt    streamlit (pinned), pandas, plotly
└── .streamlit/config.toml   theme + server settings
```
The raw competition data (135 MB) is NOT needed to run or deploy the site.
To regenerate `site_data/` from raw data, run from the project root:
`python3 eda_site/prepare_data.py && python3 eda_site/prepare_inspector.py` (needs `src/` and `data/`).

## UI: a case file
* **Concept** - the site reads like the file of an investigation: 7 steps, each a question, a one-line
  answer and the evidence. Outlined step numbers, cyan answer strip, cool "monitoring wall" background.
* **Navigation** - a vertical *case timeline* on the left: the line fills up to the current step, visited
  steps are marked, the current one shows its question. On phones and tablets it becomes a thumb-friendly
  bottom dock (prev / 03 of 07 / next + progress) that opens the same timeline as a bottom sheet.
  Keys 1-7 and arrow keys, #s3-style deep links.
* **Interactive** (all client-side, no server round-trip):
  score journey (click a stage or replay the run - autoplays once), clickable data schema, 100-square
  waffle, chart tabs instead of side-by-side chart pairs, PCA component recipe with 8-instrument loading
  bars (PC1–PC6, default PC5), alert risk simulator, real-life alert inspector with 30 cases, time-slider
  history scrubbing, dynamic q75 and contrast recalculation, analyst budget with gains curve and inspection
  effort slider, observation -> decision explorer, model leaderboard, expandable rejected ideas and
  audit checks, ceiling explorer with a stepped slider, count-up numbers, Plotly hover everywhere.
* **Phones** - every grid collapses to one column, tables turn into cards, chart tabs fill the width,
  charts do not capture drag, no horizontal scrolling on any step (checked at 390 px).
* **Loading** - a splash with the team mark shows how many steps have arrived and fades out as soon as
  the current step is ready (a CSS fallback hides it after 12 s even without JavaScript).
* **Evidence links** - dotted, ↗-marked phrases (main finding, model explorer, conclusion) open the step,
  switch to the right chart tab, scroll to it and flash it. Key charts carry their take-away as a callout.
* **Swipe** left / right on phones = next / previous step; steps slide in from the direction of travel.
* Inline SVG is removed by st.html's sanitiser, so SVG charts travel in a data-svg attribute and
  client.js inserts them.

## Why it reacts instantly
A standard Streamlit app re-runs the whole script and redraws the page on every click
(≈150–500 ms locally, plus the network round-trip to the server when deployed).
This site never does that after the first load:

* **All 7 sections are rendered once** when the page opens, each in its own container.
* The section menu, the *Direction* switch and the *Feature* dropdown are **plain HTML
  radio inputs**, not Streamlit widgets. A click only sets `html[data-tbx-…]`
  (`client.js`), and static CSS shows the matching, already-drawn container —
  no server request, no script rerun, no React render.
* Charts are built **once per server process** (`st.cache_resource`) and shared by all
  visitors; figure templates are trimmed and data arrays are float32 (≈40 % less JSON);
  websocket compression is on.
* No interactive data grids (their global pointer listeners cost ~10 ms per click).

Measured locally (headless Chromium, click → new content painted):

| action | before | now |
|---|---|---|
| switch section | 157–188 ms median, up to 516 ms | **≈12 ms** median, ≤ 23 ms |
| switch direction / feature | full rerun | **6–9 ms** |
| server work per click | 0.1–0.5 s script run | **none** |

On the deployed site the old version also waited for the network on every click; the new
one does not. The price is paid once: the first load draws all 27 charts
(a few seconds on a slow machine, in the background after the first section appears).

## Deploy to Streamlit Community Cloud (free, public URL)
1. Create a **public** GitHub repository and upload the contents of `eda_site/`
   (`app.py`, `requirements.txt`, `site_data/`, `.streamlit/`).
   Do **not** upload the raw competition data.
2. Go to https://share.streamlit.io → sign in with GitHub → **Create app**.
3. Choose the repository, branch `main`, main file `app.py` → **Deploy**.
4. Copy the URL (`https://<name>.streamlit.app`) — this is the link for the submission.
5. Open the URL in a private/incognito window to confirm it works without login.

Note: free Streamlit apps go to sleep after a period of inactivity; the first visit
then takes ~30 s to wake up. Open the link yourself shortly before the organizers review it.
