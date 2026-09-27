"""EDA website — AI Financial Alert Risk Scoring (hackathon, fintech track).

Run locally:   streamlit run app.py
All numbers are precomputed from the raw TRAIN data by prepare_data.py and stored
in site_data/ (the raw 135 MB of transactions are not needed to run the site).

The site reads like a case file: seven steps of an investigation, each a question, a short
answer and the evidence.  A vertical "case timeline" on the left (a bottom dock + sheet on
phones) is the navigation.

Performance design (why the site reacts instantly)
--------------------------------------------------
* Every section (and every variant of the switchable charts) is rendered once per session
  into its own keyed container; all charts are drawn up front.
* Navigation and chart tabs are plain HTML radio inputs (st.html), not Streamlit widgets:
  a click never triggers a script rerun, a server round-trip or a React re-render.
  client.js copies the checked value into html[data-tbx-*] inside the click handler and
  static CSS shows the matching container in the same frame.
* Interactive explainers (score journey, schema, observation explorer, ceiling…) are
  self-contained HTML: local radios + CSS :has(), so they also need no server.
* Plotly figures are built once per server process (`@st.cache_resource`) and reused by
  every session; templates are trimmed to what the charts use and data arrays are
  float32, so the page ships ~40 % less chart JSON.
"""
import html
import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

TEAM_NAME = "tobaskAplus"
TEAM_ID = "3B832E89"
HERE = Path(__file__).parent
DATA = HERE / "site_data"

st.set_page_config(page_title="Alert Escalation · EDA", page_icon=str(HERE / "assets" / "favicon.png"),
                   layout="wide", initial_sidebar_state="collapsed")

# ---------- palette: cool dark UI + the team's cyan (see style.css tokens) ----------
PANEL, PANEL_2 = "#11161D", "#161C24"                     # chart card / tooltip surfaces
INK, INK_2, INK_3, LINE, LINE_2 = "#E8EDF2", "#9AA5B1", "#6B7683", "#1F2731", "#2C3643"
FONT = "Inter, system-ui, sans-serif"
# Charts: categorical colours validated on the card surface (lightness band, CVD ΔE ≥ 11, contrast ≥ 3:1)
BLUE, ORANGE, AQUA, VIOLET = "#5B8DEF", "#DB6C4B", "#22A99A", "#9B7BE0"
GRAY = INK_3
CLASS_COLORS = {"Dismissed": BLUE, "Escalated": ORANGE}
TYPE_COLORS = {"Card": BLUE, "Bank transfer": ORANGE, "Cash": AQUA, "International": VIOLET}
DIR_COLORS = {"Incoming (kirim)": BLUE, "Outgoing (chiqim)": ORANGE}
SEQ = ["#161C24", "#123040", "#0F4454", "#0B5A6A", "#087182", "#0A8A9B", "#14A3B2"]   # one hue, dark → light

# ---------- the team page: fill in here, everything else is generated ----------
# photo / photo_back: files in assets/team/ ("" = initials); photo_back flips in on hover
# log: (step, text) — step 0–6 is the case step it belongs to (01 Overview … 07 Conclusion),
#      None for work outside the steps; the coverage map on the page is built from these
TEAM = [
    {"name": "Shoxruh Matrizayev", "initials": "SM", "photo": "shoxruh.jpg", "role": "Team lead · model training",
     "bio": "Team lead. Owns the modelling pipeline end to end — from leak-safe validation to the audited ensemble "
            "behind the 0.6533 score — and made sure the EDA and the model tell the same story.",
     "log": [(None, "Trained the models: leak-safe CV, CatBoost, LightGBM, logistic regression and the rank ensemble"),
             (2, "Target step: class balance, drift over time and the validation strategy"),
             (3, "Activity step: the burst before alerts, uniform timestamps and the common trend"),
             (5, "Model step: EDA-driven decisions, the ensemble, rejected ideas and the audit")],
     "github": "https://github.com/Shoha-ops", "linkedin": "https://www.linkedin.com/in/shoxruh-inha/"},
    {"name": "Safina Babarakhimova", "initials": "SB", "photo": "safina.jpg", "photo_back": "safina_back.jpg",
     "role": "Site structure & data visualisation",
     "bio": "A second-year Software Engineering student at Inha University in Tashkent with a passion for software "
            "architecture, UI/UX and data visualisation. I enjoy solving complex logic problems and building clean, "
            "user-friendly applications. In the team I focused on translating our technical analyses into a clear, "
            "interactive platform that presents our findings.",
     "log": [(None, "Site structure: the case-file flow, navigation and page layout"),
             (1, "Dataset step: the two-table schema, volumes and amount distributions"),
             (4, "Comparison step: feature deciles, the risk simulator and the alert inspector")],
     "github": "https://github.com/safinarefugioz", "linkedin": "https://www.linkedin.com/in/safina-babarakhimova-30b8683b9/"},
    {"name": "Samiya Abdukarimova", "initials": "SA", "photo": "samiya.jpg", "role": "Site design",
     "bio": "A Computer Engineering student at Inha University in Tashkent, passionate about software development and "
            "continuously improving my technical skills. I enjoy solving problems, building projects and learning new "
            "technologies; academic and personal projects taught me how to design and develop applications.",
     "log": [(None, "Visual design of the site: colour system, typography and components"),
             (0, "Overview step: the main finding and the score journey"),
             (6, "Conclusion step: the four findings, the analyst budget and the ceiling")],
     "github": "https://github.com/aswasa", "linkedin": "https://www.linkedin.com/in/samiya-abdukarimova-30686738b/"},
]
# project links on the team page — (label, url); "" shows "soon"
TEAM_LINKS = [
    ("Repository", "https://github.com/Shoha-ops/tobaskAplus-hack"),
    ("Notebook", "https://github.com/Shoha-ops/tobaskAplus-hack/blob/main/notebooks/solution.ipynb"),
    ("Submission", "https://github.com/Shoha-ops/tobaskAplus-hack/blob/main/submissions/team_3B832E89.csv"),
]

# (short name, the question the step answers, the answer in one line)
SECTIONS = [
    ("Overview", "Which alerts get escalated?",
     "Those where **cash deposits and card spending are large relative to bank transfers**."),
    ("Dataset", "What are we working with?",
     "14,000 alerts — each a separate customer with **~460 transactions over 180 days**."),
    ("Target", "Is the target balanced — and stable?",
     "**17 % escalated**, and the rate does not drift over time or over IDs."),
    ("Activity", "Does timing tell us anything?",
     "A burst precedes every alert, but **timing carries no signal**."),
    ("Comparison", "What separates escalated alerts?",
     "Not volume, not timing — **the balance between payment instruments**."),
    ("Model", "How did the EDA shape the model?",
     "Instrument-level features gave **+0.039 AUC**; the audited ensemble scores **0.6533**."),
    ("Conclusion", "How close are we to the limit?",
     "Very close: **0.6533** against an estimated ceiling of **0.652–0.657**."),
]
DIRECTIONS = ["Incoming", "Outgoing"]
# team mark (assets/logo.svg) as a data URI: st.html drops inline <svg> but keeps <img src="data:…">
import base64
LOGO = "data:image/svg+xml;base64," + base64.b64encode((HERE / "assets" / "logo.svg").read_bytes()).decode()


def _team_img(name):
    """A team photo from assets/team/ as a data URI (st.html keeps <img src="data:…">), or None."""
    f = HERE / "assets" / "team" / name if name else None
    if not f or not f.exists():
        return None
    mime = {".png": "png", ".webp": "webp"}.get(f.suffix.lower(), "jpeg")
    return f"data:image/{mime};base64," + base64.b64encode(f.read_bytes()).decode()


def _face(m, k):
    """Inside of a small round avatar: the photo, or the initials."""
    ph = _team_img(m.get("photo", ""))
    return f'<img src="{ph}" alt="">' if ph else html.escape(m["initials"])
PLOT_CONFIG = {"displaylogo": False, "responsive": True, "modeBarButtons": [["toImage"]],
               "toImageButtonOptions": {"format": "png", "scale": 2}}


# =====================================================================================
# Process-wide resources (computed once per server process, shared by all sessions)
# =====================================================================================
@st.cache_resource
def _template():
    """plotly_dark + our layout, with trace templates only for the trace types we draw."""
    ours = go.layout.Template(layout=dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK_2, size=12.5),
        title=dict(font=dict(color=INK, size=15)),
        colorway=[BLUE, ORANGE, AQUA, VIOLET],
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE_2, linecolor=LINE_2, tickcolor=LINE_2,
                   title=dict(font=dict(size=12, color=INK_3))),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE_2, linecolor=LINE_2, tickcolor=LINE_2,
                   title=dict(font=dict(size=12, color=INK_3))),
        legend=dict(font=dict(color=INK_2, size=12.5)),
        hoverlabel=dict(bgcolor=PANEL_2, bordercolor=LINE_2, font=dict(color=INK, family=FONT)),
        bargap=0.25, bargroupgap=0.08,
    ))
    full = pio.templates.merge_templates(pio.templates["plotly_dark"], ours).to_plotly_json()
    # Every figure carries its own copy of the template, so drop what these charts never use
    # (geo/polar/3-D/map defaults, trace types other than bar/scatter/heatmap): -60 % JSON.
    keep = {"annotationdefaults", "autotypenumbers", "colorway", "font", "hoverlabel", "hovermode",
            "paper_bgcolor", "plot_bgcolor", "shapedefaults", "title", "xaxis", "yaxis", "legend", "bargap", "bargroupgap"}
    layout = {k: v for k, v in full["layout"].items() if k in keep}
    data = {k: v for k, v in full["data"].items() if k in ("bar", "scatter", "heatmap")}
    return go.layout.Template(layout=layout, data=data)


pio.templates["tobask"] = _template()
pio.templates.default = "tobask"


@st.cache_resource
def _csv(name):
    return pd.read_csv(DATA / name)


def csv(name):
    return _csv(name).copy()          # callers may mutate; the cached frame stays pristine


@st.cache_resource
def js(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@st.cache_resource
def _figures(version):
    return {}                         # key -> (finished plotly Figure, title) (shared, read-only)


# Bump when the stored value's shape or the figure styling changes: a running server keeps
# st.cache_resource across reruns, so old entries would otherwise be served (or break unpacking).
FIGURES_VERSION = 8


def _compact(fig):
    """float64 -> float32 for data arrays: plotly ships numpy arrays as base64 binary, so this
    halves their size; 7 significant digits is far beyond any displayed precision."""
    import numpy as np
    for tr in fig.data:
        for attr in ("x", "y", "z", "customdata"):
            v = getattr(tr, attr, None)
            if v is None:
                continue
            a = np.asarray(v)
            if a.dtype == np.float64:
                setattr(tr, attr, a.astype(np.float32))


def chart(key, height=360, bare=False):
    """Decorator: build the figure once per process, then only send it.

    @chart("s2_tx")
    def _():
        return px.bar(...)

    The figure title moves out of plotly into HTML, so it wraps on narrow screens instead of
    being clipped.  bare=True: no own card (the chart sits inside a tabbed card).
    """
    def deco(build):
        store = _figures(FIGURES_VERSION)
        hit = store.get(key)
        if hit is None:
            fig = build()
            title = fig.layout.title.text or ""
            fig.update_layout(title_text=None, height=height, margin=dict(l=4, r=12, t=8, b=4),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left", title=None),
                              hoverlabel=dict(font_size=13), barcornerradius=4,
                              dragmode=False)      # no drag-zoom: touch scrolling over a chart keeps working
            _compact(fig)
            hit = store[key] = (fig, title)
        fig, title = hit
        with st.container(key=f"{'bare' if bare else 'card'}_{key}", gap=None):
            if title:
                st.html(f'<div class="{"ui-sub" if bare else "ui-card-title"}">{html.escape(title)}</div>')
            st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG)
        return build
    return deco


def callout(fig, x, y, text, ax=0, ay=-44, **kw):
    """Put the take-away on the chart itself: a small label with a leader line."""
    fig.add_annotation(x=x, y=y, text=text, showarrow=True, arrowhead=0, arrowwidth=1.2, arrowcolor=INK_3,
                       ax=ax, ay=ay, font=dict(size=12, color=INK), bgcolor=PANEL_2, bordercolor=LINE_2,
                       borderwidth=1, borderpad=5, align="left", **kw)


def tag(fig, text, x=0.01, y=0.98, xanchor="left"):
    """A take-away without a specific point: a label pinned to the top of the plot area."""
    fig.add_annotation(xref="paper", yref="paper", x=x, y=y, xanchor=xanchor, yanchor="top", text=text,
                       showarrow=False, font=dict(size=12, color=INK), bgcolor=PANEL_2, bordercolor=LINE_2,
                       borderwidth=1, borderpad=5, align="left")


MUTED = "#344152"                        # bars that are context, not the point


# ---------- CSS state machine: which section / variant is visible ----------
# Navigation and chart tabs are plain HTML radio inputs, not Streamlit widgets: choosing an
# option never contacts the server and never re-renders React.  client.js mirrors the checked
# value into html[data-tbx-*]; CSS does the rest.
HIDE = ("position:absolute!important;top:0;left:0;right:0;height:0!important;min-height:0!important;"
        "margin:0!important;overflow:hidden!important;visibility:hidden!important;pointer-events:none!important;")
# SHOW inherits visibility, so a chosen variant inside a hidden section stays hidden
SHOW = ("position:relative!important;height:auto!important;overflow:visible!important;"
        "visibility:inherit!important;pointer-events:inherit!important;")
# Motion played by whatever becomes visible (fill-mode "backwards": once finished,
# the animation leaves no transform/stacking context behind, so menus still overlay later cards)
ENTER_SEC = "enter 420ms cubic-bezier(.2,.7,.2,1) backwards"
ENTER_TAB = "fade-in 220ms ease-out backwards"
# every switch group: [number of options, default option]
TEAM_SEC = len(SECTIONS)                  # index of the team page (not a step of the case)
GROUPS = {"sec": [len(SECTIONS) + 1, 0], "ds": [2, 0], "dir": [2, 1], "tg": [2, 0], "tm": [2, 0],
          "pc": [3, 0], "cx": [3, 0]}          # "feat" added once FEATS is known


@st.cache_resource
def _switch_css(groups_json):
    """Static CSS for every switch group {prefix: [n_options, default]}.

    Each group's variants sit in one gap-less parent container (st.container(key=prefix, gap=None)),
    so a hidden variant takes no space.  Everything except the default option is hidden; client.js
    copies the checked radio into html[data-tbx-<prefix>] inside the click handler, and one
    attribute rule per option shows it.  Deliberately no `.stApp:has(...)`: a :has() on the page
    root is re-checked on every DOM change and made style recalculation ~15 % slower.
    The children of the shown option get an enter animation that restarts on every switch.
    """
    out = []
    for p, (n, default) in json.loads(groups_json).items():
        anim = ENTER_SEC if p == "sec" else ENTER_TAB
        out.append(f".st-key-{p}{{position:relative}}")
        out.append(",".join(f".st-key-{p}_{j}" for j in range(n) if j != default) + f"{{{HIDE}}}")
        for i in range(n):
            others = ",".join(f"html[data-tbx-{p}='{i}'] .st-key-{p}_{j}" for j in range(n) if j != i)
            out.append(f"{others}{{{HIDE}}}")
            out.append(f"html[data-tbx-{p}='{i}'] .st-key-{p}_{i}{{{SHOW}}}")
            sel = f"html[data-tbx-{p}='{i}'] .st-key-{p}_{i}>*"
            if i == default:
                sel += f",html:not([data-tbx-{p}]) .st-key-{p}_{i}>*"
            out.append(f"{sel}{{animation:{anim}}}")
    # case timeline: the progress line fills up to the current step
    n = len(SECTIONS)
    for i in range(n):
        sel = f"html[data-tbx-sec='{i}'] .ui-track-fill" + (",html:not([data-tbx-sec]) .ui-track-fill" if i == 0 else "")
        out.append(f"{sel}{{transform:scaleY({i / (n - 1):.4f})}}")
        out.append(f"html[data-tbx-sec='{i}'] .ui-dock-bar b{{transform:scaleX({(i + 1) / n:.4f})}}")
    out.append(f"html:not([data-tbx-sec]) .ui-dock-bar b{{transform:scaleX({1 / n:.4f})}}")
    out.append(f"html[data-tbx-sec='{n}'] .ui-dock-bar{{opacity:0}}"
               f"html[data-tbx-sec='{n}'] .ui-avatars{{background:var(--panel-2);box-shadow:inset 0 0 0 1px var(--sign)}}"
               f"html[data-tbx-sec='{n}'] .ui-dock-team{{color:var(--sign);background:var(--panel-3)}}"
               f"html[data-tbx-sec='{n}'] .ui-node.seen .dot{{background:var(--line-3)}}")
    # moving forward slides the new step in from the right, moving back from the left
    for i in range(n):
        for d in ("next", "prev"):
            out.append(f"html[data-dir='{d}'][data-tbx-sec='{i}'] .st-key-sec_{i}>*{{animation-name:enter-{d}}}")
    return "\n".join(out)


# =====================================================================================
# Small HTML helpers
# =====================================================================================
def _icon(name, cls=""):
    return f'<span class="msr {cls}" aria-hidden="true">{name}</span>'


def _md(text):
    """**bold** / *italic* / `code` → HTML (input is escaped first)."""
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    # [[text|step|css selector|group:tab]] → a link that opens the step and scrolls to the evidence
    t = re.sub(r"\[\[(.+?)\|(\d)\|([^|\]]*)(?:\|([^\]]*))?\]\]",
               lambda m: (f'<a class="ui-ref" href="#s{int(m[2]) + 1}" data-target="{m[3]}"'
                          f'{f" data-tab={chr(34)}{m[4]}{chr(34)}" if m[4] else ""}>{m[1]}</a>'), t)
    return t


def _opt(group, i, text, checked, cls="", extra=""):
    return (f'<label class="{cls}"><input type="radio" name="tbx-{group}" value="{i}"'
            f'{" checked" if checked else ""}>{extra}<span>{html.escape(text)}</span></label>')


def _style(rules):
    return f"<style>{''.join(rules)}</style>"


def _svg(markup, cls, ratio):
    """st.html's sanitiser drops <svg>, so the markup travels in a data attribute and client.js
    inserts it; the host keeps the final aspect ratio, so nothing jumps when it appears."""
    return f'<div class="svg-host {cls}-host" style="aspect-ratio:{ratio}" data-svg="{html.escape(markup, quote=True)}"></div>'


def shell_html():
    """Case timeline (left rail on wide screens, bottom sheet on phones) + the phone dock."""
    nodes = "".join(
        f'<label class="ui-node"><input type="radio" name="tbx-sec" id="tbx-sec-{i}" value="{i}"'
        f'{" checked" if i == 0 else ""}><span class="dot" aria-hidden="true"></span>'
        f'<span class="n">{i + 1:02d}</span><span class="t">{html.escape(name)}</span>'
        f'<span class="q">{html.escape(q)}</span></label>'
        for i, (name, q, _) in enumerate(SECTIONS))
    avatars = "".join(f'<i style="--h:{k}">{_face(m, k)}</i>' for k, m in enumerate(TEAM))
    return f'''
<div class="ui-splash" aria-hidden="true"><div class="sp-box"><img class="sp-logo" src="{LOGO}" alt="">
  <b class="ui-word">tobask<em>A+</em></b><span class="sp-t">Opening the case file</span>
  <span class="sp-bar"><i></i></span></div></div>
<input type="checkbox" id="ui-sheet" class="ui-sheet-state" aria-label="Open the case timeline">
<label for="ui-sheet" class="ui-scrim" aria-hidden="true"></label>
<aside class="ui-rail" aria-label="Case timeline">
  <div class="ui-rail-head"><img class="ui-logo" src="{LOGO}" alt="">
    <div><b class="ui-word">tobask<em>A+</em></b><span>Case file · alert escalation</span></div>
    <input type="radio" name="tbx-sec" id="tbx-sec-{TEAM_SEC}" value="{TEAM_SEC}" class="ui-team-radio" aria-label="Team">
    <label for="tbx-sec-{TEAM_SEC}" class="ui-avatars" tabindex="0" role="button" aria-label="Open the team page" title="Team">{avatars}</label></div>
  <div class="ui-rail-label">Investigation</div>
  <nav class="ui-track tbx-nav" role="radiogroup" aria-label="Sections">
    <span class="ui-track-line" aria-hidden="true"><i class="ui-track-fill"></i></span>{nodes}</nav>
  <div class="ui-rail-foot"><span class="ui-kbd"><kbd>1</kbd>–<kbd>7</kbd> jump · <kbd>←</kbd><kbd>→</kbd> step</span></div>
</aside>
<div class="ui-dock" role="navigation" aria-label="Section navigation">
  <button type="button" class="ui-dock-btn" data-step="-1" aria-label="Previous section">{_icon("arrow_back")}</button>
  <label for="ui-sheet" class="ui-dock-mid"><span class="ui-dock-n">01 / {len(SECTIONS):02d}</span>
    <span class="ui-dock-t">{html.escape(SECTIONS[0][0])}</span><span class="ui-dock-bar"><b></b></span></label>
  <label for="tbx-sec-{TEAM_SEC}" class="ui-dock-team" tabindex="0" role="button" aria-label="Open the team page">{_icon("group")}</label>
  <button type="button" class="ui-dock-btn" data-step="1" aria-label="Next section">{_icon("arrow_forward")}</button>
</div>'''


def head(i, lead=None):
    """Section header: outlined step number, the question, the one-line answer."""
    name, q, a = SECTIONS[i]
    lead_html = f'<p class="ui-lead">{_md(lead)}</p>' if lead else ""
    st.html(f'<header class="ui-head"><span class="ui-num" aria-hidden="true">{i + 1:02d}</span>'
            f'<div class="ui-head-body"><div class="ui-kicker">{html.escape(name)}</div>'
            f'<h1>{html.escape(q)}</h1><p class="ui-answer"><span class="lab">Answer</span><span class="txt">{_md(a)}</span></p>{lead_html}</div></header>')


def title(text, sub=None):
    """Sub-heading inside a section (replaces st.subheader: no anchor link, own style)."""
    st.html(f'<h2 class="ui-h2">{_md(text)}</h2>' + (f'<p class="ui-h2-sub">{_md(sub)}</p>' if sub else ""))


def note(text, kind="Field note", icon="lightbulb"):
    st.html(f'<aside class="ui-note ui-note-{kind.split()[0].lower()}"><div class="ui-note-h">{_icon(icon)}{html.escape(kind)}</div>'
            f'<div>{_md(text)}</div></aside>')


def tabbed(group, labels, heading=None):
    """A card whose body switches between variants without a server round-trip.
    Returns one container per variant."""
    default = GROUPS[group][1]
    opts = "".join(_opt(group, i, lab, i == default) for i, lab in enumerate(labels))
    with st.container(key=f"tcard_{group}", gap=None):
        st.html(f'<div class="ui-tc-head">{f"<b>{_md(heading)}</b>" if heading else ""}'
                f'<div class="tbx-seg" role="radiogroup" aria-label="{html.escape(heading or group)}">{opts}</div></div>')
        box = st.container(key=group, gap=None)
    return [box.container(key=f"{group}_{i}") for i in range(len(labels))]


def select_html(group, label, options, default):
    vals = "".join(f'<span class="tbx-v tbx-v{i}">{html.escape(o)}</span>' for i, o in enumerate(options))
    opts = "".join(_opt(group, i, o, i == default, "tbx-item", _icon("check", "chk")) for i, o in enumerate(options))
    show = "".join(f".tbx-select:has(input[value='{i}']:checked) .tbx-v{i},.tbx-select[data-v='{i}'] .tbx-v{i}"
                   "{display:block}" for i in range(len(options)))
    return (f'<style>{show}</style><div class="ui-toolbar tbx-field"><div class="tbx-label">{label}</div>'
            f'<div class="tbx-select-field"><details class="tbx-select" data-v="{default}"><summary aria-label="{label}">'
            f'<span class="tbx-vals">{vals}</span><span class="tbx-chev">{_icon("expand_more")}</span></summary>'
            f'<div class="tbx-list" role="radiogroup" aria-label="{label}">{opts}</div></details></div></div>')


def table_html(headers, rows, num=()):
    """Responsive table: a normal table on wide screens, one card per row on phones."""
    th = "".join(f'<th{" class=num" if j in num else ""}>{html.escape(h)}</th>' for j, h in enumerate(headers))
    body = "".join("<tr>" + "".join(
        f'<td data-label="{html.escape(headers[j])}"{" class=num" if j in num else ""}>{_md(str(c))}</td>'
        for j, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="ui-table"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


@st.cache_resource
def _static_assets(mtimes):
    # mtimes is part of the cache key: editing style.css / client.js takes effect on the next
    # page load without restarting the server
    return (HERE / "style.css").read_text(encoding="utf-8"), (HERE / "client.js").read_text(encoding="utf-8")


# =====================================================================================
# Data used by several sections
# =====================================================================================
ov = js("overview.json")
R = js("results.json")
final_auc = R["models"][-1]["auc_mean"]
_dec = _csv("deciles.csv")
FEATS = list(dict.fromkeys(_dec["feature"]))
FEAT_DEFAULT = next(i for i, x in enumerate(FEATS) if x.startswith("Main contrast"))

STYLE, CLIENT_JS = _static_assets(tuple((HERE / f).stat().st_mtime_ns for f in ("style.css", "client.js")))
assert not re.search(r"<[/\w]", CLIENT_JS), "client.js must not contain '<' + letter or '/' (DOMPurify drops it)"
GROUPS["feat"] = [len(FEATS), FEAT_DEFAULT]
SWITCH_CSS = _switch_css(json.dumps(GROUPS))


# =====================================================================================
# Interactive explainers (self-contained HTML; local radios + CSS, no server)
# =====================================================================================
def journey_html():
    """Overview: the score growing step by step — click a step (or replay) to walk through our work."""
    path = R["metric_path"]
    short = ["Baseline", "+ Shape", "+ Ensemble", "+ Instruments", "Final"]
    about = [
        "59 aggregates per alert: amount statistics for all / incoming / outgoing transactions, instrument "
        "shares and counts, balance, activity, alert date. LightGBM, repeated CV.",
        "+49 distribution-shape features: kurtosis, Bowley skew, inter-quantile ranges, tail ratios. "
        "They barely move the score — the signal is not in the shape.",
        "A 4-model rank average. Ensembling alone adds about half a point.",
        "+130 statistics for each **instrument × direction** (8 groups) — the EDA finding that "
        "[[every instrument has its own normal amount|1|.st-key-tcard_dir]]. The largest jump of the project.",
        "Honest early stopping on a separate slice, tuning, the explicit contrast feature, logistic-regression "
        "and hybrid models. The final ensemble, after the audit.",
    ]
    W, H, pl, pr, pt, pb = 640, 250, 46, 26, 30, 30
    lo, hi = 0.600, 0.660
    xs = [pl + i * (W - pl - pr) / (len(path) - 1) for i in range(len(path))]
    ys = [pt + (hi - p["auc"]) / (hi - lo) * (H - pt - pb) for p in path]
    seg = [((xs[i] - xs[i - 1]) ** 2 + (ys[i] - ys[i - 1]) ** 2) ** 0.5 for i in range(1, len(xs))]
    total = sum(seg)
    cum = [0.0] + [sum(seg[:i + 1]) for i in range(len(seg))]
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in zip(xs, ys))
    area = d + f" L{xs[-1]:.1f} {H - pb} L{xs[0]:.1f} {H - pb} Z"
    grid = "".join(f'<line x1="{pl}" x2="{W - pr}" y1="{pt + (hi - v) / (hi - lo) * (H - pt - pb):.1f}" '
                   f'y2="{pt + (hi - v) / (hi - lo) * (H - pt - pb):.1f}" class="g"/>'
                   f'<text x="{pl - 8}" y="{pt + (hi - v) / (hi - lo) * (H - pt - pb) + 4:.1f}" class="yl">{v:.2f}</text>'
                   for v in (0.60, 0.62, 0.64, 0.66))
    pts = "".join(f'<g class="jr-p jr-p{i}"><circle cx="{x:.1f}" cy="{y:.1f}" r="6"/>'
                  f'<text x="{x:.1f}" y="{y - 14:.1f}" class="vl">{p["auc"]:.4f}</text>'
                  f'<text x="{x:.1f}" y="{H - 8}" class="xl">{i + 1}</text></g>'
                  for i, (x, y, p) in enumerate(zip(xs, ys, path)))
    chips = "".join(f'<label class="jr-chip"><input type="radio" name="jr" id="jr-{i}" value="{i}"'
                    f'{" checked" if i == len(path) - 1 else ""}><span><em>{i + 1}</em>{html.escape(s)}</span></label>'
                    for i, s in enumerate(short))
    panels = ""
    for i, p in enumerate(path):
        delta = p["auc"] - path[i - 1]["auc"] if i else None
        chip = (f'<span class="jr-delta{" big" if delta and delta > 0.02 else ""}">{delta:+.4f}</span>' if delta is not None
                else '<span class="jr-delta base">start</span>')
        panels += (f'<div class="jr-panel jr-panel{i}"><div class="jr-top"><span class="jr-auc">{p["auc"]:.4f}</span>{chip}</div>'
                   f'<b>{html.escape(p["step"])}</b><p>{_md(about[i])}</p></div>')
    rules = []
    for i in range(len(path)):
        on = f".ui-jr:has(#jr-{i}:checked)"
        rules.append(f"{on} .jr-prog{{stroke-dashoffset:{total - cum[i]:.1f}}}")
        rules.append(f"{on} .jr-panel{i}{{display:block}}")
        rules.append(",".join(f"{on} .jr-p{j} circle" for j in range(i + 1)) + "{stroke:var(--sign);fill:#0a2f35}")
        rules.append(f"{on} .jr-p{i} circle{{fill:var(--sign);transform:scale(1.45)}}{on} .jr-p{i} .vl{{fill:var(--ink);font-weight:700;opacity:1}}")
    svg = (f'<svg class="jr-svg" viewBox="0 0 {W} {H}" role="img" aria-label="CV ROC-AUC after each stage">'
           f'<defs><linearGradient id="jrg" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#00f2fe" stop-opacity=".18"/>'
           f'<stop offset="1" stop-color="#00f2fe" stop-opacity="0"/></linearGradient></defs>{grid}'
           f'<path d="{area}" class="jr-area"/><path d="{d}" class="jr-base"/>'
           f'<path d="{d}" class="jr-prog" style="stroke-dasharray:{total:.1f}"/>{pts}</svg>')
    return (_style(rules) +
            f'<section class="ui-jr ui-card-x"><div class="jr-head"><div><b>How the score grew</b>'
            f'<span>Five stages of our work — pick one, or replay the whole run.</span></div>'
            f'<button type="button" class="ui-play" data-play="jr">{_icon("play_arrow")}Replay</button></div>'
            f'<div class="jr-body">{_svg(svg, "jr", f"{W}/{H}")}'
            f'<div class="jr-side"><div class="jr-chips" role="radiogroup" aria-label="Stage">{chips}</div>{panels}</div></div></section>')


def facts_html():
    items = [(f"{ov['train_alerts']:,}", "Training alerts", ""),
             (f"{ov['test_alerts']:,}", "Test alerts", ""),
             (f"{ov['escalation_rate']:.1%}", "Escalation rate", ""),
             (f"{final_auc:.3f}", "Final CV ROC-AUC", " key")]
    return '<div class="ui-facts hl">' + "".join(
        f'<div class="f{cls}"><b data-count>{v}</b><span>{lab}</span></div>' for v, lab, cls in items) + "</div>"


def schema_html():
    """Dataset: the two tables and how they link — click a column to see what it means."""
    cols = {
        "signals": [("signal_id", "key", "Alert identifier. One alert = one customer: no transaction is shared between alerts."),
                    ("signal_sanasi", "date", f"Alert date, **without time**: {ov['date_min']} … {ov['date_max']}, the same range in train and test."),
                    ("eskalatsiya", "target", "1 = the specialist escalated the alert, 0 = dismissed. Train only.")],
        "transactions": [("signal_id", "key", "Links every transaction to its alert."),
                         ("tranzaksiya_vaqti", "time", f"Timestamp. A {ov['window_days']}-day window that ends with a ~3-minute burst right before the alert."),
                         ("kirim_chiqim", "direction", "Incoming (kirim) or outgoing (chiqim)."),
                         ("tranzaksiya_turi", "instrument", "Card, bank transfer, cash or international."),
                         ("miqdor_indeksi", "amount", "Standardised size indicator — behaves like a **log-amount on a common scale**.")],
    }
    sizes = {"signals": f"{ov['train_alerts']:,} train · {ov['test_alerts']:,} test",
             "transactions": f"{ov['train_tx'] / 1e6:.2f} M train · {ov['test_tx'] / 1e6:.2f} M test"}
    uid, rules, panels, tables = 0, [], "", []
    for tname, cl in cols.items():
        items = ""
        for c, kind, desc in cl:
            checked = " checked" if c == "miqdor_indeksi" else ""
            items += (f'<label class="sc-col"><input type="radio" name="sc" id="sc-{uid}"{checked}>'
                      f'<code>{c}</code><em>{kind}</em></label>')
            panels += f'<div class="sc-desc sc-d{uid}"><code>{tname}.{c}</code><p>{_md(desc)}</p></div>'
            rules.append(f".ui-schema:has(#sc-{uid}:checked) .sc-d{uid}{{display:block}}")
            uid += 1
        tables.append(f'<div class="sc-table"><div class="sc-th">{_icon("table")}<b>{tname}</b><span>{sizes[tname]}</span></div>'
                      f'<div class="sc-cols">{items}</div></div>')
    link = '<div class="sc-link" aria-hidden="true"><span>1</span><i></i><span>N</span></div>'
    return _style(rules) + (f'<section class="ui-schema">{tables[0]}{link}{tables[1]}'
                            f'<div class="sc-panel">{panels}</div></section>')


def waffle_html():
    esc, n = ov["escalated"], ov["train_alerts"]
    k = round(esc / n * 100)
    cells = "".join(f'<i class="{"e" if j < k else "d"}" style="--j:{j}"></i>' for j in range(100))
    return (f'<section class="ui-waffle"><div class="wf-grid" role="img" aria-label="{k} of 100 alerts are escalated">{cells}</div>'
            f'<div class="wf-legend"><p class="wf-lead">Out of every <b>100</b> alerts, <b class="e">{k}</b> are escalated.</p>'
            f'<div class="wf-row"><i class="e"></i><span>Escalated</span><b>{esc:,}</b></div>'
            f'<div class="wf-row"><i class="d"></i><span>Dismissed</span><b>{n - esc:,}</b></div>'
            f'<p class="wf-foot">One square ≈ {n // 100} alerts.</p></div></section>')


def meter_html(rows, lo, hi, fmt="{:.3f}"):
    """Horizontal bars on a shared scale [lo, hi] (for small AUC comparisons)."""
    out = ""
    for lab, v, hot in rows:
        w = max(0, min(1, (v - lo) / (hi - lo)))
        out += (f'<div class="mt-row{" hot" if hot else ""}"><span class="mt-l">{_md(lab)}</span>'
                f'<span class="mt-bar"><i style="--w:{w:.3f}"></i></span><b>{fmt.format(v)}</b></div>')
    return f'<div class="ui-meter">{out}</div>'


def explorer_html():
    """Model: EDA observation → decision → effect, one at a time."""
    rows = [("Instruments have very different 'normal' amounts",
             "Statistics per **instrument × direction** (8 groups) instead of one pooled set.", "+0.039", 0.039),
            ("The signal ≈ a linear contrast hidden behind the customer scale",
             "An explicit contrast feature; logistic regression and a hybrid model in the ensemble.", "+0.004", 0.004),
            ("A single train / test split varies by ±0.015 AUC",
             "Repeated CV over 3 seeds and fixed acceptance thresholds.", "reliable comparisons", None),
            ("Timestamps are uniform; the trend is identical in both classes",
             "No time features in the final model.", "avoided noise", None),
            ("The burst before an alert carries no extra signal",
             "The burst stays inside the history, no special features.", "—", None),
            ("Instrument order is random",
             "No sequence or chain features.", "—", None)]
    proof = [(1, ".st-key-tcard_dir", ""), (4, ".st-key-tcard_pc", "pc:1"), (2, ".st-key-tcard_tg", ""),
             (3, ".st-key-card_s4_trend", ""), (3, ".ui-burst", ""), (3, ".st-key-tcard_tm", "")]
    items, rules = "", []
    for i, (obs, dec, eff, val) in enumerate(rows):
        sec, sel, tab = proof[i]
        link = _md(f"[[See it in the data|{sec}|{sel}" + (f"|{tab}" if tab else "") + "]]")
        bar = (f'<div class="ex-bar"><i style="--w:{val / 0.039:.3f}"></i></div>' if val else "")
        items += (f'<label class="ex-item"><input type="radio" name="ex" id="ex-{i}"{" checked" if i == 0 else ""}>'
                  f'<span class="ex-n">{i + 1:02d}</span><span class="ex-o">{html.escape(obs)}</span></label>'
                  f'<div class="ex-panel ex-p{i}"><div class="ex-k">Observation</div><p class="ex-obs">{html.escape(obs)}</p>'
                  f'<div class="ex-k">Decision</div><p>{_md(dec)}</p><div class="ex-k">Effect on CV ROC-AUC</div>'
                  f'<div class="ex-eff"><b>{html.escape(eff)}</b>{bar}</div><div class="ex-link">{link}</div></div>')
        rules.append(f".ui-ex:has(#ex-{i}:checked) .ex-p{i}{{display:block}}")
    return _style(rules) + f'<section class="ui-ex">{items}</section>'


def features_html():
    fg = R["feature_groups"]
    total = sum(g["n"] for g in fg)
    colors = ["#2e3947", "#3a4a5c", "#00c2cc", "#5b8def", "#9b7be0"]
    segs = "".join(f'<i class="fg-s fg-s{i}" style="flex:{g["n"]};background:{colors[i]}" title="{html.escape(g["group"])}: {g["n"]}"></i>'
                   for i, g in enumerate(fg))
    legend = "".join(f'<li class="fg-l fg-l{i}"><i style="background:{colors[i]}"></i><b>{g["n"]}</b>'
                     f'<span><em>{html.escape(g["group"])}</em>{html.escape(g["what"])}</span></li>' for i, g in enumerate(fg))
    rules = [f".ui-fg:has(.fg-l{i}:hover) .fg-s:not(.fg-s{i}),.ui-fg:has(.fg-s{i}:hover) .fg-s:not(.fg-s{i}){{opacity:.25}}"
             f".ui-fg:has(.fg-s{i}:hover) .fg-l{i},.ui-fg:has(.fg-l{i}:hover) .fg-l{i}{{background:var(--panel-2)}}" for i in range(len(fg))]
    return _style(rules) + (f'<section class="ui-fg"><div class="fg-top"><b data-count>{total}</b><span>features per alert</span></div>'
                            f'<div class="fg-bar">{segs}</div><ul class="fg-legend">{legend}</ul></section>')


def leaderboard_html():
    ms = sorted(R["models"], key=lambda m: -m["auc_mean"])
    lo, hi = 0.636, 0.655
    rows = ""
    for m in ms:
        final = m["auc_std"] is None or pd.isna(m["auc_std"])
        w = (m["auc_mean"] - lo) / (hi - lo)
        std = "" if final else f"± {m['auc_std']:.4f}"
        rows += (f'<div class="lb-row{" final" if final else ""}"><span class="lb-name">{html.escape(m["model"])}</span>'
                 f'<span class="lb-bar"><i style="--w:{w:.3f}"></i></span><b>{m["auc_mean"]:.4f}</b><small>{std}</small></div>')
    return f'<section class="ui-lb">{rows}</section>'


def rejected_html():
    fl = sorted(R["failed"], key=lambda r: (r["delta"] is None, -(r["delta"] or 0)))
    lim = 0.004
    rows = ""
    for r in fl:
        dlt = r["delta"]
        if dlt is None:
            bar, val = '<span class="rj-bar na"></span>', "n/a"
        else:
            w = min(1, abs(dlt) / lim) / 2
            side = "pos" if dlt > 0 else "neg"
            bar, val = f'<span class="rj-bar"><i class="{side}" style="--w:{w:.3f}"></i></span>', f"{dlt:+.4f}"
        rows += (f'<details class="rj-row"><summary><span class="rj-idea">{html.escape(r["idea"])}</span>{bar}'
                 f'<b>{val}</b>{_icon("expand_more", "rj-chev")}</summary><p>{html.escape(r["why"])}</p></details>')
    return (f'<section class="ui-rj"><div class="rj-axis"><span>−0.004</span><span>0</span><span>+0.004</span></div>{rows}</section>')


def audit_html():
    a = R["audit"]
    items = [
        ("Shuffled labels", f"AUC {a['permutation_auc_range'][0]:.3f}–{a['permutation_auc_range'][1]:.3f}",
         "The full pipeline trained on permuted targets scores at chance — no leakage in features or validation."),
        ("Train / test consistency", f"AUC {a['adversarial_auc']:.3f}",
         "A classifier trying to tell train rows from test rows on the final features cannot do it."),
        ("Stable contrast", "2 of 2 halves",
         "Two disjoint halves of the data independently pick the same four instruments with the same signs."),
        ("Residuals", f"{a['residual_features_above_noise']} of {a['residual_features_tested']}",
         f"candidate features correlate with the ensemble residual; ≈{a['residual_expected_by_chance']} would by chance."),
        ("Selection bias", f"≈ {a['selection_bias']:.4f}",
         f"estimated optimism from choosing among ~35 experiments (± {a['selection_bias_se']:.4f})."),
        ("Date and target", f"AUC ≤ {a['date_auc_max']:.3f}", "Alert-date features alone carry no signal."),
        ("Seed stability", f"± {a['stability_single_seed_std']:.4f}",
         f"single-seed ensemble {a['stability_single_seed_mean']:.4f}; test ranking Spearman {a['stability_rank_spearman']:.3f}; "
         f"removing any one model costs ≤ {a['leave_one_out_max_drop']:.4f}."),
        ("Junk-feature control", f"ranks {', '.join(map(str, R['junk_feature_ranks']))}",
         f"3 random columns ranked {', '.join(map(str, R['junk_feature_ranks']))} of {R['junk_total']} — only the top of the importance list is trustworthy."),
        ("Transactions after 00:00", "not leakage", a["post_date_note"] + "."),
    ]
    rows = "".join(f'<details class="au-row"{" open" if i == 0 else ""}><summary>{_icon("check_circle", "au-ok")}'
                   f'<span class="au-t">{html.escape(t)}</span><span class="au-v">{html.escape(v)}</span>'
                   f'{_icon("expand_more", "au-chev")}</summary><p>{html.escape(txt)}</p></details>'
                   for i, (t, v, txt) in enumerate(items))
    return f'<section class="ui-au"><div class="au-sum"><b>{len(items)} / {len(items)}</b><span>checks passed</span></div>{rows}</section>'


def ceiling_html():
    """Conclusion: how much of each customer's history is needed — step through the measured points."""
    ce = R["ceiling"]
    fr, au = ce["fraction"], ce["auc"]
    W, H, pl, pr, pt, pb = 640, 250, 50, 24, 20, 14
    lo, hi = 0.575, 0.665
    X = lambda f: pl + f * (W - pl - pr)
    Y = lambda v: pt + (hi - v) / (hi - lo) * (H - pt - pb)
    names = ["⅛", "¼", "½", "¾", "all"]
    grid = "".join(f'<line x1="{pl}" x2="{W - pr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="g"/>'
                   f'<text x="{pl - 8}" y="{Y(v) + 4:.1f}" class="yl">{v:.2f}</text>' for v in (0.58, 0.60, 0.62, 0.64, 0.66))
    band = (f'<rect x="{pl}" y="{Y(ce["limit_high"]):.1f}" width="{W - pl - pr}" height="{Y(ce["limit_low"]) - Y(ce["limit_high"]):.1f}" class="band"/>'
            f'<text x="{pl + 6}" y="{Y(ce["limit_high"]) - 6:.1f}" class="bl">estimated limit {ce["limit_low"]:.3f}–{ce["limit_high"]:.3f}</text>')
    fin = (f'<line x1="{pl}" x2="{W - pr}" y1="{Y(final_auc):.1f}" y2="{Y(final_auc):.1f}" class="fin"/>'
           f'<text x="{W - pr - 6}" y="{Y(final_auc) + 16:.1f}" class="fl">our ensemble {final_auc:.4f}</text>')
    d = "M" + " L".join(f"{X(f):.1f} {Y(v):.1f}" for f, v in zip(fr, au))
    pts = "".join(f'<g class="ce-p ce-p{i}"><line x1="{X(f):.1f}" x2="{X(f):.1f}" y1="{pt}" y2="{H - pb}" class="guide"/>'
                  f'<circle cx="{X(f):.1f}" cy="{Y(v):.1f}" r="5.5"/></g>'
                  for i, (f, v) in enumerate(zip(fr, au)))
    pos = [X(f) / W for f in fr]              # stops sit exactly under the points of the chart
    stops = "".join(f'<label class="ce-stop" style="left:{pos[i]:.2%}"><input type="radio" name="ce" id="ce-{i}"'
                    f'{" checked" if i == len(fr) - 1 else ""}><span>{names[i]}</span></label>' for i in range(len(fr)))
    reads, rules = "", []
    for i, (f, v) in enumerate(zip(fr, au)):
        gap = au[-1] - v
        txt = (f"With **{f:.0%}** of every customer's transactions the linear model on instrument levels reaches "
               f"**{v:.4f}** — {gap:.4f} below using all of them. Less history = noisier levels = lower AUC."
               if i < len(fr) - 1 else
               f"With **all** transactions the linear model reaches **{v:.4f}**. The curve is flattening towards "
               f"**{ce['limit_low']:.3f}–{ce['limit_high']:.3f}**; our ensemble ({final_auc:.4f}) is already inside it — the rest is "
               f"randomness in the labels.")
        reads += f'<p class="ce-read ce-r{i}">{_md(txt)}</p>'
        on = f".ui-ce:has(#ce-{i}:checked)"
        rules.append(f"{on} .ce-r{i}{{display:block}}{on} .ce-p{i} .guide{{opacity:1}}{on} .ce-p{i} circle{{fill:var(--sign);stroke:var(--sign);transform:scale(1.4)}}{on} .ce-fill{{width:{pos[i] - pos[0]:.2%}}}")
    return _style(rules) + (
        f'<section class="ui-ce ui-card-x"><div class="jr-head"><div><b>How much history does the signal need?</b>'
        f'<span>Move along the history used per customer.</span></div></div>'
        + _svg(f'<svg class="ce-svg" viewBox="0 0 {W} {H}" role="img" aria-label="ROC-AUC against share of transactions used">'
               f'{grid}{band}{fin}<path d="{d}" class="ce-line"/>{pts}</svg>', "ce", f"{W}/{H}") +
        f'<div class="ce-track" role="radiogroup" aria-label="Share of transactions">{stops}<span class="ce-fill" style="left:{pos[0]:.2%}"></span><span class="ce-rail" style="left:{pos[0]:.2%};right:{1 - pos[-1]:.2%}"></span></div>{reads}</section>')


def sim_card():
    """Comparison: an interactive 'alert risk simulator' built on the main finding.

    Four sliders set a customer's typical amount level on each instrument (in standard deviations),
    a fifth one shifts all four together (the common customer scale).  client.js recomputes, in the
    browser, the main contrast = cash in + card out − bank in − bank out, places it on the decile
    scale (≈ Φ(contrast / 2)) and shows the escalation rate observed in that decile of the train data.
    Moving the customer scale moves every thumb but leaves the risk unchanged — the key EDA finding."""
    sub = _dec[_dec["feature"] == FEATS[FEAT_DEFAULT]].sort_values("decile")
    rates, avg = sub["escalation_rate"].tolist(), ov["escalation_rate"]
    top = max(rates) * 1.08
    rows = [("cin", "+", "Cash deposits", 0.8), ("cout", "+", "Card spending", 0.6),
            ("bin", "−", "Bank transfers in", -0.2), ("bout", "−", "Bank transfers out", -0.4)]
    body = "".join(
        f'<label class="ui-sim-row" data-k="{k}"><span class="name"><b class="{"p" if sg == "+" else "m"}">{sg}</b>{name}</span>'
        f'<input class="ui-range" type="range" min="-3" max="3" step="0.1" value="{v}" aria-label="{name}, standard deviations">'
        f'<span class="out">{v:+.1f}σ</span></label>' for k, sg, name, v in rows)
    body += ('<label class="ui-sim-row scale" data-k="scale"><span class="name">Customer scale</span>'
             '<input class="ui-range" type="range" min="-1.5" max="1.5" step="0.1" value="0" '
             'aria-label="Customer scale: shifts all four levels together"><span class="out">+0.0σ</span></label>')
    bars = "".join(f'<i style="height:{r / top:.1%}" title="decile {i + 1}: {r:.1%}"></i>' for i, r in enumerate(rates))
    return (f'<div class="ui-sim" data-sim data-rates="{",".join(f"{r:.4f}" for r in rates)}" data-avg="{avg:.4f}">'
            f'<div class="ui-sim-head">{_icon("tune")}<div><b>Try it: alert risk simulator</b>'
            f'<span>Set a customer\'s typical amount on each instrument.</span></div></div>'
            f'<div class="ui-sim-grid"><div class="ui-sim-rows">{body}</div>'
            f'<div class="ui-sim-res"><div class="ui-sim-big"><div class="k">Escalation rate</div>'
            f'<div class="v" data-o="rate">{avg:.1%}</div><div class="d" data-o="dec">&nbsp;</div></div>'
            f'<div class="ui-sim-bars">{bars}<div class="avg" style="bottom:{avg / top:.1%}"></div></div>'
            f'<div class="ui-sim-note" data-o="note">Drag a slider — then try <b>Customer scale</b>.</div></div></div>'
            f'</div>')


def pca_recipe_html():
    """Comparison: PCA component recipe — loadings across 8 payment instruments."""
    pca = csv("pca.csv").head(6)
    insts = [("Card in", BLUE), ("Card out", BLUE), ("Bank in", ORANGE), ("Bank out", ORANGE),
             ("Cash in", AQUA), ("Cash out", AQUA), ("Intl in", VIOLET), ("Intl out", VIOLET)]
    tabs = "".join(
        f'<label class="pca-tab" style="position:relative"><input type="radio" name="pca-comp" id="pca-c-{i}" value="{i}"'
        f'{" checked" if row["component"] == "PC5" else ""}><span>{row["component"]}</span></label>'
        for i, row in pca.iterrows())
    panels = []
    rules = []
    for i, row in pca.iterrows():
        comp = row["component"]
        vs = row["variance_share"]
        auc = row["auc"]
        if comp == "PC5":
            desc = (f'<b>PC5</b> · {vs:.1%} variance · <b style="color:var(--hot)">AUC {auc:.3f}</b> '
                    f'— the contrast: <b style="color:var(--hot)">+cash in</b>, <b style="color:var(--hot)">+card out</b>, '
                    f'<b style="color:var(--cool)">−bank in</b>, <b style="color:var(--cool)">−bank out</b>.')
        elif comp == "PC1":
            desc = f'<b>PC1</b> · {vs:.1%} variance · AUC {auc:.3f} — common customer scale (all instruments negative).'
        else:
            desc = f'<b>{comp}</b> · {vs:.1%} variance · AUC {auc:.3f}'
        rows = []
        for name, col in insts:
            v = float(row[f"load_{name}"])
            w = min(1.0, abs(v)) * 50
            pos = v >= 0
            bar_color = ORANGE if pos else BLUE
            bar_style = f"left:50%;width:{w:.1f}%;background:{bar_color};border-radius:0 3px 3px 0;" if pos else f"right:50%;width:{w:.1f}%;background:{bar_color};border-radius:3px 0 0 3px;"
            rows.append(
                f'<div class="pca-row"><span class="pca-name"><i class="pca-dot" style="background:{col}"></i>{name}</span>'
                f'<div class="pca-track"><span class="pca-bar" style="{bar_style}"></span></div>'
                f'<b class="pca-val {"pos" if pos else "neg"}">{v:+.3f}</b></div>'
            )
        panels.append(
            f'<div class="pca-panel pca-p{i}"><div class="pca-desc">{desc}</div>'
            f'<div class="pca-axis"><span>−1</span><span>0</span><span>+1</span></div>'
            f'{"".join(rows)}</div>'
        )
        rules.append(f".ui-pca:has(#pca-c-{i}:checked) .pca-p{i}{{display:block}}")
        rules.append(f".ui-pca:has(#pca-c-{i}:checked) .pca-tab:has(#pca-c-{i}:checked){{background:var(--panel-2);border-color:var(--sign);color:var(--ink)}}")
    return _style(rules) + (
        f'<section class="ui-pca"><div class="pca-tabs">{tabs}</div>'
        f'<div class="pca-body">{"".join(panels)}</div></section>'
    )


def budget_card():
    """Conclusion: Analyst's budget — returns on reviewing top X% of alerts."""
    gdata = js("gains.json")
    points = gdata["points"]
    total_alerts = gdata["total_alerts"]
    total_esc = gdata["total_esc"]
    W, H, pl, pr, pt, pb = 600, 240, 44, 20, 16, 32
    X = lambda p: pl + (p / 100) * (W - pl - pr)
    Y = lambda g: pt + (1 - g) * (H - pt - pb)
    grid_h = "".join(f'<line x1="{pl}" x2="{W - pr}" y1="{Y(g / 100):.1f}" y2="{Y(g / 100):.1f}" class="bg-grid"/>'
                     f'<text x="{pl - 6}" y="{Y(g / 100) + 4:.1f}" class="bg-yl">{g}%</text>' for g in (25, 50, 75, 100))
    grid_v = "".join(f'<line x1="{X(p):.1f}" x2="{X(p):.1f}" y1="{pt}" y2="{H - pb}" class="bg-grid"/>'
                     f'<text x="{X(p):.1f}" y="{H - pb + 16:.1f}" class="bg-xl">{p}%</text>' for p in (25, 50, 75, 100))
    diag = (f'<line x1="{pl}" y1="{H - pb}" x2="{W - pr}" y2="{pt}" class="bg-diag"/>'
            f'<text x="{(W - pr) * 0.65:.1f}" y="{(H - pb) * 0.65:.1f}" class="bg-diag-lab" transform="rotate(-21, {(W - pr) * 0.65:.1f}, {(H - pb) * 0.65:.1f})">random selection</text>')
    path_d = "M" + " L".join(f"{X(pt_['p']):.1f} {Y(pt_['gains']):.1f}" for pt_ in points)
    area_d = f"{path_d} L{X(100):.1f} {H - pb:.1f} L{X(0):.1f} {H - pb:.1f} Z"
    p0 = points[30]
    cx0, cy0 = X(30), Y(p0["gains"])
    marker = (f'<g class="bg-marker" id="bg-marker">'
              f'<line class="bg-gv" id="bg-gv" x1="{cx0:.1f}" x2="{cx0:.1f}" y1="{cy0:.1f}" y2="{H - pb:.1f}"/>'
              f'<line class="bg-gh" id="bg-gh" x1="{pl:.1f}" x2="{cx0:.1f}" y1="{cy0:.1f}" y2="{cy0:.1f}"/>'
              f'<circle class="bg-dot" id="bg-dot" cx="{cx0:.1f}" cy="{cy0:.1f}" r="5.5"/>'
              f'</g>')
    svg_markup = (f'<svg class="bg-svg" viewBox="0 0 {W} {H}" role="img" aria-label="Gains curve: escalations caught vs alerts checked">'
                  f'<defs><linearGradient id="bg-grad" x1="0" y1="0" x2="0" y2="1">'
                  f'<stop offset="0%" stop-color="#00f2fe" stop-opacity="0.25"/>'
                  f'<stop offset="100%" stop-color="#00f2fe" stop-opacity="0.0"/>'
                  f'</linearGradient></defs>'
                  f'{grid_h}{grid_v}{diag}'
                  f'<path d="{area_d}" class="bg-area" fill="url(#bg-grad)"/>'
                  f'<path d="{path_d}" class="bg-path"/>'
                  f'{marker}'
                  f'<text x="{pl + 6}" y="{pt + 14}" class="bg-title-svg">Gains curve: escalations caught (%)</text>'
                  f'</svg>')
    return (f'<div class="ui-budget ui-card-x" data-budget>'
            f'<div class="ui-sim-head">{_icon("account_balance_wallet")}<div><b>Analyst budget: return on inspection effort</b>'
            f'<span>Move the slider to see how many escalations are caught by checking the top X% of alerts.</span></div></div>'
            f'<div class="bg-body">'
            f'<div class="bg-slider-row">'
            f'<span class="bg-slider-lab">Alerts checked: <b data-b="p">30 %</b></span>'
            f'<input class="ui-range bg-range" id="budget-range" type="range" min="1" max="100" value="30" aria-label="Percentage of alerts inspected">'
            f'<div class="bg-slider-axis"><span>1 % · riskiest first</span><span>100 % · all alerts</span></div>'
            f'</div>'
            f'<div class="bg-readout">Checking <b class="hot" data-b="p">30 %</b> of alerts catches '
            f'<b class="hot" data-b="caught">{p0["gains"]:.1%}</b> of all escalations (<b data-b="mult">{p0["gains"] / 0.3:.2f}×</b> to random selection).</div>'
            f'<div class="bg-chips">'
            f'<div class="bg-chip"><span>Alerts checked</span><b data-b="alerts">{p0["alerts_checked"]:,}</b><small>of {total_alerts:,}</small></div>'
            f'<div class="bg-chip"><span>Escalations caught</span><b class="hot" data-b="esc">{p0["esc_caught"]:,}</b><small>of {total_esc:,}</small></div>'
            f'<div class="bg-chip"><span>Workload avoided</span><b data-b="saved">70 %</b><small>alerts skipped</small></div>'
            f'</div>'
            f'{_svg(svg_markup, "bg", f"{W}/{H}")}'
            f'</div></div>')


def inspector_card():
    """Comparison: 30 real training alerts seen through the formula; the history slider limits the
    transactions used (client.js recomputes the four q75 levels, the contrast and its decile)."""
    flt = "".join(f'<label><input type="radio" name="insp-filter" value="{v}"{" checked" if v == "all" else ""}><span>{t}</span></label>'
                  for v, t in (("all", "All"), ("1", "Escalated"), ("0", "Dismissed")))
    term = lambda cls, name, oid: f'<span class="t {cls}"><em>{name}</em><b id="{oid}">—</b></span>'
    eq = (term("r", "contrast", "insp-contrast") + '<span class="op">→</span>'
          + term("r", "risk", "insp-decile") + '<span class="op">→</span>'
          + '<span class="t v" id="insp-truth-card"><em>actual</em><b id="insp-truth">—</b></span>')
    return (
        f'<div class="ui-insp" data-insp>'
        f'<div class="ui-sim-head">{_icon("manage_search")}<div><b>Alert inspector</b>'
        f'<span>30 real training alerts, seen through the formula.</span></div></div>'
        f'<div class="insp-body">'
        f'<div class="insp-top"><div class="tbx-seg" role="radiogroup" aria-label="Filter alerts">{flt}</div>'
        f'<div class="insp-nav"><button type="button" data-insp-step="-1" aria-label="Previous alert">{_icon("arrow_back")}</button>'
        f'<span id="insp-case">—</span>'
        f'<button type="button" data-insp-step="1" aria-label="Next alert">{_icon("arrow_forward")}</button></div></div>'
        f'<div class="insp-canvas-wrap"><canvas class="insp-canvas" id="insp-canvas" width="700" height="280"></canvas></div>'
        f'<div class="insp-legend"><span class="dash">dot = one transaction · bold tick = q75 level</span></div>'
        f'<div class="insp-hist"><span class="k">History used</span>'
        f'<input class="ui-range" id="insp-range" type="range" min="-179" max="0" value="0" step="1" aria-label="History used">'
        f'<b id="insp-cutoff-val">all 180 days</b></div>'
        f'<div class="insp-eq">{eq}</div><p class="insp-note" id="insp-truth-info"></p>'
        f'</div></div>'
    )


INSP_PATH = DATA / "inspector.json"
GAINS_PATH = DATA / "gains.json"
INSP_DATA_RAW = INSP_PATH.read_text(encoding="utf-8") if INSP_PATH.exists() else "{}"
GAINS_DATA_RAW = GAINS_PATH.read_text(encoding="utf-8") if GAINS_PATH.exists() else "{}"


# =====================================================================================
# App shell (first in the page, so the navigation works as soon as it arrives)
# =====================================================================================
with st.container(key="shell", gap=None):
    st.markdown(f"<style>{STYLE}\n{SWITCH_CSS}</style>", unsafe_allow_html=True)
    st.html(shell_html())
    # (the leading element keeps DOMPurify from hoisting a lone <script> into <head> and dropping it)
    st.html(f'<span hidden></span><script>'
            f'window.__INSP_DATA__ = {INSP_DATA_RAW};\n'
            f'window.__GAINS_DATA__ = {GAINS_DATA_RAW};\n'
            f'{CLIENT_JS}</script>', unsafe_allow_javascript=True)


# =====================================================================================
# Sections
# =====================================================================================
def s1_overview():
    head(0, "A monitoring unit receives automated alerts built from customers' transaction history; specialists "
            "either **dismiss** an alert or **escalate** it. We estimate the probability of escalation for every "
            "hidden-test alert. The metric is **ROC-AUC**, so only the *ranking* of alerts matters.")
    st.html(facts_html())
    note("We treat the task as **ranking alerts by escalation risk**. Each alert's ~460 transactions are collapsed "
         "into one row of features computed only from that alert's own history — mainly amount statistics per "
         "**payment instrument × direction** and an explicit contrast between instruments, which the EDA showed to be "
         "the core signal. On these features we train a rank-averaged ensemble of CatBoost, LightGBM, logistic "
         "regression and a hybrid model, validated with repeated stratified 5-fold CV and audited for leakage and "
         "overfitting.", "Our approach", "route")
    finding = [
        "Every customer has a common **[[scale of amounts (62 % of variance)|4|.st-key-tcard_pc|pc:0]]** that carries "
        "almost no escalation signal ([[AUC 0.535|4|.st-key-tcard_pc|pc:1]]).",
        "The true signal is a **behavioral contrast**: escalation risk rises when **[[cash deposits and card spending "
        "dominate over official bank transfers|4|.ui-sim]]** — a classic AML transit / cashing pattern.",
        "A simple rule based on this contrast achieves **[[ROC-AUC 0.638 without any training|4|.st-key-tcard_cx|cx:1]]**, proving the insight is "
        "structural. Our 5-model ensemble builds on this foundation to capture non-linear refinements, pushing the "
        "final score to **[[0.6533|5|.ui-lb]]**.",
    ]
    st.html(f'<section class="ui-finding"><div class="ui-finding-h">{_icon("lightbulb")}Main finding</div>'
            + "".join(f"<p>{_md(t)}</p>" for t in finding) + "</section>")
    st.html(journey_html())


def s2_dataset():
    head(1)
    title("Two tables, one link")
    st.html(schema_html())

    title("How much each alert carries")
    v = tabbed("ds", ["Per alert", "Instrument mix"])
    with v[0]:
        @chart("s2_tx", 340, bare=True)
        def _():
            h = csv("tx_per_alert_hist.csv")
            fig = px.bar(h, x="tx_from", y="alerts", color_discrete_sequence=[BLUE],
                         labels={"tx_from": "transactions per alert", "alerts": "alerts"},
                         title=f"Transactions per alert — median {ov['tx_per_alert_median']:.0f}")
            fig.update_traces(marker_line_width=0, hovertemplate="%{x}–%{x}+50 tx: %{y} alerts<extra></extra>")
            med = ov["tx_per_alert_median"]
            fig.add_vline(x=med, line_dash="dot", line_color=INK_3)
            callout(fig, med, h["alerts"].max() * 0.92, f"median {med:.0f}", ax=60, ay=-10, xanchor="left")
            return fig
    with v[1]:
        @chart("s2_comp", 340, bare=True)
        def _():
            comp = csv("composition_type_direction.csv")
            fig = px.bar(comp, y="type", x="share", color="direction", orientation="h", barmode="group",
                         color_discrete_map=DIR_COLORS, title="Share of all transactions by instrument and direction",
                         labels={"share": "share of transactions", "type": ""},
                         category_orders={"type": ["Card", "Bank transfer", "Cash", "International"]})
            fig.update_xaxes(tickformat=".0%")
            fig.update_traces(hovertemplate="%{y}: %{x:.1%}<extra></extra>")
            return fig

    title("Every instrument has its own normal amount")
    v = tabbed("dir", DIRECTIONS, "Transaction size by instrument")
    for i, direction in enumerate(DIRECTIONS):
        with v[i]:
            @chart(f"s2_hist_{direction}", 340, bare=True)
            def _():
                hist = csv("amount_hist.csv")
                sub = hist[hist["cross"].str.endswith(direction.lower())].copy()
                sub["instrument"] = sub["cross"].str.split(" · ").str[0]
                fig = px.line(sub, x="x", y="density", color="instrument", color_discrete_map=TYPE_COLORS,
                              labels={"x": "miqdor_indeksi (standardised log-amount)", "density": "density"},
                              title=f"{direction} transactions")
                fig.update_traces(line_width=2, hovertemplate="%{x:.2f}: %{y:.3f}<extra></extra>")
                nm = csv("amount_norms.csv").set_index("cross")["mean"]
                for inst, ax in (("Card", -40), ("International", 40)):
                    cur = sub[sub["instrument"] == inst]
                    top = cur.loc[cur["density"].idxmax()]
                    callout(fig, top["x"], top["density"], f"{inst}: norm {nm[f'{inst} · {direction.lower()}']:+.2f}", ax=ax, ay=-30)
                return fig
    norms = csv("amount_norms.csv")
    st.html(table_html(["instrument · direction", "transactions", "share", "mean", "std", "median", "max"],
                       [[r.cross, f"{r['count']:,.0f}", f"{r.share_of_tx:.1%}", f"{r['mean']:+.2f}", f"{r['std']:.2f}",
                         f"{r['median']:+.2f}", f"{r['max']:.3f}"] for _, r in norms.iterrows()], num=(1, 2, 3, 4, 5, 6)))
    note("Card incoming sits around −0.54, international around +2. Averaging all transactions together mixes these "
         "norms and hides information — this observation later produced our largest single improvement. Four exact "
         "values repeat many times (4.1835, 4.8628, 6.4304, 6.6917): they are per-instrument **caps** of the "
         "generator, unrelated to the target.")


def s3_target():
    head(2)
    st.html(waffle_html())
    rate = ov["escalation_rate"]
    title("Does the rate drift?")
    v = tabbed("tg", ["By quarter", "By ID block"])
    with v[0]:
        @chart("s3_quarter", 320, bare=True)
        def _():
            q = csv("target_by_quarter.csv")
            fig = px.bar(q, x="quarter", y="rate", color_discrete_sequence=[BLUE], custom_data=["alerts"],
                         title="Escalation rate by alert quarter", labels={"rate": "escalation rate", "quarter": ""})
            fig.add_hline(y=rate, line_dash="dot", line_color=GRAY, annotation_text=f"overall {rate:.1%}")
            fig.update_yaxes(tickformat=".0%", range=[0, 0.25])
            fig.update_traces(hovertemplate="%{x}: %{y:.1%} of %{customdata[0]} alerts<extra></extra>")
            tag(fig, f"every quarter within {q['rate'].min():.1%}–{q['rate'].max():.1%}")
            return fig
    with v[1]:
        @chart("s3_idblock", 320, bare=True)
        def _():
            ib = csv("target_by_id_block.csv")
            ib["id_block"] = ib["id_block"] + 1
            fig = px.bar(ib, x="id_block", y="rate", color_discrete_sequence=[BLUE],
                         title="Escalation rate by 20 equal blocks of signal_id",
                         labels={"rate": "escalation rate", "id_block": "signal_id block"})
            fig.add_hline(y=rate, line_dash="dot", line_color=GRAY)
            fig.update_yaxes(tickformat=".0%", range=[0, 0.25])
            fig.update_traces(hovertemplate="block %{x}: %{y:.1%}<extra></extra>")
            tag(fig, "only sampling noise around the overall rate")
            return fig
    note("Quarterly rates stay within 15.8–18.9 % and the ID blocks vary only by sampling noise, so neither time nor "
         "the identifier encodes the target. **No resampling** (ROC-AUC is rank-based) and **no time-based split** — "
         "StratifiedKFold with repeated seeds, because a single split varied by ±0.015 AUC, more than the effects we "
         "were measuring.", "Decision", "check_circle")


def s4_time():
    head(3)
    title("A burst in the last three minutes before every alert")

    @chart("s4_before", 330)
    def _():
        ab = csv("activity_before_alert.csv")
        fig = px.bar(ab, x="before_alert", y="per_hour", log_y=True, color_discrete_sequence=[BLUE],
                     custom_data=["transactions"], title="Transaction intensity by time before the alert (log scale)",
                     labels={"before_alert": "time before alert", "per_hour": "transactions per hour (all alerts)"})
        fig.update_traces(hovertemplate="%{x}: %{customdata[0]:,} transactions<extra></extra>")
        callout(fig, "2–3m", ab["per_hour"].iloc[2], "the burst: last 3 minutes", ax=70, ay=-6, xanchor="left")
        return fig

    bs = _csv("burst_size_stats.csv").iloc[0]
    st.markdown(
        f"**{ov['alerts_with_burst']:.0%}** of alerts contain a burst — median **{bs['50%']:.0f}** operations within "
        f"~3 minutes before the alert ({ov['burst_share_of_tx']:.1%} of all transactions), then almost nothing "
        "for the preceding hours. This is the activity that most likely *triggered* the alert.")
    # the one piece of evidence for "the burst adds nothing": linear-model AUC with and without it
    st.html('<section class="ui-card-x ui-pad ui-burst"><b class="ui-card-title">Does the burst help predict escalation?</b>'
            + meter_html([("history only", 0.637, False), ("history + burst", 0.637, False),
                          ("burst shift vs history", 0.525, True)], 0.5, 0.66)
            + '<p class="ui-small">Linear-model AUC. The burst is card-heavier and smaller in amount, but adds '
              'nothing: its smaller amounts are simply the end of the common trend.</p></section>')
    title("Timestamps are uniform — no human rhythm")
    v = tabbed("tm", ["Hour", "Weekday"])
    with v[0]:
        @chart("s4_hour", 300, bare=True)
        def _():
            hs = csv("hour_share.csv")
            fig = px.bar(hs, x="hour", y="share", color_discrete_sequence=[BLUE], title="Share of transactions by hour of day",
                         labels={"share": "share of transactions"})
            fig.add_hline(y=1 / 24, line_dash="dot", line_color=GRAY, annotation_text="uniform 1/24")
            fig.update_yaxes(tickformat=".1%", range=[0, 0.072])
            ns = csv("night_share_by_type.csv")["night_share"]
            tag(fig, f"flat: no day / night rhythm<br>every instrument does {ns.min() * 100:.1f}–{ns.max():.1%} of its "
                     "transactions at night (00–06) = 6 of 24 hours")
            return fig
    with v[1]:
        @chart("s4_dow", 300, bare=True)
        def _():
            dw = csv("dow_share.csv")
            fig = px.bar(dw, x="day", y="share", color_discrete_sequence=[BLUE], title="Share of transactions by weekday",
                         labels={"share": "share", "day": ""})
            fig.add_hline(y=1 / 7, line_dash="dot", line_color=GRAY)
            fig.update_yaxes(tickformat=".0%", range=[0, 0.2])
            tag(fig, "weekends look exactly like weekdays")
            return fig
    order = js("order_randomness.json")
    st.caption(f"The order of instruments is random too: the 'stickiness' of a customer's sequence is "
               f"{order['stickiness_median']:.3f} (1.0 = random) and predicts nothing (AUC {order['stickiness_auc']:.3f}).")

    title("A common drift toward the alert — identical in both classes")

    @chart("s4_trend", 340)
    def _():
        tr = csv("amount_trend_by_class.csv").melt(id_vars="days_before_alert", var_name="class", value_name="dev")
        fig = px.line(tr, x="days_before_alert", y="dev", color="class", markers=True,
                      color_discrete_map=CLASS_COLORS, title="Amount vs. the customer's own level",
                      labels={"days_before_alert": "time before alert", "dev": "deviation (log-amount)"})
        fig.update_xaxes(autorange="reversed")
        fig.update_traces(line_width=2, marker_size=8, hovertemplate="%{x}: %{y:+.3f}<extra></extra>")
        tag(fig, "the two classes overlap: gap ≤ 0.012", x=0.99, xanchor="right")
        return fig

    note("Amounts drift down by ≈0.4 log-units over the 180 days before an alert (the calendar date has no effect, "
         "−0.02 per year), but escalated and dismissed customers **overlap almost perfectly** (difference ≤ 0.012). "
         "Time-of-day, rolling windows and 'recent vs historical' "
         "features were built and tested — all gave ≤ 0 AUC and were dropped.", "Decision", "check_circle")


def s5_differences():
    head(4)
    title("Single features, split into deciles")
    st.html(select_html("feat", "Feature", FEATS, FEAT_DEFAULT))
    variants = st.container(key="feat", gap=None)
    for i, f in enumerate(FEATS):
        with variants.container(key=f"feat_{i}"):
            @chart(f"s5_dec_{i}", 340)
            def _():
                dec = csv("deciles.csv")
                sub = dec[dec["feature"] == f]
                fig = px.bar(sub, x="decile", y="escalation_rate",
                             title=f"{f} — single-feature AUC {sub['auc'].iloc[0]:.3f}",
                             labels={"decile": "decile of the feature (1 = lowest)", "escalation_rate": "escalation rate"})
                fig.add_hline(y=ov["escalation_rate"], line_dash="dot", line_color=GRAY, annotation_text="overall")
                fig.update_yaxes(tickformat=".0%")
                fig.update_xaxes(dtick=1)
                r = sub["escalation_rate"].to_numpy()
                fig.update_traces(hovertemplate="decile %{x}: %{y:.1%}<extra></extra>",
                                  marker_color=[ORANGE if v > ov["escalation_rate"] else MUTED for v in r])
                hi, lo = int(r.argmax()), int(r.argmin())
                callout(fig, hi + 1, r[hi], f"{r[hi] / r[lo]:.1f}× decile {lo + 1}", ax=-50 if hi > 5 else 50, ay=-24)
                return fig
    st.caption("Switch the feature: volume (number of transactions, international count) barely matters — comparing "
               "amount **levels between instruments** separates the classes far better.")

    title("Why comparing instruments works: a common customer scale")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        @chart("s5_corr", 400)
        def _():
            cm = csv("corr_instrument_means.csv").set_index("instrument")
            fig = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, colorscale=SEQ, zmin=0.4, zmax=1,
                                       text=cm.values, texttemplate="%{text:.2f}", textfont=dict(color="#FFFFFF", size=12),
                                       xgap=2, ygap=2, hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"))
            fig.update_layout(title="Correlation of a customer's mean amount between instruments")
            fig.update_yaxes(autorange="reversed")
            return fig
    with c2:
        v = tabbed("pc", ["Variance", "Predictive power", "Recipe"], "Principal components")
        with v[0]:
            @chart("s5_pca_var", 330, bare=True)
            def _():
                pca = csv("pca.csv").head(6)
                fig = px.bar(pca, x="component", y="variance_share",
                             title="Share of variance per component", labels={"variance_share": "variance share", "component": ""})
                fig.update_traces(marker_color=[BLUE if c == "PC1" else MUTED for c in pca["component"]])
                fig.update_yaxes(tickformat=".0%")
                callout(fig, "PC1", pca["variance_share"].iloc[0], "customer scale: 62 %", ax=70, ay=10, xanchor="left")
                return fig
        with v[1]:
            @chart("s5_pca_auc", 330, bare=True)
            def _():
                pca = csv("pca.csv").head(6)
                fig = px.bar(pca, x="component", y="auc",
                             title="ROC-AUC of each component alone", labels={"auc": "ROC-AUC", "component": ""})
                fig.update_traces(marker_color=[ORANGE if c == "PC5" else MUTED for c in pca["component"]])
                fig.update_yaxes(range=[0.5, 0.64])
                callout(fig, "PC5", pca["auc"].iloc[4], "PC5: 4 % of variance,<br>AUC 0.617", ax=-80, ay=-8, xanchor="right")
                return fig
        with v[2]:
            st.html(pca_recipe_html())
    note("Mean amounts of the same customer are strongly correlated across instruments (0.48–0.88): some customers "
         "simply transact large amounts, others small ones. This **customer scale (PC1) explains 62 % of the variance "
         "but has AUC 0.535**. A small component (PC5, 4 % of variance) has **AUC 0.617**: it contrasts bank transfers "
         "against card spending and cash deposits. Trees split on single means, which are dominated by the useless "
         "scale — so we had to hand the contrast to the model explicitly.", "Why it matters", "lightbulb")

    st.html(sim_card())
    st.html(inspector_card())

    title("The main contrast, from three angles",
          "cash in + card out − bank in − bank out: one formula with fixed ±1 weights.")
    v = tabbed("cx", ["Model weights", "Quantile", "History window"])
    with v[0]:
        @chart("s5_lr", 340, bare=True)
        def _():
            w = csv("lr_weights.csv")
            w["effect"] = w["weight"].map(lambda x: "raises escalation risk" if x > 0 else "lowers escalation risk")
            fig = px.bar(w, y="instrument", x="weight", color="effect", orientation="h",
                         color_discrete_map={"raises escalation risk": ORANGE, "lowers escalation risk": BLUE},
                         title="Logistic regression on 8 standardised instrument means — the signs give the formula",
                         labels={"weight": "coefficient", "instrument": ""})
            fig.update_traces(hovertemplate="%{y}: %{x:+.3f}<extra></extra>")
            return fig
    with v[1]:
        @chart("s5_quant", 340, bare=True)
        def _():
            cq = csv("contrast_by_quantile.csv")
            fig = px.line(cq, x="quantile", y="auc", markers=True, color_discrete_sequence=[ORANGE],
                          title="Strongest around the 80th percentile: a shift of the whole distribution",
                          labels={"auc": "ROC-AUC (no training)", "quantile": "quantile used per instrument"})
            fig.update_xaxes(type="category")
            fig.update_traces(line_width=2, marker_size=9, hovertemplate="q=%{x}: %{y:.4f}<extra></extra>")
            best = cq.loc[cq["auc"].idxmax()]
            callout(fig, str(best["quantile"]), best["auc"], f"peak {best['auc']:.3f} at q = {best['quantile']}", ay=36)
            return fig
    with v[2]:
        @chart("s5_window", 340, bare=True)
        def _():
            cw = csv("contrast_by_window.csv")
            fig = px.bar(cw, x="window", y="auc", color_discrete_sequence=[BLUE], custom_data=["coverage"],
                         title="Every month of history shows it — more history just means less noise",
                         labels={"auc": "ROC-AUC", "window": ""})
            fig.update_yaxes(range=[0.5, 0.65])
            fig.update_traces(hovertemplate="%{x}: AUC %{y:.4f}, available for %{customdata[0]:.0%} of alerts<extra></extra>")
            tag(fig, "every window carries the signal")
            return fig


def s6_model():
    head(5)
    title("From observation to decision")
    st.html(explorer_html())

    c1, c2 = st.columns([1, 1.15])
    with c1:
        title("What each alert becomes")
        st.html(features_html())
    with c2:
        title("Models in the ensemble")
        st.html(leaderboard_html())
        st.caption("Each model is trained 15 times (3 seeds × 5 folds) with early stopping on a separate 20 % slice. "
                   "The rule — equal weights for every model above 0.640 — was fixed **before** looking at blend results.")

    title("Ideas we tested and rejected", "Accepted only if it adds ≥ +0.001 AUC consistently across seeds.")
    st.html(rejected_html())

    title("The audit")
    c1, c2 = st.columns([1, 1.1])
    with c1:
        st.html(audit_html())
    with c2:
        @chart("s6_splits", 380)
        def _():
            sp = pd.DataFrame(R["audit"]["splits"])
            sp["split"] = sp["split"].astype(str)
            spm = sp.melt(id_vars="split", var_name="model", value_name="gap")
            spm["model"] = spm["model"].map({"final_gap": "Final model",
                                             "control_gap": "Control (built before any label-based decisions)"})
            fig = px.bar(spm, x="split", y="gap", color="model", barmode="group",
                         color_discrete_map={"Final model": ORANGE,
                                             "Control (built before any label-based decisions)": BLUE},
                         title="Hold-out AUC minus CV AUC on 8 fresh 10k / 4k splits",
                         labels={"gap": "hold-out − CV", "split": "random split seed"})
            fig.add_hline(y=0, line_color=GRAY)
            fig.update_xaxes(type="category")
            fig.update_traces(hovertemplate="split %{x}: %{y:+.4f}<extra></extra>")
            tag(fig, "final ≈ control: no overfitting to the CV")
            return fig


def s7_conclusion():
    head(6)
    findings = [
        ("The signal is a contrast between payment instruments, not volume or timing.",
         "Escalation is more likely when cash deposits and card spending are large relative to bank transfers. "
         "A single fixed formula reaches ROC-AUC 0.638."),
        ("A common customer scale hides it.",
         "Instrument means are strongly correlated (62 % of variance) but that scale is almost useless for "
         "prediction; separating instruments gave our largest jump (+0.039)."),
        ("Time carries no signal in this data.",
         "Timestamps are uniform, instrument order is random, and the downward trend of amounts is identical for "
         "both classes. The 3-minute burst before each alert is real but uninformative."),
        ("The final model is audited, not just tuned.",
         f"An equal-weight rank ensemble of CatBoost ×2, LightGBM, logistic regression and a hybrid — "
         f"CV ROC-AUC {final_auc:.4f}, checked for leakage, train / test consistency, stability and selection bias."),
    ]
    proof = ["[[Evidence|4|.st-key-feat]]", "[[Evidence|4|.st-key-tcard_pc|pc:1]]",
             "[[Evidence|3|.st-key-card_s4_trend]]", "[[Evidence|5|.ui-au]]"]
    st.html('<ol class="ui-findings">' + "".join(f'<li><b>{html.escape(h)}</b><p>{html.escape(t)} {_md(e)}</p></li>'
                                                 for (h, t), e in zip(findings, proof)) + "</ol>")
    st.html(budget_card())
    st.html(ceiling_html())


def _photo(m):
    return _team_img(m.get("photo", ""))


def _ref(label, url):
    """A mono reference link, or a muted 'soon' when there is no URL yet."""
    if url:
        return f'<a class="dz-ref" href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">{html.escape(label)} ↗</a>'
    return f'<span class="dz-ref off" title="no link yet">{html.escape(label)} · soon</span>'


def _step_chip(i):
    """A step of the case as a chip; clicking it opens that step (a label for the nav radio)."""
    if i is None:
        return '<span class="dz-step web">web</span>'
    return (f'<label class="dz-step" for="tbx-sec-{i}" tabindex="0" role="link">'
            f'{i + 1:02d} · {html.escape(SECTIONS[i][0])}</label>')


def s8_team():
    """The team page, styled as the credits of the case file: who worked on which step, then one
    dossier per investigator.  Reached from the rail avatars, the dock, key T or #team."""
    refs = "".join(_ref(t, u) for t, u in TEAM_LINKS)
    st.html(f'<header class="ui-head ui-team-head"><span class="ui-num" aria-hidden="true">A+</span>'
            f'<div class="ui-head-body"><div class="ui-kicker">Team</div><h1>Who worked on this case?</h1>'
            f'<p class="ui-answer"><span class="lab">Team</span><span class="txt">{len(TEAM)} investigators, one case file — '
            f'<b>tobaskA+ · {TEAM_ID}</b>.</span></p><div class="dz-refs">{refs}</div></div></header>')

    # coverage map: the 7 steps of the case × the investigators
    head = "".join(f'<label class="cv-h" for="tbx-sec-{i}" tabindex="0" role="link" title="{html.escape(n)}">'
                   f'<b>{i + 1:02d}</b><span>{html.escape(n)}</span></label>' for i, (n, _, _) in enumerate(SECTIONS))
    rows = ""
    for k, m in enumerate(TEAM):
        steps = {s_ for s_, _ in m["log"] if s_ is not None}
        cells = "".join(f'<i class="{"on" if i in steps else ""}" title="{html.escape(m["name"])} · {i + 1:02d}"></i>'
                        for i in range(len(SECTIONS)))
        rows += (f'<div class="cv-row"><span class="cv-who"><span class="cv-av" style="--h:{k}">{_face(m, k)}</span>'
                 f'{html.escape(m["name"])}</span>{cells}</div>')
    st.html(f'<section class="ui-cv"><div class="cv-title">Case coverage</div>'
            f'<div class="cv-grid" style="--n:{len(SECTIONS)}"><div class="cv-row cv-top"><span></span>{head}</div>{rows}</div></section>')

    # one dossier per investigator
    out = ""
    for k, m in enumerate(TEAM):
        ph, back = _photo(m), _team_img(m.get("photo_back", ""))
        face = (f'<img src="{ph}" alt="{html.escape(m["name"])}">' if ph
                else f'<span class="dz-ini" style="--h:{k}">{html.escape(m["initials"])}</span>')
        if ph and back:                      # a second photo flips in on hover (tap on phones)
            face = (f'<span class="fl-in"><img class="fr" src="{ph}" alt="{html.escape(m["name"])}">'
                    f'<img class="bk" src="{back}" alt=""></span>')
        log = "".join(f'<li>{_step_chip(s_)}<span>{_md(t)}</span></li>' for s_, t in m["log"])
        out += (f'<article class="dz"><div class="dz-id"><span class="dz-no" aria-hidden="true">{k + 1:02d}</span>'
                f'<figure class="dz-ph{" flip" if ph and back else ""}"{" tabindex=0" if ph and back else ""}>{face}</figure>'
                '</div>'
                f'<div class="dz-main"><div class="dz-k">Investigator {k + 1:02d}</div><h3>{html.escape(m["name"])}</h3>'
                f'<div class="dz-role"><span>Assignment</span>{html.escape(m["role"])}</div>'
                + (f'<p class="dz-bio">{_md(m["bio"])}</p>' if m.get("bio") else "") +
                f'<ul class="dz-log">{log}</ul>'
                f'<div class="dz-refs">{_ref("GitHub", m.get("github", ""))}{_ref("LinkedIn", m.get("linkedin", ""))}</div></div></article>')
    st.html(f'<section class="dz-list">{out}</section>')


# =====================================================================================
# Page: all sections are rendered once per session; CSS shows the chosen one.
# =====================================================================================
RENDER = [s1_overview, s2_dataset, s3_target, s4_time, s5_differences, s6_model, s7_conclusion, s8_team]
pages = st.container(key="sec", gap=None)
slots = [pages.container(key=f"sec_{i}") for i in range(len(RENDER))]
for slot, render in zip(slots, RENDER):
    with slot:
        render()
