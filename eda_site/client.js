// Client-side behaviour of the zero-latency controls (see app.py).
// A click on a section / chart tab only flips html[data-tbx-GROUP]; static CSS then shows the
// matching, already-drawn container in the same frame (and plays the enter transition).
// No server round-trip, no Streamlit rerun, no React render.
// Also: the case timeline (seen steps, phone dock + sheet), #section deep links, keyboard
// shortcuts, the score-journey replay, count-up numbers, the alert risk simulator, the splash
// screen, evidence links (jump to a step and highlight the chart) and swipe navigation on phones.
// NB: keep this file free of "<" followed by a letter or "/": DOMPurify (st.html) drops such scripts.
(function () {
  const doc = document;
  const win = window;
  if (win.__tobaskClient) return;
  win.__tobaskClient = true;
  const root = doc.documentElement;
  const baseTitle = doc.title || "Alert Escalation · EDA";
  const reduce = win.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const N = function () { return doc.querySelectorAll(".ui-node").length || 7; };   // steps of the case
  const TEAM = function () { return N(); };        // the team page: a section, but not a step
  let lastStep = 0;                                 // where "back" from the team page returns
  const cur = function () { return Number(root.getAttribute("data-tbx-sec") || "0"); };

  function nodeOf(v) {
    const input = doc.getElementById("tbx-sec-" + v);
    return input ? input.closest("label") : null;
  }

  // ---- count-up of big numbers ----
  function countUp(scope) {
    if (reduce || !scope) return;
    for (const el of scope.querySelectorAll("[data-count]")) {
      const final = el.getAttribute("data-final") || el.textContent;
      el.setAttribute("data-final", final);
      const m = /^([^0-9]*)([0-9][0-9,]*(?:[.][0-9]+)?)(.*)$/.exec(final);
      if (!m) continue;
      const target = Number(m[2].replace(/,/g, ""));
      const dec = (m[2].split(".")[1] || "").length, comma = m[2].indexOf(",") !== -1;
      const t0 = performance.now(), dur = 900;
      (function step(now) {
        const k = Math.min(1, Math.max(0, (now - t0) / dur)), e = 1 - Math.pow(1 - k, 3);
        let txt = (target * e).toFixed(dec);
        if (comma) txt = Number(txt).toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec });
        el.textContent = k === 1 ? final : m[1] + txt + m[3];
        if (k !== 1) win.requestAnimationFrame(step);
      })(t0);
    }
  }

  // ---- section change: scroll, timeline state, dock, title, hash ----
  function onSection(v, fromUser) {
    hydrateSvgs();
    const main = doc.querySelector('[data-testid="stMain"]');
    for (const el of [doc.scrollingElement, main]) if (el && el.scrollTop) el.scrollTop = 0;
    const n = N(), i = Number(v), isTeam = i === TEAM();
    if (!isTeam) lastStep = i;
    const node = isTeam ? null : nodeOf(v);
    if (node) node.classList.add("seen");
    const sheet = doc.getElementById("ui-sheet");
    if (sheet) sheet.checked = false;
    const name = isTeam ? "Team" : node ? node.querySelector(".t").textContent : "";
    const dn = doc.querySelector(".ui-dock-n"), dt = doc.querySelector(".ui-dock-t");
    if (dn) dn.textContent = isTeam ? "" : String(i + 1).padStart(2, "0") + " / " + String(n).padStart(2, "0");
    if (dt) dt.textContent = name;
    for (const b of doc.querySelectorAll(".ui-dock-btn")) {
      const back = Number(b.getAttribute("data-step")) === -1;
      b.disabled = isTeam ? !back : (back ? i === 0 : i === n - 1);
    }
    doc.title = name && i ? name + " · " + baseTitle : baseTitle;
    countUp(doc.querySelector(".st-key-sec_" + v));
    if (fromUser) {
      try { history.replaceState(null, "", isTeam ? "#team" : "#s" + (i + 1)); } catch (e) { /* sandboxed */ }
    }
    if (i === 4 && win.__renderAlert__) win.setTimeout(win.__renderAlert__, 60);
    if (i === 6 && win.__renderBudget__) win.setTimeout(function () { win.__renderBudget__(30); }, 60);
  }
  function go(i) {
    const t = doc.getElementById("tbx-sec-" + i);
    if (t && i !== cur()) t.click();
  }
  function stepBy(d) {
    if (cur() === TEAM()) { if (d < 0) go(lastStep); return; }
    const to = cur() + d;
    if (to >= 0 && to < N()) go(to);
  }
  function toggleTeam() { go(cur() === TEAM() ? lastStep : TEAM()); }

  doc.addEventListener("change", function (e) {
    const t = e.target;
    if (!t || t.type !== "radio" || !t.name) return;
    if (t.name === "jr") stopPlay();          // a real click (replay sets .checked without events)
    if (t.name.indexOf("tbx-") !== 0) return;
    const group = t.name.slice(4);
    if (group === "sec") root.setAttribute("data-dir", Number(t.value) === TEAM() || Number(t.value) > cur() ? "next" : "prev");
    root.setAttribute("data-tbx-" + group, t.value);
    if (group === "sec") onSection(t.value, true);
    const sel = t.closest(".tbx-select");
    if (sel) { sel.setAttribute("data-v", t.value); sel.open = false; }
  }, true);

  // deep link: #s3 opens section 3 (on load and when only the hash changes)
  function fromHash() {
    const h = win.location.hash || "", m = /^#s(\d+)$/.exec(h);
    const want = h === "#team" ? doc.getElementById("tbx-sec-" + TEAM())
      : m && Number(m[1]) <= N() && doc.getElementById("tbx-sec-" + (Number(m[1]) - 1));
    if (want && root.getAttribute("data-tbx-sec") !== want.value) {
      want.checked = true; root.setAttribute("data-tbx-sec", want.value); onSection(want.value, false);
    }
  }
  fromHash();
  win.addEventListener("hashchange", fromHash);

  // clicks made before this script arrived
  for (const t of doc.querySelectorAll("input[type=radio][name^='tbx-']:checked"))
    root.setAttribute("data-tbx-" + t.name.slice(4), t.value);

  // if Streamlit ever re-mounts a control (e.g. after a websocket reconnect), put the
  // radios back in sync with the page state instead of silently resetting them
  // st.html drops inline SVG, so charts drawn by app.py arrive as data-svg and are inserted here
  function hydrateSvgs() {
    for (const host of doc.querySelectorAll("[data-svg]")) {
      host.innerHTML = host.getAttribute("data-svg");
      host.removeAttribute("data-svg");
    }
  }
  hydrateSvgs();

  let queued = false;
  function restore() {
    queued = false;
    hydrateSvgs();
    for (const a of root.getAttributeNames()) {
      if (a.indexOf("data-tbx-") !== 0) continue;
      const v = root.getAttribute(a), name = "tbx-" + a.slice(9);
      const want = doc.querySelector("input[name='" + name + "'][value='" + v + "']");
      if (want && !want.checked) {
        want.checked = true;
        const sel = want.closest(".tbx-select");
        if (sel) sel.setAttribute("data-v", v);
      }
    }
  }
  new MutationObserver(function () {
    hydrateSvgs();
    if (!queued) { queued = true; win.requestAnimationFrame(restore); }
  }).observe(doc.body, { childList: true, subtree: true });

  // ---- clicks: dock arrows, replay, dropdown outside-click ----
  doc.addEventListener("click", function (e) {
    const step = e.target.closest && e.target.closest("[data-step]");
    if (step) { stepBy(Number(step.getAttribute("data-step"))); return; }
    const play = e.target.closest && e.target.closest("[data-play]");
    if (play) startPlay(play);
    const flip = e.target.closest && e.target.closest(".dz-ph.flip");   // team photo: tap turns it over
    if (flip) flip.classList.toggle("on");
  });
  doc.addEventListener("pointerdown", function (e) {
    for (const d of doc.querySelectorAll(".tbx-select[open]")) if (!d.contains(e.target)) d.open = false;
  }, true);

  // ---- evidence links: open the step, pick the right chart tab, scroll to it and flash it ----
  function reveal(sel, tab) {
    if (tab) {
      const parts = tab.split(":");
      const input = doc.querySelector("input[name='tbx-" + parts[0] + "'][value='" + parts[1] + "']");
      if (input && !input.checked) input.click();
    }
    const el = sel && doc.querySelector(sel);
    if (!el) return;
    const main = doc.querySelector('[data-testid="stMain"]');
    const off = win.innerWidth > 1099 ? 28 : 16;
    if (main) main.scrollTop += el.getBoundingClientRect().top - off;
    el.classList.remove("ui-flash");
    void el.offsetWidth;                    // restart the animation
    el.classList.add("ui-flash");
    win.setTimeout(function () { el.classList.remove("ui-flash"); }, 1900);
  }
  doc.addEventListener("click", function (e) {
    const a = e.target.closest && e.target.closest("a.ui-ref");
    if (!a) return;
    e.preventDefault();
    const m = /#s(\d+)$/.exec(a.getAttribute("href") || "");
    const to = m ? Number(m[1]) - 1 : cur();
    const sel = a.getAttribute("data-target"), tab = a.getAttribute("data-tab");
    if (to !== cur()) { go(to); win.setTimeout(function () { reveal(sel, tab); }, 60); }
    else reveal(sel, tab);
  }, true);

  // ---- swipe left / right on phones = next / previous step ----
  let sx = 0, sy = 0, st = 0, swipeOk = false;
  const noSwipe = "input, .tbx-seg, .jr-chips, .ui-table, .ce-track, .js-plotly-plot, .ui-rail, .ui-dock, .tbx-select, .wf-grid";
  doc.addEventListener("touchstart", function (e) {
    const t = e.touches[0];
    swipeOk = e.touches.length === 1 && !(e.target.closest && e.target.closest(noSwipe));
    sx = t.clientX; sy = t.clientY; st = Date.now();
  }, { passive: true });
  doc.addEventListener("touchend", function (e) {
    if (!swipeOk) return;
    const t = e.changedTouches[0];
    const dx = t.clientX - sx, dy = t.clientY - sy;
    if (Date.now() - st > 700 || Math.abs(dx) < 70 || Math.abs(dy) > Math.abs(dx) * 0.6) return;
    stepBy(dx < 0 ? 1 : -1);
  }, { passive: true });

  // ---- splash: shows how many steps have arrived, disappears once the current one is there ----
  function splash() {
    if (root.hasAttribute("data-ready")) return true;
    const heads = doc.querySelectorAll(".ui-head:not(.ui-team-head)").length, n = N();
    const bar = doc.querySelector(".sp-bar i");
    if (bar && heads) { root.setAttribute("data-sp", ""); bar.style.width = Math.round(heads / n * 100) + "%"; }
    const t = doc.querySelector(".sp-t");
    if (t && heads) t.textContent = "Opening the case file \u00b7 " + heads + " / " + n;
    const here = doc.querySelector(".st-key-sec_" + cur() + " .ui-head, .st-key-sec_" + cur() + " .ui-team-head");
    if (here && (cur() !== 0 || doc.querySelector(".ui-jr"))) {
      win.setTimeout(function () { root.setAttribute("data-ready", ""); }, 180);
      return true;
    }
    return false;
  }

  // ---- keyboard ----
  doc.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
      for (const d of doc.querySelectorAll(".tbx-select[open]")) { d.open = false; d.querySelector("summary").focus(); }
      const sheet = doc.getElementById("ui-sheet");
      if (sheet) sheet.checked = false;
      return;
    }
    if ((e.key === "Enter" || e.key === " ") && e.target.matches && e.target.matches("label[for^='tbx-sec-']")) {
      e.preventDefault(); e.target.click(); return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.closest && e.target.closest("input, textarea, select, summary, [contenteditable]")) return;
    if (e.key === "t" || e.key === "T") { e.preventDefault(); toggleTeam(); return; }
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); stepBy(e.key === "ArrowRight" ? 1 : -1); return; }
    if (/^[1-9]$/.test(e.key) && Number(e.key) <= N()) { e.preventDefault(); go(Number(e.key) - 1); }
  }, true);

  // ---- score journey: replay the run stage by stage ----
  let playing = false, timer = 0;
  function stopPlay() {
    playing = false; win.clearTimeout(timer);
    for (const b of doc.querySelectorAll("[data-play].running")) b.classList.remove("running");
  }
  function startPlay(btn) {
    const name = btn.getAttribute("data-play");
    const inputs = doc.querySelectorAll("input[name='" + name + "']");
    if (!inputs.length) return;
    stopPlay();
    playing = true; btn.classList.add("running");
    let i = 0;
    (function next() {
      if (!playing) return;
      inputs[i].checked = true;
      i += 1;
      if (i === inputs.length) { timer = win.setTimeout(stopPlay, 300); return; }
      timer = win.setTimeout(next, i === 1 ? 700 : 1100);
    })();
  }

  // ---- alert risk simulator ----
  function phi(z) {                         // standard normal CDF (Abramowitz-Stegun 26.2.17)
    const t = 1 / (1 + 0.2316419 * Math.abs(z));
    const d = 0.3989423 * Math.exp(-z * z / 2);
    const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return z > 0 ? 1 - p : p;
  }
  function fmtSigma(x) { return (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(1) + "σ"; }
  function simRender(card, moved) {
    const rates = card.getAttribute("data-rates").split(",").map(Number);
    const avg = Number(card.getAttribute("data-avg"));
    const val = {};
    for (const row of card.querySelectorAll("[data-k]")) {
      const inp = row.querySelector("input");
      const x = Number(inp.value);
      val[row.getAttribute("data-k")] = x;
      inp.style.setProperty("--fill", ((x - inp.min) / (inp.max - inp.min) * 100) + "%");
      row.querySelector(".out").textContent = fmtSigma(x);
    }
    const contrast = val.cin + val.cout - val.bin - val.bout;
    const dec = Math.min(9, Math.floor(phi(contrast / 2) * 10));
    const rate = rates[dec], ratio = rate / avg;
    const big = card.querySelector('[data-o="rate"]');
    big.textContent = (rate * 100).toFixed(1) + "%";
    big.classList.toggle("hot", ratio > 1.15);
    big.classList.toggle("cool", ratio < 0.85);
    card.querySelector('[data-o="dec"]').textContent =
      "decile " + (dec + 1) + " of 10 · " + ratio.toFixed(2) + "× average · contrast " + fmtSigma(contrast);
    card.querySelectorAll(".ui-sim-bars i").forEach(function (b, i) { b.classList.toggle("on", i === dec); });
    const note = card.querySelector('[data-o="note"]');
    const bold = function (t) { const x = doc.createElement("b"); x.textContent = t; return x; };
    const say = function () { note.textContent = ""; for (const part of arguments) note.append(part); };
    if (moved === "scale")
      say("All four levels moved together and the contrast stayed at " + fmtSigma(contrast) +
          ", so the risk did not move. This ", bold("customer scale"), " is 62 % of the variance and almost useless for prediction.");
    else if (moved)
      contrast > 1 ? say("Cash and card are high ", bold("relative to bank transfers"), " → the alert lands in the top deciles.")
        : contrast > -1 ? say("Instruments are balanced → close to the average escalation rate.")
        : say("Bank transfers dominate → one of the least escalated deciles.");
  }
  let simReady = false;
  function initSim() {
    if (simReady) return;
    const card = doc.querySelector("[data-sim]");
    if (!card) return;
    simReady = true;
    let lastScale = 0;
    card.addEventListener("input", function (e) {
      const row = e.target.closest("[data-k]");
      if (!row) return;
      const k = row.getAttribute("data-k");
      if (k === "scale") {                  // the common scale moves every instrument by the same step
        const now = Number(e.target.value), delta = now - lastScale;
        lastScale = now;
        for (const r of card.querySelectorAll("[data-k]:not([data-k='scale']) input"))
          r.value = Math.max(Number(r.min), Math.min(Number(r.max), Number(r.value) + delta)).toFixed(1);
      }
      simRender(card, k);
    });
    simRender(card, null);
  }

  // ---- analyst budget ----
  let budgetReady = false;
  function initBudget() {
    if (budgetReady) return;
    const card = doc.querySelector("[data-budget]");
    if (!card) return;
    const gdata = win.__GAINS_DATA__;
    if (!gdata || !gdata.points) return;
    budgetReady = true;

    const points = gdata.points;
    const totalAlerts = gdata.total_alerts || 14000;
    const totalEsc = gdata.total_esc || 2405;
    const slider = card.querySelector("#budget-range");
    if (!slider) return;

    const W = 600, H = 240, pl = 44, pr = 20, pt = 16, pb = 32;
    const dot = card.querySelector("#bg-dot");
    const gv = card.querySelector("#bg-gv");
    const gh = card.querySelector("#bg-gh");

    function renderBudget(p) {
      const idx = Math.max(0, Math.min(100, Math.round(p)));
      const ptData = points[idx] || points[30];
      const gains = ptData.gains;
      const mult = p > 0 ? (gains / (p / 100)).toFixed(2) : "1.00";

      slider.value = p;
      slider.style.setProperty("--fill", p + "%");

      for (const el of card.querySelectorAll('[data-b="p"]')) el.textContent = p + " %";
      const caughtEl = card.querySelector('[data-b="caught"]');
      if (caughtEl) caughtEl.textContent = (gains * 100).toFixed(1) + " %";
      const multEl = card.querySelector('[data-b="mult"]');
      if (multEl) multEl.textContent = mult + " \u00d7";
      const alertsEl = card.querySelector('[data-b="alerts"]');
      if (alertsEl) alertsEl.textContent = ptData.alerts_checked.toLocaleString("en-US");
      const escEl = card.querySelector('[data-b="esc"]');
      if (escEl) escEl.textContent = ptData.esc_caught.toLocaleString("en-US");
      const savedEl = card.querySelector('[data-b="saved"]');
      if (savedEl) savedEl.textContent = (100 - p) + " %";

      if (dot && gv && gh) {
        const cx = pl + (p / 100) * (W - pl - pr);
        const cy = pt + (1 - gains) * (H - pt - pb);
        dot.setAttribute("cx", cx.toFixed(1));
        dot.setAttribute("cy", cy.toFixed(1));
        gv.setAttribute("x1", cx.toFixed(1));
        gv.setAttribute("x2", cx.toFixed(1));
        gv.setAttribute("y1", cy.toFixed(1));
        gv.setAttribute("y2", (H - pb).toFixed(1));
        gh.setAttribute("x1", pl.toFixed(1));
        gh.setAttribute("x2", cx.toFixed(1));
        gh.setAttribute("y1", cy.toFixed(1));
        gh.setAttribute("y2", cy.toFixed(1));
      }
    }

    slider.addEventListener("input", function (e) {
      renderBudget(Number(e.target.value));
    });
    renderBudget(30);
    win.__renderBudget__ = renderBudget;
  }

  // ---- alert inspector: one real alert at a time, seen through the formula ----
  let inspReady = false;
  function initInspector() {
    if (inspReady) return;
    const card = doc.querySelector("[data-insp]");
    if (!card) return;
    const idata = win.__INSP_DATA__;
    if (!idata || !idata.alerts || !idata.alerts.length) return;
    inspReady = true;

    const alerts = idata.alerts;
    const bins = idata.decile_bins || [];
    const HOT = "#E07A5B", COOL = "#5B8DEF";
    // the four instrument x direction groups of the formula: [type code, out flag, sign, label]
    const TERMS = [[2, 0, 1, "cash in"], [0, 1, 1, "card out"], [1, 0, -1, "bank in"], [1, 1, -1, "bank out"]];
    const $ = function (id) { return card.querySelector("#" + id); };
    const canvas = $("insp-canvas"), range = $("insp-range");
    const levelEls = [$("lvl-cin"), $("lvl-cout"), $("lvl-bin"), $("lvl-bout")];

    let filterVal = "all", cutoff = 0;
    let active = alerts.findIndex(function (a) { return a.y === 1 && a.decile >= 9; });
    if (active === -1) active = 0;
    const visible = function () {
      const out = [];
      for (let i = 0; i < alerts.length; i += 1) if (filterVal === "all" || String(alerts[i].y) === filterVal) out.push(i);
      return out;
    };
    const fmt = function (v) { return v === null ? "—" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2); };

    function quantile(arr, q) {                 // linear interpolation, as pandas does
      if (!arr.length) return null;
      arr.sort(function (u, v) { return u - v; });
      const pos = (arr.length - 1) * q, base = Math.floor(pos), rest = pos - base;
      return base + 1 === arr.length ? arr[base] : arr[base] + rest * (arr[base + 1] - arr[base]);
    }
    function termOf(t) {
      for (let k = 0; k < TERMS.length; k += 1) if (TERMS[k][0] === t[2] && TERMS[k][1] === t[3]) return k;
      return -1;
    }

    // four lanes, one per term of the formula: x = transaction size, a bold tick = the q75 level
    function draw(a, levels) {
      const ctx = canvas.getContext("2d");
      const dpr = win.devicePixelRatio || 1;
      const W = (canvas.parentElement && canvas.parentElement.clientWidth) || 700;
      const small = W > 560 ? 0 : 1;
      const lane = small ? 40 : 44, pt = 8, pb = 22, H = pt + lane * TERMS.length + pb;
      canvas.width = W * dpr; canvas.height = H * dpr; canvas.style.height = H + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const pl = small ? 74 : 96, pr = small ? 50 : 64, cw = W - pl - pr;
      const LO = -3, HI = 4;                    // the same scale for every alert, so cases compare
      const X = function (v) { return pl + (Math.max(LO, Math.min(HI, v)) - LO) / (HI - LO) * cw; };
      const mid = function (k) { return pt + lane * k + lane / 2; };

      ctx.font = "10.5px Inter, sans-serif";
      for (const v of [-2, 0, 2, 4]) {          // x grid
        ctx.strokeStyle = v === 0 ? "#2C3643" : "#1A212B"; ctx.lineWidth = 1;
        ctx.setLineDash(v === 0 ? [3, 3] : []);
        ctx.beginPath(); ctx.moveTo(X(v), pt); ctx.lineTo(X(v), pt + lane * TERMS.length); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#6B7683"; ctx.textAlign = "center";
        ctx.fillText((v > 0 ? "+" : "") + v, X(v), H - 7);
      }
      for (let k = 0; k < TERMS.length; k += 1) {
        const col = TERMS[k][2] > 0 ? HOT : COOL, y = mid(k);
        if (k) { ctx.strokeStyle = "#161C24"; ctx.beginPath(); ctx.moveTo(8, pt + lane * k); ctx.lineTo(W - 8, pt + lane * k); ctx.stroke(); }
        ctx.fillStyle = col; ctx.textAlign = "left"; ctx.font = "600 12px Inter, sans-serif";
        ctx.fillText((TERMS[k][2] > 0 ? "+ " : "− ") + TERMS[k][3], 10, y + 4);
      }
      // the transactions of each term; unused history is almost invisible
      let n = 0;
      for (const t of a.tx) {
        const k = termOf(t);
        if (k === -1) continue;
        n += 1;
        const jit = ((n * 9301 + 49297) % 233280) / 233280 - 0.5;   // stable vertical spread inside the lane
        ctx.globalAlpha = t[0] <= cutoff ? 0.55 : 0.06;
        ctx.fillStyle = TERMS[k][2] > 0 ? HOT : COOL;
        ctx.beginPath(); ctx.arc(X(t[1]), mid(k) + jit * lane * 0.62, small ? 1.9 : 2.2, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
      // q75 of each lane: bold tick + its value on the right
      for (let k = 0; k < TERMS.length; k += 1) {
        const col = TERMS[k][2] > 0 ? HOT : COOL, y = mid(k), v = levels[k];
        ctx.fillStyle = col; ctx.textAlign = "right"; ctx.font = "600 12px 'JetBrains Mono', monospace";
        ctx.fillText(fmt(v), W - 10, y + 4);
        if (v === null) continue;
        ctx.strokeStyle = "#0B0E13"; ctx.lineWidth = 6;
        ctx.beginPath(); ctx.moveTo(X(v), y - lane * 0.38); ctx.lineTo(X(v), y + lane * 0.38); ctx.stroke();
        ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.lineCap = "round";
        ctx.beginPath(); ctx.moveTo(X(v), y - lane * 0.36); ctx.lineTo(X(v), y + lane * 0.36); ctx.stroke();
        ctx.lineCap = "butt";
      }
    }

    function render() {
      const list = visible();
      if (list.indexOf(active) === -1) active = list[0];
      const a = alerts[active];
      const groups = [[], [], [], []];
      for (const t of a.tx) { const k = termOf(t); if (k !== -1 && t[0] <= cutoff) groups[k].push(t[1]); }
      const levels = groups.map(function (g) { return quantile(g, 0.75); });
      draw(a, levels);
      levels.forEach(function (v, k) { if (levelEls[k]) levelEls[k].textContent = fmt(v); });

      const ok = levels.every(function (v) { return v !== null; });
      const ctr = ok ? levels[0] + levels[1] - levels[2] - levels[3] : null;
      let dec = 10;
      if (ok) for (let k = 1; k < bins.length - 1; k += 1) if (ctr < bins[k]) { dec = k; break; }
      $("insp-contrast").textContent = fmt(ctr);
      $("insp-decile").textContent = ok ? "decile " + dec : "—";
      const esc = a.y === 1;
      $("insp-truth").textContent = esc ? "Escalated" : "Dismissed";
      $("insp-truth-card").classList.toggle("is-esc", esc);
      $("insp-truth-card").classList.toggle("is-dism", !esc);
      $("insp-truth-info").textContent = !ok ? "Not enough history yet: the formula needs all four instruments."
        : (esc && dec >= 7) || (!esc && dec <= 4) ? "Matches the rule."
        : "Against the rule — the contrast shifts the odds, it does not decide every alert.";
      $("insp-case").textContent = "Case " + (list.indexOf(active) + 1) + " / " + list.length + " · " + a.id;
      $("insp-cutoff-val").textContent = cutoff === 0 ? "all 180 days" : "first " + (180 + cutoff) + " of 180 days";
      range.style.setProperty("--fill", ((cutoff + 179) / 179 * 100) + "%");
    }

    card.addEventListener("click", function (e) {
      const b = e.target.closest && e.target.closest("[data-insp-step]");
      if (!b) return;
      const list = visible(), i = list.indexOf(active);
      active = list[(i + Number(b.getAttribute("data-insp-step")) + list.length) % list.length];
      render();
    });
    card.addEventListener("change", function (e) {
      if (e.target.name === "insp-filter") { filterVal = e.target.value; render(); }
    });
    range.addEventListener("input", function (e) { cutoff = Number(e.target.value); render(); });
    win.addEventListener("resize", function () { if (card.getBoundingClientRect().height) render(); }, { passive: true });
    render();
    win.__renderAlert__ = render;
  }

  // ---- things that exist only once Streamlit has drawn the page ----
  let booted = false, autoplayed = false;
  function boot() {
    const ready = splash();
    initSim();
    initBudget();
    initInspector();
    if (!booted && doc.querySelector(".ui-dock")) { booted = true; onSection(String(cur()), false); }
    const btn = doc.querySelector("[data-play='jr']");
    if (!autoplayed && btn && cur() === 0) {
      autoplayed = true;
      if (!reduce) win.setTimeout(function () { if (cur() === 0 && !playing) startPlay(btn); }, 1300);
    }
    if (!(ready && simReady && budgetReady && inspReady && booted && (autoplayed || cur() !== 0))) win.setTimeout(boot, 120);
  }
  boot();
})();

