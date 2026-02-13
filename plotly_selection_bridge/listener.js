function sendToStreamlit(type, payload) {
  try {
    window.parent.postMessage(
      {
        isStreamlitMessage: true,
        type,
        ...(payload || {}),
      },
      "*",
    );
  } catch (e) {}
}

let latestArgs = null;
let lastDataId = null;
let lastChartId = null;
let lastCommitToken = -1;
let lastClearToken = -1;
let lastResetToken = -1;
let selectedCases = new Set();
let bindLoopId = 0;

function storageKey(dataId, chartId) {
  return `fsSweepSelBridge:${String(dataId)}:${String(chartId)}`;
}

function commitSentKey(dataId, chartId) {
  return `fsSweepSelBridgeCommit:${String(dataId)}:${String(chartId)}`;
}

function safeGet(key) {
  try {
    return window.localStorage ? window.localStorage.getItem(key) : null;
  } catch (e) {
    return null;
  }
}

function safeSet(key, value) {
  try {
    if (!window.localStorage) return;
    window.localStorage.setItem(key, value);
  } catch (e) {}
}

function loadSelection(dataId, chartId) {
  const raw = safeGet(storageKey(dataId, chartId));
  if (!raw) return new Set();
  try {
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return new Set();
    return new Set(arr.map((v) => String(v)));
  } catch (e) {
    return new Set();
  }
}

function persistSelection(dataId, chartId) {
  safeSet(storageKey(dataId, chartId), JSON.stringify(Array.from(selectedCases).sort()));
}

function loadLastSentCommit(dataId, chartId) {
  const raw = safeGet(commitSentKey(dataId, chartId));
  const n = Number(raw);
  return Number.isFinite(n) ? n : -1;
}

function persistLastSentCommit(dataId, chartId, token) {
  safeSet(commitSentKey(dataId, chartId), String(Number(token)));
}

function getPlotDivs() {
  const out = [];
  try {
    const doc = (window.parent || window).document;
    const blocks = doc.querySelectorAll('div[data-testid="stPlotlyChart"], div.stPlotlyChart');
    for (const block of blocks) {
      const fr = block.querySelector("iframe");
      if (fr && fr.contentWindow && fr.contentWindow.document) {
        const gd = fr.contentWindow.document.querySelector("div.js-plotly-plot");
        if (gd) {
          out.push(gd);
          continue;
        }
      }
      const gd2 = block.querySelector("div.js-plotly-plot");
      if (gd2) out.push(gd2);
    }
  } catch (e) {}
  return out;
}

function caseNameFromCustomData(cd) {
  if (Array.isArray(cd) && cd.length > 0) return String(cd[0] || "");
  if (cd == null) return "";
  return String(cd);
}

function rebuildIndexMap(gd) {
  const byTrace = new Map();
  if (!gd || !Array.isArray(gd.data)) return byTrace;
  for (let ti = 0; ti < gd.data.length; ti++) {
    const map = new Map();
    const tr = gd.data[ti];
    const cds = Array.isArray(tr.customdata) ? tr.customdata : [];
    for (let i = 0; i < cds.length; i++) {
      const c = caseNameFromCustomData(cds[i]);
      if (!c) continue;
      if (!map.has(c)) map.set(c, []);
      map.get(c).push(i);
    }
    byTrace.set(ti, map);
  }
  return byTrace;
}

function ensureIndexMap(gd) {
  if (!gd) return new Map();
  if (!gd.__selBridgeIndexMap) {
    gd.__selBridgeIndexMap = rebuildIndexMap(gd);
  }
  return gd.__selBridgeIndexMap || new Map();
}

function applyVisual(gd) {
  if (!gd || !latestArgs || !Array.isArray(gd.data)) return;
  const win = gd.ownerDocument && gd.ownerDocument.defaultView ? gd.ownerDocument.defaultView : null;
  const Plotly = win && win.Plotly ? win.Plotly : null;
  if (!Plotly || !Plotly.restyle) return;
  const indexByTrace = ensureIndexMap(gd);
  const traceIndices = [];
  const selectedpoints = [];
  const unselectedOpacity = [];
  const selectedSize = [];
  const selectedOpacity = [];
  for (let ti = 0; ti < gd.data.length; ti++) {
    const map = indexByTrace.get(ti);
    if (!map || map.size === 0) continue;
    const idxs = [];
    for (const c of selectedCases) {
      const part = map.get(c);
      if (part && part.length) idxs.push(...part);
    }
    traceIndices.push(ti);
    selectedpoints.push(idxs.length ? idxs : null);
    unselectedOpacity.push(idxs.length ? Number(latestArgs.unselected_marker_opacity || 0.3) : 1.0);
    selectedSize.push(Number(latestArgs.selected_marker_size || 10));
    selectedOpacity.push(1.0);
  }
  if (!traceIndices.length) return;
  try {
    Plotly.restyle(
      gd,
      {
        selectedpoints,
        "selected.marker.opacity": selectedOpacity,
        "selected.marker.size": selectedSize,
        "unselected.marker.opacity": unselectedOpacity,
      },
      traceIndices,
    );
  } catch (e) {}
}

function bindToTarget() {
  if (!latestArgs) return;
  const plots = getPlotDivs();
  const idx = Number(latestArgs.plot_index || 0);
  if (!Array.isArray(plots) || idx < 0 || idx >= plots.length) return false;
  const gd = plots[idx];
  if (!gd || !gd.on) return false;
  const bindKey = `${String(latestArgs.data_id || "")}|${String(latestArgs.chart_id || "")}|${idx}`;
  try {
    if (gd.__selBridgeKey === bindKey) {
      applyVisual(gd);
      return true;
    }
  } catch (e) {}
  try {
    if (gd.__selBridgeClick && gd.removeListener) gd.removeListener("plotly_click", gd.__selBridgeClick);
    if (gd.__selBridgeAnimated && gd.removeListener) gd.removeListener("plotly_animated", gd.__selBridgeAnimated);
    if (gd.__selBridgeSlider && gd.removeListener) gd.removeListener("plotly_sliderchange", gd.__selBridgeSlider);
  } catch (e) {}

  const clickHandler = (evt) => {
    try {
      if (!evt || !Array.isArray(evt.points) || evt.points.length === 0) return;
      const c = caseNameFromCustomData(evt.points[0].customdata);
      if (!c) return;
      if (selectedCases.has(c)) selectedCases.delete(c);
      else selectedCases.add(c);
      persistSelection(String(latestArgs.data_id || ""), String(latestArgs.chart_id || ""));
      applyVisual(gd);
    } catch (e) {}
  };
  const refreshHandler = () => applyVisual(gd);
  try {
    gd.__selBridgeKey = bindKey;
    gd.__selBridgeIndexMap = rebuildIndexMap(gd);
    gd.__selBridgeClick = clickHandler;
    gd.__selBridgeAnimated = refreshHandler;
    gd.__selBridgeSlider = refreshHandler;
  } catch (e) {}
  gd.on("plotly_click", clickHandler);
  gd.on("plotly_animated", refreshHandler);
  gd.on("plotly_sliderchange", refreshHandler);
  applyVisual(gd);
  return true;
}

function pruneSelectionByAllowed() {
  if (!latestArgs) return;
  const allowed = Array.isArray(latestArgs.allowed_cases) ? latestArgs.allowed_cases.map((v) => String(v)) : [];
  if (!allowed.length) return;
  const allowedSet = new Set(allowed);
  let changed = false;
  for (const c of Array.from(selectedCases)) {
    if (!allowedSet.has(c)) {
      selectedCases.delete(c);
      changed = true;
    }
  }
  if (changed) persistSelection(String(latestArgs.data_id || ""), String(latestArgs.chart_id || ""));
}

function rebindLoop() {
  const myLoopId = bindLoopId;
  let tries = 0;
  (function tick() {
    if (myLoopId !== bindLoopId) return;
    const ok = bindToTarget();
    if (ok) return;
    tries += 1;
    if (tries < 20) setTimeout(tick, 80);
  })();
}

window.addEventListener("message", (event) => {
  const data = event && event.data ? event.data : null;
  if (!data || data.type !== "streamlit:render") return;
  latestArgs = data.args || {};

  const dataId = String(latestArgs.data_id || "");
  const chartId = String(latestArgs.chart_id || "");
  const commitToken = Number(latestArgs.commit_token || 0);
  const commitApplied = Number(latestArgs.commit_applied || 0);
  const clearToken = Number(latestArgs.clear_token || 0);
  const resetToken = Number(latestArgs.reset_token || 0);

  if (dataId !== lastDataId || chartId !== lastChartId) {
    selectedCases = loadSelection(dataId, chartId);
    lastDataId = dataId;
    lastChartId = chartId;
    lastCommitToken = loadLastSentCommit(dataId, chartId);
    lastClearToken = -1;
    lastResetToken = -1;
    bindLoopId += 1;
  }

  if (clearToken !== lastClearToken) {
    lastClearToken = clearToken;
    selectedCases = new Set();
    persistSelection(dataId, chartId);
    bindLoopId += 1;
  }

  if (resetToken !== lastResetToken) {
    lastResetToken = resetToken;
    selectedCases = new Set();
    persistSelection(dataId, chartId);
    bindLoopId += 1;
  }

  pruneSelectionByAllowed();
  const bound = bindToTarget();
  if (!bound) rebindLoop();

  if (commitToken > commitApplied && commitToken > 0 && commitToken !== lastCommitToken) {
    lastCommitToken = commitToken;
    persistLastSentCommit(dataId, chartId, commitToken);
    sendToStreamlit("streamlit:setComponentValue", {
      value: {
        commit_token: commitToken,
        selected_cases: Array.from(selectedCases).sort(),
        nonce: commitToken,
      },
    });
  }
});

sendToStreamlit("streamlit:componentReady", { apiVersion: 1 });
sendToStreamlit("streamlit:setFrameHeight", { height: 0 });
