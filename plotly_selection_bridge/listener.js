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
let lastResetToken = null;
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

function caseNameFromPoint(point) {
  try {
    if (point && point.id != null && String(point.id) !== "") return String(point.id);
  } catch (e) {}
  try {
    const cd = point ? point.customdata : null;
    if (Array.isArray(cd) && cd.length > 0) return String(cd[0] || "");
    if (cd && typeof cd === "object") {
      if (cd.case != null && String(cd.case) !== "") return String(cd.case);
      if (cd.id != null && String(cd.id) !== "") return String(cd.id);
    }
    if (cd != null) return String(cd);
  } catch (e) {}
  return "";
}

function rebuildIndexMap(gd) {
  const byTrace = new Map();
  if (!gd || !Array.isArray(gd.data)) return byTrace;
  for (let ti = 0; ti < gd.data.length; ti++) {
    const map = new Map();
    const tr = gd.data[ti];
    const ids = Array.isArray(tr.ids) ? tr.ids : [];
    const cds = Array.isArray(tr.customdata) ? tr.customdata : [];
    const n = Math.max(ids.length, cds.length);
    for (let i = 0; i < n; i++) {
      const c = ids[i] != null && String(ids[i]) !== "" ? String(ids[i]) : caseNameFromCustomData(cds[i]);
      if (!c) continue;
      if (!map.has(c)) map.set(c, []);
      map.get(c).push(i);
    }
    byTrace.set(ti, map);
  }
  return byTrace;
}

function applyVisual(gd) {
  if (!gd || !latestArgs || !Array.isArray(gd.data)) return;
  const win = gd.ownerDocument && gd.ownerDocument.defaultView ? gd.ownerDocument.defaultView : null;
  const Plotly = win && win.Plotly ? win.Plotly : null;
  if (!Plotly || !Plotly.restyle) return;
  const indexByTrace = rebuildIndexMap(gd);
  const traceIndices = [];
  const selectedpoints = [];
  const unselectedOpacity = [];
  const selectedSize = [];
  const selectedOpacity = [];

  for (let ti = 0; ti < gd.data.length; ti++) {
    const tr = gd.data[ti] || {};
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
  if (traceIndices.length) {
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
}

function bindToTarget() {
  if (!latestArgs) return;
  const plots = getPlotDivs();
  if (!Array.isArray(plots) || plots.length === 0) return false;
  const desiredPlotId = String(latestArgs.plot_id || "");
  const idx = Number(latestArgs.plot_index || 0);
  let gd = null;
  if (desiredPlotId === "rx") {
    for (const p of plots) {
      const ui = p && p.layout ? String(p.layout.uirevision || "") : "";
      if (ui.startsWith("rx:")) {
        gd = p;
        break;
      }
    }
    // RX bridge must bind only to RX plot. If not found yet, retry instead of falling back.
    if (!gd) return false;
  }
  if (!gd) {
    if (idx < 0 || idx >= plots.length) return false;
    gd = plots[idx];
  }
  if (!gd || !gd.on) return false;
  const bindKey = `${String(latestArgs.data_id || "")}|${String(latestArgs.chart_id || "")}|${String(desiredPlotId || idx)}`;
  // Fast path: already bound for this render signature.
  try {
    if (gd.__selBridgeKey === bindKey && gd.__selBridgeClick) {
      applyVisual(gd);
      return true;
    }
  } catch (e) {}
  try {
    if (gd.__selBridgeClick && gd.removeListener) gd.removeListener("plotly_click", gd.__selBridgeClick);
    if (gd.__selBridgeAnimated && gd.removeListener) gd.removeListener("plotly_animated", gd.__selBridgeAnimated);
    if (gd.__selBridgeSlider && gd.removeListener) gd.removeListener("plotly_sliderchange", gd.__selBridgeSlider);
  } catch (e) {}

  const toggleCasesFromPoints = (points) => {
    try {
      if (!Array.isArray(points) || points.length === 0) return;
      const seen = new Set();
      for (const p of points) {
        const c = caseNameFromPoint(p);
        if (!c || seen.has(c)) continue;
        seen.add(c);
        if (selectedCases.has(c)) selectedCases.delete(c);
        else selectedCases.add(c);
      }
      if (!seen.size) return;
      persistSelection(String(latestArgs.data_id || ""), String(latestArgs.chart_id || ""));
      applyVisual(gd);
    } catch (e) {}
  };
  const clickHandler = (evt) => {
    if (!evt || !Array.isArray(evt.points) || evt.points.length === 0) return;
    toggleCasesFromPoints([evt.points[0]]);
  };
  const refreshHandler = () => {
    applyVisual(gd);
  };
  try {
    gd.__selBridgeKey = bindKey;
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
    // On first mount/rebind for a chart, adopt current clear token without clearing selection.
    lastClearToken = clearToken;
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
