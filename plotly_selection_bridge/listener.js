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
let applyPassNonce = 0;
let applyQueued = false;
let lastRenderNonceSeen = -1;
let lastSentFrameHeight = 0;

function rootWindow() {
  return window.parent || window;
}

function getStateStore() {
  const root = rootWindow();
  if (!root.__fsCaseUiStore || typeof root.__fsCaseUiStore !== "object") {
    root.__fsCaseUiStore = {};
  }
  return root.__fsCaseUiStore;
}

function pruneStoreByDataId(store, dataId) {
  if (!store || typeof store !== "object") return;
  const did = String(dataId || "");
  if (!did) return;
  for (const k of Object.keys(store)) {
    if (!k.startsWith(`${did}|`)) {
      delete store[k];
    }
  }
}

function asArray(v) {
  return Array.isArray(v) ? v : [];
}

function escapeHtml(txt) {
  return String(txt == null ? "" : txt)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function displayValue(v) {
  return String(v) === "" ? "<empty>" : String(v);
}

function normalizeCaseMeta(rawItems) {
  const out = [];
  for (const item of asArray(rawItems)) {
    if (!item || typeof item !== "object") continue;
    const caseId = String(item.case_id || "");
    if (!caseId) continue;
    const displayCase = String(item.display_case || caseId);
    const parts = asArray(item.parts).map((v) => String(v || ""));
    out.push({
      case_id: caseId,
      display_case: displayCase,
      parts,
    });
  }
  return out;
}

function buildPartOptions(casesMeta, partCount) {
  const out = [];
  for (let i = 0; i < partCount; i++) {
    const vals = new Set();
    for (const item of casesMeta) {
      vals.add(String((item.parts && item.parts[i]) || ""));
    }
    out.push(Array.from(vals).sort((a, b) => a.localeCompare(b)));
  }
  return out;
}

function bumpStateVersion(state) {
  const cur = Number(state && state.__applyVersion != null ? state.__applyVersion : 0);
  const next = Number.isFinite(cur) ? cur + 1 : 1;
  if (state && typeof state === "object") {
    state.__applyVersion = next;
  }
  return next;
}

function createDefaultState(partOptions, colorByOptions, colorByDefault, showOnlyDefault, baseFreqOptions, baseFreqDefault) {
  const selectedByPart = partOptions.map((opts) => new Set(opts));
  const validDefaultColor = colorByOptions.includes(String(colorByDefault || ""))
    ? String(colorByDefault)
    : String(colorByOptions[0] || "Auto");
  const baseOptions = Array.isArray(baseFreqOptions) && baseFreqOptions.length ? baseFreqOptions : [50, 60];
  const baseDefault = Number(baseFreqDefault || 50);
  const validBase = baseOptions.includes(baseDefault) ? baseDefault : Number(baseOptions[0] || 50);
  return {
    selectedByPart,
    selectedCases: new Set(),
    colorBy: validDefaultColor,
    baseFrequencyHz: validBase,
    showOnlySelected: Boolean(showOnlyDefault),
    showHarmonics: Boolean(latestArgs && latestArgs.show_harmonics_default),
    binWidthHz: Number(latestArgs && latestArgs.bin_width_hz_default ? latestArgs.bin_width_hz_default : 0),
    importStatus: "",
    __applyVersion: 1,
    __version: 1,
    __resetToken: 0,
  };
}

function sanitizeState(state, casesMeta, partOptions, colorByOptions, colorByDefault, baseFreqOptions, baseFreqDefault) {
  const knownCaseIds = new Set(casesMeta.map((x) => String(x.case_id)));
  const next = state && typeof state === "object" ? state : {};

  const rawSelected = next.selectedCases;
  const selectedCases = new Set();
  const selectedSource =
    rawSelected instanceof Set ? Array.from(rawSelected) : Array.isArray(rawSelected) ? rawSelected : [];
  for (const cid of selectedSource) {
    const c = String(cid || "");
    if (c && knownCaseIds.has(c)) selectedCases.add(c);
  }
  next.selectedCases = selectedCases;

  const selectedByPart = [];
  const rawByPart = Array.isArray(next.selectedByPart) ? next.selectedByPart : [];
  for (let i = 0; i < partOptions.length; i++) {
    const opts = partOptions[i];
    const allowed = new Set(opts);
    const rawPart = rawByPart[i];
    const rawVals =
      rawPart instanceof Set ? Array.from(rawPart) : Array.isArray(rawPart) ? rawPart.map((v) => String(v || "")) : opts;
    const setPart = new Set();
    for (const v of rawVals) {
      const s = String(v || "");
      if (allowed.has(s)) setPart.add(s);
    }
    selectedByPart.push(setPart);
  }
  next.selectedByPart = selectedByPart;

  const fallbackColor = colorByOptions.includes(String(colorByDefault || ""))
    ? String(colorByDefault)
    : String(colorByOptions[0] || "Auto");
  next.colorBy = colorByOptions.includes(String(next.colorBy || "")) ? String(next.colorBy) : fallbackColor;
  const baseOptions = Array.isArray(baseFreqOptions) && baseFreqOptions.length ? baseFreqOptions : [50, 60];
  const baseDefault = Number(baseFreqDefault || 50);
  const rawBase = Number(next.baseFrequencyHz != null ? next.baseFrequencyHz : baseDefault);
  next.baseFrequencyHz = baseOptions.includes(rawBase) ? rawBase : (baseOptions.includes(baseDefault) ? baseDefault : Number(baseOptions[0] || 50));
  next.showOnlySelected = Boolean(next.showOnlySelected);
  next.showHarmonics = Boolean(next.showHarmonics);
  const bw = Number(next.binWidthHz || 0);
  next.binWidthHz = Number.isFinite(bw) && bw >= 0 ? bw : 0;
  const ver = Number(next.__applyVersion != null ? next.__applyVersion : 1);
  next.__applyVersion = Number.isFinite(ver) && ver >= 1 ? Math.floor(ver) : 1;
  next.importStatus = String(next.importStatus || "");
  next.__version = 1;
  return next;
}

function ensureContext() {
  if (!latestArgs) return null;
  const dataId = String(latestArgs.data_id || "");
  const chartId = String(latestArgs.chart_id || "");
  const resetToken = Number(latestArgs.reset_token || 0);
  const casesMeta = normalizeCaseMeta(latestArgs.cases_meta);
  const partLabels = asArray(latestArgs.part_labels).map((v) => String(v || ""));
  const partCount = partLabels.length;
  const partOptions = buildPartOptions(casesMeta, partCount);
  const colorByOptions = asArray(latestArgs.color_by_options).map((v) => String(v || "")).filter((v) => v !== "");
  const colorOptions = colorByOptions.length ? colorByOptions : ["Auto"];
  const colorByDefault = String(latestArgs.color_by_default || "Auto");
  const showOnlyDefault = Boolean(latestArgs.show_only_default);
  const baseFreqOptionsRaw = asArray(latestArgs.base_frequency_options).map((v) => Number(v)).filter((v) => Number.isFinite(v) && v > 0);
  const baseFreqOptions = baseFreqOptionsRaw.length ? baseFreqOptionsRaw : [50, 60];
  const baseFreqDefaultRaw = Number(latestArgs.base_frequency_default || 50);
  const baseFreqDefault = baseFreqOptions.includes(baseFreqDefaultRaw) ? baseFreqDefaultRaw : Number(baseFreqOptions[0] || 50);
  const selectionResetToken = Number(latestArgs.selection_reset_token || 0);
  const stateKey = `${dataId}|${chartId}`;
  const store = getStateStore();
  pruneStoreByDataId(store, dataId);
  const prev = store[stateKey];
  let state;
  if (!prev || Number(prev.__resetToken || 0) !== resetToken || Number(prev.__version || 0) !== 1) {
    state = createDefaultState(partOptions, colorOptions, colorByDefault, showOnlyDefault, baseFreqOptions, baseFreqDefault);
  } else {
    state = sanitizeState(prev, casesMeta, partOptions, colorOptions, colorByDefault, baseFreqOptions, baseFreqDefault);
  }

  const prevSelReset = Number(state && state.__selectionResetToken != null ? state.__selectionResetToken : 0);
  if (prevSelReset !== selectionResetToken) {
    state.selectedCases = new Set();
    state.importStatus = "";
    bumpStateVersion(state);
  }
  state.__selectionResetToken = selectionResetToken;
  state.__resetToken = resetToken;
  store[stateKey] = state;

  const caseById = new Map();
  const exactLookup = new Map();
  const displayLookup = new Map();
  for (const item of casesMeta) {
    const cid = String(item.case_id);
    const disp = String(item.display_case || cid);
    caseById.set(cid, item);
    exactLookup.set(cid.toLowerCase(), cid);
    if (!displayLookup.has(disp.toLowerCase())) displayLookup.set(disp.toLowerCase(), []);
    displayLookup.get(disp.toLowerCase()).push(cid);
  }

  return {
    dataId,
    chartId,
    state,
    casesMeta,
    partLabels,
    partOptions,
    baseFreqOptions,
    colorOptions,
    caseById,
    exactLookup,
    displayLookup,
  };
}

function getColorMap(ctx) {
  const all = latestArgs && latestArgs.color_maps && typeof latestArgs.color_maps === "object" ? latestArgs.color_maps : {};
  const key = String(ctx.state.colorBy || "Auto");
  const m = all[key];
  return m && typeof m === "object" ? m : {};
}

function colorForCase(ctx, caseId) {
  const cmap = getColorMap(ctx);
  const c = cmap ? cmap[String(caseId)] : null;
  return c ? String(c) : "#1f77b4";
}

function parseHexColor(raw) {
  const s = String(raw || "").trim();
  const m6 = s.match(/^#([0-9a-fA-F]{6})$/);
  if (m6) {
    const v = m6[1];
    return {
      r: parseInt(v.slice(0, 2), 16),
      g: parseInt(v.slice(2, 4), 16),
      b: parseInt(v.slice(4, 6), 16),
    };
  }
  const m3 = s.match(/^#([0-9a-fA-F]{3})$/);
  if (m3) {
    const v = m3[1];
    return {
      r: parseInt(v[0] + v[0], 16),
      g: parseInt(v[1] + v[1], 16),
      b: parseInt(v[2] + v[2], 16),
    };
  }
  return null;
}

function rgbToHexColor(r, g, b) {
  const clamp = (x) => Math.max(0, Math.min(255, Math.round(Number(x || 0))));
  const toHex = (x) => clamp(x).toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function buildActivePartValueColorMap(ctx) {
  let activeLabel = String(ctx && ctx.state ? ctx.state.colorBy || "" : "");
  if (activeLabel === "Auto") {
    activeLabel = String(latestArgs && latestArgs.auto_color_part_label ? latestArgs.auto_color_part_label : "");
  }
  const partIdx =
    ctx && Array.isArray(ctx.partLabels) ? ctx.partLabels.findIndex((lbl) => String(lbl || "") === activeLabel) : -1;
  if (partIdx < 0) return { partIdx: -1, valueColorMap: new Map() };

  const cmap = getColorMap(ctx);
  const sums = new Map();
  for (const item of asArray(ctx && ctx.casesMeta)) {
    const caseId = String(item && item.case_id ? item.case_id : "");
    if (!caseId) continue;
    const partVal = String((item && item.parts && item.parts[partIdx]) || "");
    const rgb = parseHexColor(cmap && cmap[caseId] ? cmap[caseId] : null);
    if (!rgb) continue;
    const prev = sums.get(partVal) || { r: 0, g: 0, b: 0, n: 0 };
    prev.r += Number(rgb.r || 0);
    prev.g += Number(rgb.g || 0);
    prev.b += Number(rgb.b || 0);
    prev.n += 1;
    sums.set(partVal, prev);
  }

  const valueColorMap = new Map();
  for (const [partVal, agg] of sums.entries()) {
    const n = Number(agg && agg.n ? agg.n : 0);
    if (!(n > 0)) continue;
    valueColorMap.set(
      String(partVal || ""),
      rgbToHexColor(Number(agg.r || 0) / n, Number(agg.g || 0) / n, Number(agg.b || 0) / n),
    );
  }
  return { partIdx, valueColorMap };
}

function computeAllowedSet(ctx) {
  const allowed = new Set();
  for (const item of ctx.casesMeta) {
    let ok = true;
    for (let i = 0; i < ctx.partOptions.length; i++) {
      const sel = ctx.state.selectedByPart[i];
      if (!(sel instanceof Set) || sel.size === 0) {
        ok = false;
        break;
      }
      const partVal = String((item.parts && item.parts[i]) || "");
      if (!sel.has(partVal)) {
        ok = false;
        break;
      }
    }
    if (ok) allowed.add(String(item.case_id));
  }
  return allowed;
}

function caseIdFromCustomData(cd) {
  if (Array.isArray(cd)) {
    if (cd.length >= 1 && cd[0] != null && String(cd[0]) !== "") return String(cd[0]);
    if (cd.length >= 2 && cd[1] != null && String(cd[1]) !== "") return String(cd[1]);
    return "";
  }
  if (cd == null) return "";
  return String(cd);
}

function getPlotDivs() {
  const out = [];
  try {
    const doc = rootWindow().document;
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

function buildHarmonicShapes(ctx, nMin, nMax, fBase) {
  if (!ctx) return [];
  const showMarkers = Boolean(ctx.state.showHarmonics);
  const binWidthHz = Number(ctx.state.binWidthHz || 0);
  if (!showMarkers && !(Number.isFinite(binWidthHz) && binWidthHz > 0)) return [];
  if (!Number.isFinite(nMin) || !Number.isFinite(nMax) || nMin >= nMax) return [];
  if (!Number.isFinite(fBase) || fBase <= 0) return [];
  const out = [];
  const kStart = Math.max(1, Math.floor(nMin));
  const kEnd = Math.ceil(nMax);
  for (let k = kStart; k <= kEnd; k++) {
    if (showMarkers) {
      out.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: k,
        x1: k,
        y0: 0,
        y1: 1,
        line: { color: "rgba(0,0,0,0.3)", width: 1.5 },
      });
    }
    if (Number.isFinite(binWidthHz) && binWidthHz > 0) {
      const dn = Number(binWidthHz) / (2.0 * Number(fBase));
      const edges = [k - dn, k + dn];
      for (const edge of edges) {
        out.push({
          type: "line",
          xref: "x",
          yref: "paper",
          x0: edge,
          x1: edge,
          y0: 0,
          y1: 1,
          line: { color: "rgba(0,0,0,0.2)", width: 1, dash: "dot" },
        });
      }
    }
  }
  return out;
}

function isVisibleForCase(caseId, allowedSet, selectedSet, showOnlySelected) {
  if (!allowedSet.has(caseId)) return false;
  if (!showOnlySelected) return true;
  // Preserve toggle state; empty selection means "no selection filter applied".
  if (selectedSet.size === 0) return true;
  return selectedSet.has(caseId);
}

function applyLineBaseFrequency(ctx, gd) {
  if (!gd || !Array.isArray(gd.data)) return;
  const win = gd.ownerDocument && gd.ownerDocument.defaultView ? gd.ownerDocument.defaultView : null;
  const Plotly = win && win.Plotly ? win.Plotly : null;
  if (!Plotly || !Plotly.restyle || !Plotly.relayout) return;

  const baseHz = Number(ctx && ctx.state && ctx.state.baseFrequencyHz != null ? ctx.state.baseFrequencyHz : 50);
  if (!Number.isFinite(baseHz) || baseHz <= 0) return;

  try {
    const sameBase = Number(gd.__fsBaseFreqHz || 0) === baseHz;
    const sameDataRef = gd.__fsBaseFreqDataRef === gd.data;
    const sameCount = Number(gd.__fsBaseFreqTraceCount || 0) === gd.data.length;
    if (sameBase && sameDataRef && sameCount) return;
  } catch (e) {}

  const traceIndices = [];
  const xArrays = [];
  let minHz = Number.POSITIVE_INFINITY;
  let maxHz = Number.NEGATIVE_INFINITY;

  for (let ti = 0; ti < gd.data.length; ti++) {
    const tr = gd.data[ti];
    const meta = tr && typeof tr.meta === "object" ? tr.meta : null;
    const kind = meta && meta.kind != null ? String(meta.kind) : "";
    if (kind !== "line") continue;
    const cds = Array.isArray(tr.customdata) ? tr.customdata : [];
    const xVals = [];
    for (let i = 0; i < cds.length; i++) {
      const raw = Array.isArray(cds[i]) ? cds[i][0] : cds[i];
      const hz = Number(raw);
      if (!Number.isFinite(hz)) {
        xVals.push(null);
        continue;
      }
      minHz = Math.min(minHz, hz);
      maxHz = Math.max(maxHz, hz);
      xVals.push(hz / baseHz);
    }
    traceIndices.push(ti);
    xArrays.push(xVals);
  }

  if (!traceIndices.length) return;
  try {
    Plotly.restyle(gd, { x: xArrays }, traceIndices);
    if (Number.isFinite(minHz) && Number.isFinite(maxHz) && maxHz > minHz) {
      Plotly.relayout(gd, { "xaxis.range": [minHz / baseHz, maxHz / baseHz] });
    }
    gd.__fsBaseFreqHz = baseHz;
    gd.__fsBaseFreqDataRef = gd.data;
    gd.__fsBaseFreqTraceCount = gd.data.length;
  } catch (e) {}
}

function applyLineStyles(ctx, gd, allowedSet) {
  if (!gd || !Array.isArray(gd.data)) return;
  const win = gd.ownerDocument && gd.ownerDocument.defaultView ? gd.ownerDocument.defaultView : null;
  const Plotly = win && win.Plotly ? win.Plotly : null;
  if (!Plotly || !Plotly.restyle) return;
  const applyToken = Number(ctx.state && ctx.state.__applyVersion != null ? ctx.state.__applyVersion : 0);
  try {
    const sameToken = Number(gd.__fsLineApplyToken || 0) === applyToken;
    const sameCount = Number(gd.__fsLineTraceCount || 0) === gd.data.length;
    const sameDataRef = gd.__fsLineDataRef === gd.data;
    if (sameToken && sameCount && sameDataRef) return;
  } catch (e) {}

  const hasSelection = ctx.state.selectedCases.size > 0;
  const showOnly = Boolean(ctx.state.showOnlySelected);
  const dimLineWidth = Number(latestArgs.dim_line_width || 1.0);
  const dimLineOpacity = Number(latestArgs.dim_line_opacity || 0.35);
  const dimLineColor = String(latestArgs.dim_line_color || "#B8B8B8");
  const selectedLineWidth = Number(latestArgs.selected_line_width || 2.5);

  if (!gd.__fsCaseUiBase) gd.__fsCaseUiBase = {};
  const baseMap = gd.__fsCaseUiBase;

  const traceIndices = [];
  const visibleVals = [];
  const legendVals = [];
  const lineColors = [];
  const lineWidths = [];
  const opacities = [];

  for (let ti = 0; ti < gd.data.length; ti++) {
    const tr = gd.data[ti];
    const meta = tr && typeof tr.meta === "object" ? tr.meta : null;
    const caseId = meta && meta.case_id != null ? String(meta.case_id) : "";
    if (!caseId || !ctx.caseById.has(caseId)) continue;

    if (!baseMap[ti]) {
      const baseLineWidth = tr && tr.line && tr.line.width != null ? Number(tr.line.width) : 2.0;
      const baseOpacity = tr && tr.opacity != null ? Number(tr.opacity) : 1.0;
      baseMap[ti] = {
        lineWidth: baseLineWidth,
        opacity: baseOpacity,
      };
    }
    const base = baseMap[ti];
    const isSelected = ctx.state.selectedCases.has(caseId);
    const visible = isVisibleForCase(caseId, allowedSet, ctx.state.selectedCases, showOnly);
    const legendVisible = visible && (!hasSelection || isSelected);
    const shouldDim = visible && hasSelection && !showOnly && !isSelected;
    const shouldHighlight = visible && hasSelection && isSelected;

    let finalColor = colorForCase(ctx, caseId);
    let finalWidth = Number(base.lineWidth || 2.0);
    let finalOpacity = 1.0;
    if (shouldDim) {
      finalColor = dimLineColor;
      finalWidth = dimLineWidth;
      finalOpacity = dimLineOpacity;
    } else if (shouldHighlight) {
      finalWidth = selectedLineWidth;
      finalOpacity = 1.0;
    } else if (!hasSelection) {
      finalOpacity = Number(base.opacity || 1.0);
    }

    traceIndices.push(ti);
    visibleVals.push(Boolean(visible));
    legendVals.push(Boolean(legendVisible));
    lineColors.push(String(finalColor));
    lineWidths.push(Number(finalWidth));
    opacities.push(Number(finalOpacity));
  }

  if (!traceIndices.length) return;
  try {
    Plotly.restyle(
      gd,
      {
        visible: visibleVals,
        showlegend: legendVals,
        "line.color": lineColors,
        "line.width": lineWidths,
        opacity: opacities,
      },
      traceIndices,
    );
    gd.__fsLineApplyToken = applyToken;
    gd.__fsLineTraceCount = gd.data.length;
    gd.__fsLineDataRef = gd.data;
  } catch (e) {}
}

function applyHarmonicsToLinePlot(ctx, gd) {
  if (!gd) return;
  const win = gd.ownerDocument && gd.ownerDocument.defaultView ? gd.ownerDocument.defaultView : null;
  const Plotly = win && win.Plotly ? win.Plotly : null;
  if (!Plotly || !Plotly.relayout) return;
  const fBase = Number(ctx && ctx.state && ctx.state.baseFrequencyHz != null ? ctx.state.baseFrequencyHz : (latestArgs && latestArgs.f_base != null ? latestArgs.f_base : 50));
  let n0 = Number(latestArgs && latestArgs.n_min != null ? latestArgs.n_min : 0);
  let n1 = Number(latestArgs && latestArgs.n_max != null ? latestArgs.n_max : 1);
  try {
    const xr = gd && gd.layout && gd.layout.xaxis && Array.isArray(gd.layout.xaxis.range) ? gd.layout.xaxis.range : null;
    if (xr && xr.length === 2) {
      const a = Number(xr[0]);
      const b = Number(xr[1]);
      if (Number.isFinite(a) && Number.isFinite(b)) {
        n0 = Math.min(a, b);
        n1 = Math.max(a, b);
      }
    }
  } catch (e) {}

  const harmonicShapes = buildHarmonicShapes(ctx, n0, n1, fBase);
  const hKey = JSON.stringify({
    show: Boolean(ctx.state.showHarmonics),
    bw: Number(ctx.state.binWidthHz || 0),
    n0: Number(n0),
    n1: Number(n1),
    fb: Number(fBase),
  });
  try {
    if (gd.__fsHarmKey === hKey) return;
    gd.__fsHarmKey = hKey;
  } catch (e) {}
  try {
    Plotly.relayout(gd, { shapes: harmonicShapes });
  } catch (e) {}
}

function updateRxStatusLabel(filteredCount, freqSteps) {
  const domId = String(latestArgs && latestArgs.rx_status_dom_id ? latestArgs.rx_status_dom_id : "");
  if (!domId) return;
  try {
    const doc = rootWindow().document;
    const el = doc ? doc.getElementById(domId) : null;
    if (!el) return;
    el.textContent = `R vs X points shown: ${Number(filteredCount || 0)} | Frequency steps: ${Number(freqSteps || 0)}`;
  } catch (e) {}
}

function buildScatterMarkerArrays(cds, ctx, allowedSet, selectedMarkerSize, dimMarkerOpacity, dimMarkerSize) {
  const hasSelection = ctx.state.selectedCases.size > 0;
  const colors = [];
  const opacities = [];
  const sizes = [];
  let filteredCount = 0;

  for (let i = 0; i < cds.length; i++) {
    const caseId = caseIdFromCustomData(cds[i]);
    const passesFilter = allowedSet.has(caseId);
    if (passesFilter) filteredCount += 1;
    const selected = ctx.state.selectedCases.has(caseId);
    // Scatter should always keep allowed points visible for interactive picking.
    // "Show only selected sweeps" is applied to line plots only.
    const visible = Boolean(passesFilter);
    const shouldDim = visible && hasSelection && !selected;
    const c = colorForCase(ctx, caseId);

    if (!visible) {
      colors.push(c);
      opacities.push(0.0);
      sizes.push(0.0);
      continue;
    }

    colors.push(c);
    if (shouldDim) {
      opacities.push(dimMarkerOpacity);
      sizes.push(dimMarkerSize);
    } else {
      opacities.push(1.0);
      sizes.push(selectedMarkerSize);
    }
  }
  return { colors, opacities, sizes, filteredCount };
}

function applyScatterStyles(ctx, gd, allowedSet, forceApply) {
  if (!gd || !Array.isArray(gd.data)) return;
  const win = gd.ownerDocument && gd.ownerDocument.defaultView ? gd.ownerDocument.defaultView : null;
  const Plotly = win && win.Plotly ? win.Plotly : null;
  if (!Plotly || !Plotly.restyle) return;

  let pointTraceIndex = -1;
  for (let i = 0; i < gd.data.length; i++) {
    const tr = gd.data[i];
    const kind = tr && tr.meta && typeof tr.meta === "object" ? String(tr.meta.kind || "") : "";
    if (kind === "points") {
      pointTraceIndex = i;
      break;
    }
  }
  if (pointTraceIndex < 0 && gd.data.length > 0) pointTraceIndex = 0;
  if (pointTraceIndex < 0) return;

  const tr = gd.data[pointTraceIndex];
  const cds = Array.isArray(tr.customdata) ? tr.customdata : [];
  const selectedMarkerSize = Number(latestArgs.selected_marker_size || 10.0);
  const dimMarkerOpacity = Number(latestArgs.dim_marker_opacity || 0.28);
  const dimMarkerSize = Math.max(2.0, Number(selectedMarkerSize * 0.8));
  const applyToken = Number(ctx.state && ctx.state.__applyVersion != null ? ctx.state.__applyVersion : 0);
  const dataKey = cds.map((cd) => caseIdFromCustomData(cd)).join("|");
  const scatterKey = `${applyToken}|${ctx.state.colorBy}|${dataKey}`;
  const doForce = Boolean(forceApply);
  try {
    if (!doForce && gd.__fsScatterApplyKey === scatterKey) return;
  } catch (e) {}

  const currentArr = buildScatterMarkerArrays(cds, ctx, allowedSet, selectedMarkerSize, dimMarkerOpacity, dimMarkerSize);
  const colors = currentArr.colors;
  const opacities = currentArr.opacities;
  const sizes = currentArr.sizes;
  const filteredCount = currentArr.filteredCount;

  const frames =
    (gd._transitionData && Array.isArray(gd._transitionData._frames) && gd._transitionData._frames) ||
    (Array.isArray(gd.frames) ? gd.frames : []);
  const freqSteps = Array.isArray(frames) && frames.length > 0 ? frames.length : Number(latestArgs.rx_freq_steps || 0);
  updateRxStatusLabel(filteredCount, freqSteps);

  try {
    const frameStyleToken = `${applyToken}|${ctx.state.colorBy}`;
    const hasFrames = Array.isArray(frames) && frames.length > 0;
    const frameRefChanged = hasFrames && gd.__fsScatterFrameRef !== frames;
    const shouldPatchFrames = hasFrames && (gd.__fsScatterFrameStyleToken !== frameStyleToken || frameRefChanged);
    if (shouldPatchFrames) {
      for (let fi = 0; fi < frames.length; fi++) {
        const fr = frames[fi];
        if (!fr || !Array.isArray(fr.data) || fr.data.length === 0) continue;
        const frTr = fr.data[0];
        if (!frTr) continue;
        const frCds = Array.isArray(frTr.customdata) ? frTr.customdata : [];
        const arr = buildScatterMarkerArrays(frCds, ctx, allowedSet, selectedMarkerSize, dimMarkerOpacity, dimMarkerSize);
        if (!frTr.marker || typeof frTr.marker !== "object") frTr.marker = {};
        frTr.marker.color = arr.colors;
        frTr.marker.opacity = arr.opacities;
        frTr.marker.size = arr.sizes;
      }
      gd.__fsScatterFrameStyleToken = frameStyleToken;
      gd.__fsScatterFrameRef = frames;
    }
  } catch (e) {}

  try {
    Plotly.restyle(
      gd,
      {
        "marker.color": [colors],
        "marker.opacity": [opacities],
        "marker.size": [sizes],
      },
      [pointTraceIndex],
    );
    gd.__fsScatterApplyKey = scatterKey;
  } catch (e) {}
}

function bindScatterHandlers(ctx, gd, plotIndex) {
  if (!gd || !gd.on) return;
  const renderNonce = Number(latestArgs && latestArgs.render_nonce ? latestArgs.render_nonce : 0);
  const bindKey = `${ctx.dataId}|${ctx.chartId}|${plotIndex}|${renderNonce}`;
  try {
    if (gd.__fsCaseUiScatterKey === bindKey) return;
  } catch (e) {}

  try {
    if (gd.__fsCaseUiClick && gd.removeListener) gd.removeListener("plotly_click", gd.__fsCaseUiClick);
    if (gd.__fsCaseUiAnimated && gd.removeListener) gd.removeListener("plotly_animated", gd.__fsCaseUiAnimated);
    if (gd.__fsCaseUiSlider && gd.removeListener) gd.removeListener("plotly_sliderchange", gd.__fsCaseUiSlider);
  } catch (e) {}

  const clickHandler = (evt) => {
    const nextCtx = ensureContext();
    if (!nextCtx || !Boolean(latestArgs.enable_selection)) return;
    try {
      if (!evt || !Array.isArray(evt.points) || evt.points.length === 0) return;
      const caseId = caseIdFromCustomData(evt.points[0].customdata);
      if (!caseId || !nextCtx.caseById.has(caseId)) return;
      if (nextCtx.state.selectedCases.has(caseId)) nextCtx.state.selectedCases.delete(caseId);
      else nextCtx.state.selectedCases.add(caseId);
      bumpStateVersion(nextCtx.state);
      nextCtx.state.importStatus = "";
      renderPanel();
      applyAllPlots();
    } catch (e) {}
  };
  const refreshHandler = () => {
    const nextCtx = ensureContext();
    if (!nextCtx) return;
    const allowed = computeAllowedSet(nextCtx);
    applyScatterStyles(nextCtx, gd, allowed, true);
  };

  try {
    gd.__fsCaseUiScatterKey = bindKey;
    gd.__fsCaseUiClick = clickHandler;
    gd.__fsCaseUiAnimated = refreshHandler;
    gd.__fsCaseUiSlider = refreshHandler;
  } catch (e) {}

  gd.on("plotly_click", clickHandler);
  gd.on("plotly_animated", refreshHandler);
  gd.on("plotly_sliderchange", refreshHandler);
}

function applyAllPlots() {
  const ctx = ensureContext();
  if (!ctx) {
    return { touchedCount: 0, matchedCount: 0, expectedCount: 0 };
  }
  const plotIds = asArray(latestArgs.plot_ids).map((v) => String(v || ""));
  const plots = getPlotDivs();
  const allowedSet = computeAllowedSet(ctx);
  let touchedCount = 0;
  let matchedCount = 0;
  const expectedCount = plotIds.length;

  for (let i = 0; i < plotIds.length; i++) {
    const plotId = plotIds[i];
    if (i < 0 || i >= plots.length) continue;
    const gd = plots[i];
    if (!gd) continue;
    if (!Array.isArray(gd.data)) continue;
    matchedCount += 1;
    touchedCount += 1;
    if (plotId === "rx") {
      bindScatterHandlers(ctx, gd, i);
      applyScatterStyles(ctx, gd, allowedSet, false);
    } else {
      applyLineBaseFrequency(ctx, gd);
      applyLineStyles(ctx, gd, allowedSet);
      applyHarmonicsToLinePlot(ctx, gd);
    }
  }
  return { touchedCount, matchedCount, expectedCount };
}

function scheduleApplyPasses() {
  const myNonce = ++applyPassNonce;
  const splineEnabled = Boolean(latestArgs && latestArgs.spline_enabled);
  // Keep a short stabilization sequence for non-spline, and a longer tail for spline.
  // Stop scheduling once all expected plot DOMs are matched for a minimum number of passes.
  const delays = splineEnabled ? [0, 120, 320, 700, 1300, 2200, 3500] : [0, 120, 320, 700];
  const minPassBeforeStop = splineEnabled ? 3 : 2;
  let stabilized = false;
  for (let idx = 0; idx < delays.length; idx++) {
    const d = delays[idx];
    setTimeout(() => {
      if (myNonce !== applyPassNonce || stabilized) return;
      const res = applyAllPlots() || {};
      const expected = Number(res.expectedCount || 0);
      const matched = Number(res.matchedCount || 0);
      if (idx >= minPassBeforeStop && expected > 0 && matched >= expected) {
        stabilized = true;
      }
    }, d);
  }
}

function scheduleApplyOnce() {
  if (applyQueued) return;
  applyQueued = true;
  const run = () => {
    applyQueued = false;
    applyAllPlots();
  };
  if (typeof window.requestAnimationFrame === "function") {
    window.requestAnimationFrame(run);
  } else {
    setTimeout(run, 0);
  }
}

function sendFrameHeightIfNeeded() {
  try {
    const panel = document.querySelector(".fs-panel");
    const rawPanelH = panel ? Number(panel.scrollHeight || 0) + 12 : 0;
    const rawDocH = (document.documentElement && Number(document.documentElement.scrollHeight || 0)) || 0;
    const rawBodyH = (document.body && Number(document.body.scrollHeight || 0)) || 0;
    const raw = Math.max(rawPanelH, rawDocH, rawBodyH, 260);
    const nextH = Math.max(260, Math.min(5000, Math.ceil(raw)));
    if (Math.abs(nextH - Number(lastSentFrameHeight || 0)) < 2) return;
    lastSentFrameHeight = nextH;
    sendToStreamlit("streamlit:setFrameHeight", { height: nextH });
  } catch (e) {}
}

function scheduleFrameHeightSync() {
  sendFrameHeightIfNeeded();
  setTimeout(sendFrameHeightIfNeeded, 80);
}

function renderPanel() {
  const ctx = ensureContext();
  if (!ctx) return;
  registerExternalSelectionApi(ctx);
  const renderNonceNow = Number(latestArgs && latestArgs.render_nonce ? latestArgs.render_nonce : 0);
  const isFreshStreamlitRender = renderNonceNow !== lastRenderNonceSeen;
  lastRenderNonceSeen = renderNonceNow;
  const root = document.getElementById("app");
  if (!root) return;
  const allowedSet = computeAllowedSet(ctx);
  const rows = selectedRowsForTable(ctx, allowedSet);
  const hiddenSelectedCount = rows.filter((r) => r.hidden).length;
  const activePartColorInfo = buildActivePartValueColorMap(ctx);

  const partBlocks = [];
  for (let i = 0; i < ctx.partLabels.length; i++) {
    const opts = ctx.partOptions[i] || [];
    const sel = ctx.state.selectedByPart[i] instanceof Set ? ctx.state.selectedByPart[i] : new Set();
    const checksHtml = opts
      .map((v, optIdx) => {
        const checkedAttr = sel.has(v) ? " checked" : "";
        const partValue = String(v || "");
        const showColorDot = activePartColorInfo.partIdx === i;
        const dotColor = showColorDot ? String(activePartColorInfo.valueColorMap.get(partValue) || "#1f77b4") : "";
        const dotHtml = showColorDot
          ? `<span class="fs-color-dot" style="background:${escapeHtml(dotColor)};" title="${escapeHtml(dotColor)}"></span>`
          : "";
        return `
          <label class="fs-check-item">
            <input type="checkbox" data-part-check="${i}" data-part-opt="${optIdx}"${checkedAttr} />
            <span class="fs-check-text"><span>${escapeHtml(displayValue(v))}</span>${dotHtml}</span>
          </label>
        `;
      })
      .join("");
    partBlocks.push(`
      <div class="fs-block">
        <div class="fs-row fs-row-space">
          <div class="fs-title">${escapeHtml(ctx.partLabels[i])}</div>
          <div class="fs-mini-actions">
            <button type="button" class="fs-mini-btn" data-part-all="${i}">All</button>
            <button type="button" class="fs-mini-btn" data-part-none="${i}">None</button>
          </div>
        </div>
        <div class="fs-check-list">
          ${checksHtml}
        </div>
      </div>
    `);
  }

  const activeBase = Number(ctx.state.baseFrequencyHz || 50);
  const baseFreqButtonsHtml = (Array.isArray(ctx.baseFreqOptions) ? ctx.baseFreqOptions : [])
    .map((hz) => {
      const f = Number(hz);
      const activeCls = Math.abs(f - activeBase) < 1e-9 ? " fs-mini-btn-active" : "";
      return `<button type="button" class="fs-mini-btn${activeCls}" data-base-freq="${f}">${Number.isFinite(f) ? `${Math.round(f)} Hz` : "?"}</button>`;
    })
    .join("");

  const colorOptionsHtml = ctx.colorOptions
    .map((opt) => {
      const selectedAttr = String(opt) === String(ctx.state.colorBy) ? " selected" : "";
      return `<option value="${escapeHtml(opt)}"${selectedAttr}>${escapeHtml(opt)}</option>`;
    })
    .join("");

  const rowsHtml = rows.length
    ? rows
        .map(
          (r) => `
            <tr>
              <td>${escapeHtml(r.displayCase)}</td>
              <td>${r.hidden ? "Hidden" : "Visible"}</td>
              <td><button type="button" class="fs-mini-btn fs-mini-btn-del" data-remove-case="${escapeHtml(r.caseId)}" title="Remove">Del</button></td>
            </tr>
          `,
        )
        .join("")
    : `<tr><td colspan="3" class="fs-empty">No selected cases</td></tr>`;

  const selectionDisabledNote =
    Boolean(latestArgs.enable_selection) || latestArgs.enable_selection == null
      ? ""
      : `<div class="fs-note">Scatter is hidden. Use import list to build a selection.</div>`;
  const showSelectionToolbarInPanel = !Boolean(latestArgs.enable_selection);
  const selectionToolbarHtml = showSelectionToolbarInPanel
    ? `
      <label class="fs-row">
        <input id="fs-show-only" type="checkbox" ${ctx.state.showOnlySelected ? "checked" : ""} />
        <span style="font-size:12px;">Show only selected sweeps</span>
      </label>
      <div class="fs-actions">
        <button type="button" id="fs-clear-list" class="fs-btn">Clear list</button>
        <button type="button" id="fs-download-csv" class="fs-btn fs-btn-primary">Download selected CSV</button>
      </div>
    `
    : "";

  root.innerHTML = `
    <style>
      :root {
        color-scheme: light;
        --fs-bg: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
        --fs-border: #cfd7e3;
        --fs-text: #1f2937;
        --fs-muted: #6b7280;
        --fs-soft: #eef2f8;
        --fs-accent: #1f6fd6;
        --fs-accent-soft: #e8f1ff;
      }

      html, body { margin: 0; padding: 0; }
      body {
        font-family: "Open Sans", "Segoe UI", sans-serif;
        color: var(--fs-text);
      }

      .fs-panel {
        border: 1px solid var(--fs-border);
        border-radius: 14px;
        padding: 12px;
        background: var(--fs-bg);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
      }

      .fs-section {
        margin: 12px 0 0 0;
        padding-top: 10px;
        border-top: 1px solid #e3e9f2;
      }

      .fs-section-tight {
        margin-top: 0;
        padding-top: 0;
        border-top: none;
      }

      .fs-row {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .fs-row-space { justify-content: space-between; }
      .fs-row-inline { margin-top: 6px; }

      .fs-title {
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.01em;
      }

      .fs-subtitle {
        font-weight: 700;
        font-size: 13px;
        margin: 0 0 8px 0;
        letter-spacing: 0.01em;
      }

      .fs-block { margin: 10px 0 0 0; }
      .fs-block:last-of-type { margin-bottom: 0; }

      .fs-mini-actions {
        display: inline-flex;
        gap: 6px;
      }

      .fs-mini-actions-segment {
        background: #edf2fa;
        border: 1px solid #d4deeb;
        border-radius: 11px;
        padding: 3px;
        gap: 4px;
      }

      .fs-mini-btn,
      .fs-btn {
        border: 1px solid #b9c5d6;
        background: #ffffff;
        color: #223043;
        border-radius: 8px;
        cursor: pointer;
        transition: background 120ms ease, border-color 120ms ease, box-shadow 120ms ease, color 120ms ease;
      }

      .fs-mini-btn {
        padding: 3px 9px;
        font-size: 12px;
        font-weight: 600;
      }

      .fs-mini-btn-del {
        padding: 2px 7px;
        font-size: 11px;
        min-width: 34px;
      }

      .fs-btn {
        padding: 6px 11px;
        font-size: 12px;
        font-weight: 600;
      }

      .fs-btn-primary {
        background: var(--fs-accent-soft);
        border-color: #91b2e5;
        color: #18488f;
      }

      .fs-mini-btn-active {
        background: var(--fs-accent-soft);
        border-color: #7aa0dc;
        color: #18488f;
        box-shadow: inset 0 0 0 1px rgba(31, 111, 214, 0.1);
      }

      .fs-mini-btn:hover,
      .fs-btn:hover {
        background: #f6f9ff;
        border-color: #96aeca;
      }

      .fs-mini-btn:focus-visible,
      .fs-btn:focus-visible,
      .fs-select:focus-visible,
      .fs-textarea:focus-visible {
        outline: none;
        border-color: #5f93da;
        box-shadow: 0 0 0 2px rgba(31, 111, 214, 0.18);
      }

      .fs-select,
      .fs-textarea {
        width: 100%;
        box-sizing: border-box;
        border: 1px solid #c4cfdd;
        border-radius: 9px;
        background: #ffffff;
        color: #111827;
        font-size: 13px;
      }

      .fs-select { padding: 6px 8px; }
      .fs-textarea {
        min-height: 64px;
        resize: vertical;
        padding: 8px;
        line-height: 1.35;
      }

      .fs-check-list {
        border: 1px solid #d4ddea;
        border-radius: 10px;
        padding: 6px;
        max-height: 180px;
        overflow: auto;
        background: #ffffff;
      }

      .fs-check-item {
        display: flex;
        align-items: center;
        gap: 7px;
        font-size: 12px;
        line-height: 1.25;
        margin: 2px 0;
        padding: 4px 5px;
        border-radius: 7px;
        user-select: none;
      }

      .fs-check-item:hover { background: #f5f8ff; }
      .fs-check-item input[type="checkbox"] { accent-color: var(--fs-accent); }

      .fs-check-text {
        display: inline-flex;
        align-items: center;
        gap: 7px;
      }

      .fs-color-dot {
        display: inline-block;
        width: 11px;
        height: 11px;
        border-radius: 50%;
        border: 1px solid rgba(0, 0, 0, 0.28);
        box-sizing: border-box;
        flex: 0 0 auto;
      }

      .fs-note {
        font-size: 12px;
        color: var(--fs-muted);
        margin: 4px 0 0 0;
      }

      .fs-label {
        font-size: 12px;
        color: #2c3748;
      }

      .fs-label-fixed { min-width: 110px; }

      .fs-table-wrap {
        max-height: 220px;
        overflow: auto;
        border: 1px solid #d9e1ed;
        border-radius: 9px;
        margin-top: 6px;
        background: #fff;
      }

      table.fs-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
      }

      .fs-table th,
      .fs-table td {
        border-bottom: 1px solid #edf1f7;
        padding: 6px;
        text-align: left;
        vertical-align: middle;
      }

      .fs-table tbody tr:hover td { background: #f8faff; }

      .fs-table th {
        position: sticky;
        top: 0;
        background: #f5f8fc;
        z-index: 1;
      }

      .fs-empty {
        color: #7c7c7c;
        text-align: center;
      }

      .fs-status {
        font-size: 12px;
        margin-top: 5px;
        color: #3f4d5f;
        min-height: 16px;
      }

      .fs-actions {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 8px;
      }
    </style>

    <div class="fs-panel">
      <section class="fs-section fs-section-tight">
        <div class="fs-row fs-row-space">
          <div class="fs-title">Base frequency</div>
          <div class="fs-mini-actions fs-mini-actions-segment">${baseFreqButtonsHtml}</div>
        </div>
      </section>

      <section class="fs-section">
        <div class="fs-row fs-row-space">
          <div class="fs-title">Case part filters</div>
          <button type="button" id="fs-reset-filters" class="fs-btn">Reset filters</button>
        </div>
        ${partBlocks.join("")}
      </section>

      <section class="fs-section">
        <div class="fs-subtitle">Color</div>
        <select id="fs-color-by" class="fs-select">${colorOptionsHtml}</select>
      </section>

      <section class="fs-section">
        <div class="fs-subtitle">Selection</div>
        ${selectionDisabledNote}
        ${selectionToolbarHtml}
        <div class="fs-note">Selected: ${rows.length} case(s)${hiddenSelectedCount > 0 ? ` | Hidden by filters: ${hiddenSelectedCount}` : ""}</div>
      </section>

      <section class="fs-section">
        <div class="fs-subtitle">Add cases to selection</div>
        <textarea id="fs-import-text" class="fs-textarea" placeholder="Paste case IDs or display names (space/comma/newline separated)"></textarea>
        <div class="fs-actions">
          <button type="button" id="fs-import-add" class="fs-btn">Add list</button>
        </div>
        <div id="fs-import-status" class="fs-status">${escapeHtml(ctx.state.importStatus || "")}</div>
      </section>

      <section class="fs-section">
        <div class="fs-subtitle">Selected cases table</div>
        <div class="fs-table-wrap">
          <table class="fs-table">
            <thead>
              <tr><th>Case</th><th>Status</th><th>Action</th></tr>
            </thead>
            <tbody>${rowsHtml}</tbody>
          </table>
        </div>
      </section>

      <section class="fs-section">
        <div class="fs-subtitle">Harmonics</div>
        <label class="fs-row fs-row-inline">
          <input id="fs-show-harmonics" type="checkbox" ${ctx.state.showHarmonics ? "checked" : ""} />
          <span class="fs-label">Show harmonic lines</span>
        </label>
        <label class="fs-row fs-row-inline">
          <span class="fs-label fs-label-fixed">Bin width (Hz)</span>
          <input id="fs-bin-width" type="number" min="0" step="1" value="${Number(ctx.state.binWidthHz || 0)}" class="fs-select" />
        </label>
      </section>
    </div>
  `;

  const applyUiChange = () => {
    bumpStateVersion(ctx.state);
    applyPassNonce += 1;
    renderPanel();
    scheduleApplyOnce();
  };

  const btnResetFilters = document.getElementById("fs-reset-filters");
  if (btnResetFilters) {
    btnResetFilters.addEventListener("click", () => {
      for (let i = 0; i < ctx.partOptions.length; i++) {
        ctx.state.selectedByPart[i] = new Set(ctx.partOptions[i] || []);
      }
      ctx.state.importStatus = "";
      applyUiChange();
    });
  }

  for (const btn of Array.from(document.querySelectorAll("button[data-part-all]"))) {
    btn.addEventListener("click", () => {
      const idx = Number(btn.getAttribute("data-part-all"));
      if (!Number.isFinite(idx) || idx < 0 || idx >= ctx.partOptions.length) return;
      ctx.state.selectedByPart[idx] = new Set(ctx.partOptions[idx] || []);
      ctx.state.importStatus = "";
      applyUiChange();
    });
  }
  for (const btn of Array.from(document.querySelectorAll("button[data-part-none]"))) {
    btn.addEventListener("click", () => {
      const idx = Number(btn.getAttribute("data-part-none"));
      if (!Number.isFinite(idx) || idx < 0 || idx >= ctx.partOptions.length) return;
      ctx.state.selectedByPart[idx] = new Set();
      ctx.state.importStatus = "";
      applyUiChange();
    });
  }
  for (const btn of Array.from(document.querySelectorAll("button[data-base-freq]"))) {
    btn.addEventListener("click", () => {
      const hz = Number(btn.getAttribute("data-base-freq"));
      if (!Number.isFinite(hz) || hz <= 0) return;
      if (Math.abs(Number(ctx.state.baseFrequencyHz || 0) - hz) < 1e-9) return;
      ctx.state.baseFrequencyHz = hz;
      ctx.state.importStatus = "";
      applyUiChange();
    });
  }
  for (const inp of Array.from(document.querySelectorAll("input[data-part-check]"))) {
    inp.addEventListener("change", () => {
      const idx = Number(inp.getAttribute("data-part-check"));
      const optIdx = Number(inp.getAttribute("data-part-opt"));
      if (!Number.isFinite(idx) || idx < 0 || idx >= ctx.partOptions.length) return;
      if (!Number.isFinite(optIdx) || optIdx < 0 || optIdx >= (ctx.partOptions[idx] || []).length) return;
      const val = String(ctx.partOptions[idx][optIdx] || "");
      const set = ctx.state.selectedByPart[idx] instanceof Set ? new Set(ctx.state.selectedByPart[idx]) : new Set();
      if (Boolean(inp.checked)) set.add(val);
      else set.delete(val);
      ctx.state.selectedByPart[idx] = set;
      ctx.state.importStatus = "";
      applyUiChange();
    });
  }

  const colorBy = document.getElementById("fs-color-by");
  if (colorBy) {
    colorBy.addEventListener("change", () => {
      ctx.state.colorBy = String(colorBy.value || "Auto");
      ctx.state.importStatus = "";
      applyUiChange();
    });
  }

  const showOnly = document.getElementById("fs-show-only");
  if (showOnly) {
    showOnly.addEventListener("change", () => {
      ctx.state.showOnlySelected = Boolean(showOnly.checked);
      ctx.state.importStatus = "";
      applyUiChange();
    });
  }

  const showHarmonics = document.getElementById("fs-show-harmonics");
  if (showHarmonics) {
    showHarmonics.addEventListener("change", () => {
      ctx.state.showHarmonics = Boolean(showHarmonics.checked);
      scheduleApplyOnce();
    });
  }

  const binWidth = document.getElementById("fs-bin-width");
  if (binWidth) {
    const onBinWidth = () => {
      const v = Number(binWidth.value);
      ctx.state.binWidthHz = Number.isFinite(v) && v >= 0 ? v : 0;
      scheduleApplyOnce();
    };
    binWidth.addEventListener("input", onBinWidth);
    binWidth.addEventListener("change", onBinWidth);
  }

  const clearList = document.getElementById("fs-clear-list");
  if (clearList) {
    clearList.addEventListener("click", () => {
      ctx.state.selectedCases = new Set();
      ctx.state.importStatus = "";
      applyUiChange();
    });
  }

  const addList = document.getElementById("fs-import-add");
  const importText = document.getElementById("fs-import-text");
  if (addList && importText) {
    addList.addEventListener("click", () => {
      const tokens = tokenizeImportList(importText.value || "");
      let added = 0;
      let missing = 0;
      for (const tok of tokens) {
        const key = tok.toLowerCase();
        const direct = ctx.exactLookup.get(key);
        if (direct) {
          if (!ctx.state.selectedCases.has(direct)) {
            ctx.state.selectedCases.add(direct);
            added += 1;
          }
          continue;
        }
        const byDisplay = ctx.displayLookup.get(key) || [];
        if (!byDisplay.length) {
          missing += 1;
          continue;
        }
        for (const cid of byDisplay) {
          if (!ctx.state.selectedCases.has(cid)) {
            ctx.state.selectedCases.add(cid);
            added += 1;
          }
        }
      }
      ctx.state.importStatus = `Added ${added} case(s)` + (missing > 0 ? ` | Not found: ${missing}` : "");
      applyUiChange();
    });
  }

  const downloadCsv = document.getElementById("fs-download-csv");
  if (downloadCsv) {
    downloadCsv.addEventListener("click", () => {
      downloadSelectedCsv(rows);
    });
  }

  for (const btn of Array.from(document.querySelectorAll("button[data-remove-case]"))) {
    btn.addEventListener("click", () => {
      const caseId = String(btn.getAttribute("data-remove-case") || "");
      if (!caseId) return;
      ctx.state.selectedCases.delete(caseId);
      ctx.state.importStatus = "";
      applyUiChange();
    });
  }

  if (isFreshStreamlitRender) {
    try {
      const sc = document.scrollingElement || document.documentElement || document.body;
      if (sc) sc.scrollTop = 0;
      const panel = document.querySelector(".fs-panel");
      if (panel) panel.scrollTop = 0;
    } catch (e) {}
  }
  scheduleFrameHeightSync();
}

window.addEventListener("message", (event) => {
  const data = event && event.data ? event.data : null;
  if (!data || data.type !== "streamlit:render") return;
  latestArgs = data.args || {};
  renderPanel();
  scheduleApplyPasses();
});

sendToStreamlit("streamlit:componentReady", { apiVersion: 1 });
sendToStreamlit("streamlit:setFrameHeight", { height: 260 });
