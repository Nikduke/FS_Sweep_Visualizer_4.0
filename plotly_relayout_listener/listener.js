// Minimal Streamlit component (no build tooling), using Streamlit's postMessage protocol:
// - Binds to Plotly charts rendered by Streamlit (`stPlotlyChart` blocks)
// - Captures Plotly `plotly_relayout` events
// - Persists ranges in browser localStorage (so zoom does not trigger a Streamlit rerun)

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
let debounceMs = 120;
let lastDataId = null;
let saveTimersByKey = new Map();
let lastResetToken = null;
let bindTries = 80;
let bindIntervalMs = 100;
let ignoreAutorangeMs = 1200;

function nowMs() {
  return Date.now ? Date.now() : new Date().getTime();
}

function storageKey(dataId, plotIndex) {
  return `fsSweepZoom:${String(dataId)}:${String(plotIndex)}`;
}

function safeLocalStorageGet(key) {
  try {
    return window.localStorage ? window.localStorage.getItem(key) : null;
  } catch (e) {
    return null;
  }
}

function safeLocalStorageSet(key, value) {
  try {
    if (!window.localStorage) return;
    window.localStorage.setItem(key, value);
  } catch (e) {}
}

function safeLocalStorageRemove(key) {
  try {
    if (!window.localStorage) return;
    window.localStorage.removeItem(key);
  } catch (e) {}
}

function scheduleSave(key, jsonValue) {
  const t = saveTimersByKey.get(key);
  if (t) clearTimeout(t);
  const timer = setTimeout(() => {
    saveTimersByKey.delete(key);
    if (jsonValue === null) safeLocalStorageRemove(key);
    else safeLocalStorageSet(key, jsonValue);
  }, Math.max(0, Number(debounceMs || 0)));
  saveTimersByKey.set(key, timer);
}

function getStreamlitPlotDivsFromDoc(doc) {
  const out = [];
  try {
    const blocks = doc?.querySelectorAll?.('div[data-testid="stPlotlyChart"], div.stPlotlyChart') || [];
    for (const block of blocks) {
      // Prefer Plotly inside chart iframe if present.
      const fr = block.querySelector?.("iframe");
      if (fr && fr.contentWindow && fr.contentWindow.document) {
        const gd = fr.contentWindow.document.querySelector?.("div.js-plotly-plot");
        if (gd) {
          out.push(gd);
          continue;
        }
      }
      // Fallback: Plotly directly in the block.
      const gd2 = block.querySelector?.("div.js-plotly-plot");
      if (gd2) out.push(gd2);
    }
  } catch (e) {}
  return out;
}

function getPlotDivs() {
  // Only bind to Streamlit plotly charts. Do NOT bind to other Plotly instances
  // (e.g., offscreen exporter plots created in component iframes).
  const parentDoc = (window.parent || window).document;
  return getStreamlitPlotDivsFromDoc(parentDoc);
}

function applyStoredZoomIfAny(gd, idx, dataId) {
  const key = storageKey(dataId, idx);

  try {
    if (gd.__fsZoomAppliedKey === key) return;
    gd.__fsZoomAppliedKey = key;
  } catch (e) {}

  const raw = safeLocalStorageGet(key);
  if (!raw) return;

  let saved = null;
  try {
    saved = JSON.parse(raw);
  } catch (e) {
    return;
  }
  if (!saved || typeof saved !== "object") return;

  const update = {};
  if (Array.isArray(saved.x) && saved.x.length === 2) {
    update["xaxis.range"] = saved.x;
    update["xaxis.autorange"] = false;
  }
  if (Array.isArray(saved.y) && saved.y.length === 2) {
    update["yaxis.range"] = saved.y;
    update["yaxis.autorange"] = false;
  }
  if (Object.keys(update).length === 0) return;

  const win = gd?.ownerDocument?.defaultView;
  const Plotly = win && win.Plotly ? win.Plotly : null;
  if (!Plotly || !Plotly.relayout) return;

  try {
    gd.__fsApplyingZoom = true;
  } catch (e) {}

  try {
    const p = Plotly.relayout(gd, update);
    if (p && typeof p.then === "function") {
      p.then(
        () => {
          try {
            gd.__fsApplyingZoom = false;
          } catch (e) {}
        },
        () => {
          try {
            gd.__fsApplyingZoom = false;
          } catch (e) {}
        },
      );
    } else {
      try {
        gd.__fsApplyingZoom = false;
      } catch (e) {}
    }
  } catch (e) {
    try {
      gd.__fsApplyingZoom = false;
    } catch (e2) {}
  }
}

function bindOne(gd, idx, dataId) {
  if (!gd || !gd.on) return;
  try {
    if (gd.__fsRelayoutHandler && gd.removeListener) {
      gd.removeListener("plotly_relayout", gd.__fsRelayoutHandler);
    }
  } catch (e) {}

  try {
    gd.__fsBindTimeMs = nowMs();
  } catch (e) {}

  const handler = (evt) => {
    try {
      if (gd.__fsApplyingZoom) return;
      if (!evt || typeof evt !== "object") return;
      const payload = { data_id: String(dataId), plot_index: idx };

      if (evt["xaxis.autorange"] === true) payload.xautorange = true;
      if (evt["yaxis.autorange"] === true) payload.yautorange = true;

      if (evt["xaxis.range[0]"] != null && evt["xaxis.range[1]"] != null) {
        payload.x0 = evt["xaxis.range[0]"];
        payload.x1 = evt["xaxis.range[1]"];
      }
      if (evt["yaxis.range[0]"] != null && evt["yaxis.range[1]"] != null) {
        payload.y0 = evt["yaxis.range[0]"];
        payload.y1 = evt["yaxis.range[1]"];
      }

      // Ignore events that don't carry axis info.
      if (
        payload.x0 === undefined &&
        payload.xautorange !== true &&
        payload.y0 === undefined &&
        payload.yautorange !== true
      ) {
        return;
      }

      // During Streamlit reruns, Plotly may emit an initial autorange relayout on mount.
      // Ignore autorange-only events shortly after binding so we don't clear saved zoom unintentionally.
      try {
        const boundAt = Number(gd.__fsBindTimeMs || 0);
        const age = nowMs() - boundAt;
        const autorangeOnly =
          (payload.xautorange === true || payload.yautorange === true) &&
          payload.x0 === undefined &&
          payload.y0 === undefined;
        if (autorangeOnly && age >= 0 && age < Number(ignoreAutorangeMs || 0)) return;
      } catch (e) {}

      const key = storageKey(dataId, idx);
      const existingRaw = safeLocalStorageGet(key);
      let existing = null;
      try {
        existing = existingRaw ? JSON.parse(existingRaw) : null;
      } catch (e) {
        existing = null;
      }
      if (!existing || typeof existing !== "object") existing = {};

      const next = { ...existing };

      if (payload.xautorange === true) delete next.x;
      else if (payload.x0 !== undefined && payload.x1 !== undefined) next.x = [payload.x0, payload.x1];

      if (payload.yautorange === true) delete next.y;
      else if (payload.y0 !== undefined && payload.y1 !== undefined) next.y = [payload.y0, payload.y1];

      const hasAny = Array.isArray(next.x) || Array.isArray(next.y);
      const nextRaw = hasAny ? JSON.stringify(next) : null;
      if (existingRaw === nextRaw) return;

      scheduleSave(key, nextRaw);
    } catch (e) {}
  };

  try {
    gd.__fsRelayoutHandler = handler;
  } catch (e) {}
  gd.on("plotly_relayout", handler);
  applyStoredZoomIfAny(gd, idx, dataId);
}

function syncBindings() {
  if (!latestArgs) return;
  const plotCount = Number(latestArgs.plot_count || 3);
  const dataId = String(latestArgs.data_id || "");
  const plots = getPlotDivs();
  const n = Math.min(plotCount, plots.length);
  for (let i = 0; i < n; i++) bindOne(plots[i], i, dataId);
}

function kickRebindLoop() {
  let tries = 0;
  (function tick() {
    syncBindings();
    tries += 1;
    if (tries < Number(bindTries || 0)) setTimeout(tick, Number(bindIntervalMs || 0));
  })();
}

window.addEventListener("message", (event) => {
  // Incoming Streamlit messages may or may not include an `isStreamlitMessage` flag.
  // Match by `type` to avoid dropping the render message (which would prevent any binding).
  const data = event && event.data ? event.data : null;
  if (!data || data.type !== "streamlit:render") return;
  latestArgs = data.args || {};
  debounceMs = Number(latestArgs.debounce_ms || 120);
  bindTries = Number(latestArgs.bind_tries || 80);
  bindIntervalMs = Number(latestArgs.bind_interval_ms || 100);
  ignoreAutorangeMs = Number(latestArgs.ignore_autorange_ms || 1200);
  try {
    const newDataId = String(latestArgs.data_id || "");
    if (lastDataId !== newDataId) {
      lastDataId = newDataId;
      saveTimersByKey = new Map();
    }
  } catch (e) {}

  // If Python indicates an upload reset, clear stored zoom for this data_id so charts start autoscaled.
  try {
    const resetTok = Number(latestArgs.reset_token || 0);
    const dataId = String(latestArgs.data_id || "");
    if (lastResetToken !== resetTok) {
      lastResetToken = resetTok;
      if (dataId) {
        const prefix = `fsSweepZoom:${dataId}:`;
        const toDelete = [];
        for (let i = 0; i < window.localStorage.length; i++) {
          const k = window.localStorage.key(i);
          if (k && k.startsWith(prefix)) toDelete.push(k);
        }
        for (const k of toDelete) {
          try {
            window.localStorage.removeItem(k);
          } catch (e2) {}
        }
      }
    }
  } catch (e) {}
  try {
    sendToStreamlit("streamlit:setFrameHeight", { height: 0 });
  } catch (e) {}
  kickRebindLoop();
});

sendToStreamlit("streamlit:componentReady", { apiVersion: 1 });
