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

let latestArgs = {};
let syncTimer = null;

function asNumber(name, fallback) {
  const v = Number(latestArgs && latestArgs[name]);
  return Number.isFinite(v) ? v : Number(fallback);
}

function asString(name, fallback) {
  const v = latestArgs ? latestArgs[name] : null;
  if (v == null) return String(fallback);
  return String(v);
}

function getPlots() {
  const out = [];
  try {
    const doc = window.parent.document;
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

function getSelectionApi(stateKey) {
  try {
    const rootWin = window.parent;
    const apiStore = rootWin && rootWin.__fsCaseUiApi && typeof rootWin.__fsCaseUiApi === "object" ? rootWin.__fsCaseUiApi : null;
    if (!apiStore) return null;
    const api = apiStore[stateKey];
    return api && typeof api === "object" ? api : null;
  } catch (e) {}
  return null;
}

function syncSelectionControls(stateKey) {
  try {
    const api = getSelectionApi(stateKey);
    if (!api || typeof api.getState !== "function") return;
    const st = api.getState();
    const cb = document.getElementById("rx-showonly");
    if (cb && st && Object.prototype.hasOwnProperty.call(st, "showOnlySelected")) {
      cb.checked = Boolean(st.showOnlySelected);
    }
  } catch (e) {}
}

function stepFrequency(delta) {
  const plotIndex = Math.max(0, Math.round(asNumber("plot_index", 0)));
  try {
    const plots = getPlots();
    const gd = plots[plotIndex];
    if (!gd) return;
    const frames =
      (gd._transitionData && Array.isArray(gd._transitionData._frames) && gd._transitionData._frames) ||
      (Array.isArray(gd.frames) ? gd.frames : []);
    if (!Array.isArray(frames) || frames.length === 0) return;

    let active = 0;
    try {
      const sliders = gd.layout && Array.isArray(gd.layout.sliders) ? gd.layout.sliders : [];
      if (sliders.length && sliders[0] && sliders[0].active != null) active = Number(sliders[0].active);
    } catch (e) {}
    if (!Number.isFinite(active)) active = 0;
    active = Math.max(0, Math.min(frames.length - 1, Math.floor(active)));
    const next = Math.max(0, Math.min(frames.length - 1, active + Number(delta || 0)));
    const frameObj = frames[next];
    const frameName = frameObj && frameObj.name != null ? String(frameObj.name) : String(next);
    const win = gd.ownerDocument && gd.ownerDocument.defaultView ? gd.ownerDocument.defaultView : null;
    const Plotly = win && win.Plotly ? win.Plotly : null;
    if (!Plotly || !Plotly.animate) return;
    Plotly.animate(gd, [frameName], {
      mode: "immediate",
      frame: { duration: 0, redraw: false },
      transition: { duration: 0 },
    });
  } catch (e) {}
}

function render() {
  const root = document.getElementById("app");
  if (!root) return;

  const dataId = asString("data_id", "");
  const chartId = asString("chart_id", "");
  const stateKey = `${String(dataId)}|${String(chartId)}`;

  root.innerHTML = `
    <style>
      #rx-step-root {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
        margin: 2px 0 6px 0;
        font-family: "Open Sans", verdana, arial, sans-serif;
      }
      #rx-step-root .rx-btn {
        padding: 4px 10px;
        font-size: 12px;
        cursor: pointer;
        font-family: inherit;
        color: #222;
      }
      #rx-step-root .rx-showonly {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #222;
      }
    </style>
    <div id="rx-step-root">
      <button id="rx-prev" type="button" class="rx-btn">&#8592; Prev frequency</button>
      <button id="rx-next" type="button" class="rx-btn">Next frequency &#8594;</button>
      <button id="rx-clear" type="button" class="rx-btn">Clear list</button>
      <button id="rx-csv" type="button" class="rx-btn">Download selected CSV</button>
      <label class="rx-showonly">
        <input id="rx-showonly" type="checkbox" />
        <span>Show only selected sweeps</span>
      </label>
    </div>
  `;

  const prev = document.getElementById("rx-prev");
  const next = document.getElementById("rx-next");
  const showOnly = document.getElementById("rx-showonly");
  const clearBtn = document.getElementById("rx-clear");
  const csvBtn = document.getElementById("rx-csv");

  if (prev) prev.addEventListener("click", (ev) => { ev.preventDefault(); stepFrequency(-1); });
  if (next) next.addEventListener("click", (ev) => { ev.preventDefault(); stepFrequency(1); });
  if (showOnly) {
    showOnly.addEventListener("change", () => {
      try {
        const api = getSelectionApi(stateKey);
        if (api && typeof api.setShowOnlySelected === "function") {
          api.setShowOnlySelected(Boolean(showOnly.checked));
        }
      } catch (e) {}
    });
  }
  if (clearBtn) {
    clearBtn.addEventListener("click", (ev) => {
      ev.preventDefault();
      try {
        const api = getSelectionApi(stateKey);
        if (api && typeof api.clearSelection === "function") {
          api.clearSelection();
          syncSelectionControls(stateKey);
        }
      } catch (e) {}
    });
  }
  if (csvBtn) {
    csvBtn.addEventListener("click", (ev) => {
      ev.preventDefault();
      try {
        const api = getSelectionApi(stateKey);
        if (api && typeof api.downloadSelectedCsv === "function") {
          api.downloadSelectedCsv();
        }
      } catch (e) {}
    });
  }

  try {
    if (syncTimer) {
      clearInterval(syncTimer);
      syncTimer = null;
    }
    syncTimer = setInterval(() => {
      syncSelectionControls(stateKey);
    }, 250);
  } catch (e) {}
  syncSelectionControls(stateKey);
  sendToStreamlit("streamlit:setFrameHeight", { height: 86 });
}

window.addEventListener("message", (event) => {
  const data = event && event.data ? event.data : null;
  if (!data || data.type !== "streamlit:render") return;
  latestArgs = data.args || {};
  render();
});

sendToStreamlit("streamlit:componentReady", { apiVersion: 1 });
sendToStreamlit("streamlit:setFrameHeight", { height: 86 });
