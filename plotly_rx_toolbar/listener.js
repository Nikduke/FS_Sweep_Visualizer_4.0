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
let thresholdDebounceTimer = null;

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

function setInputValueSafe(inputEl, value) {
  if (!inputEl) return;
  if (document.activeElement === inputEl) return;
  inputEl.value = String(value);
}

function syncSelectionControls(stateKey) {
  try {
    const api = getSelectionApi(stateKey);
    if (!api || typeof api.getState !== "function") return;
    const st = api.getState();
    if (!st || typeof st !== "object") return;

    const cbShowOnly = document.getElementById("rx-showonly");
    if (cbShowOnly && Object.prototype.hasOwnProperty.call(st, "showOnlySelected")) {
      cbShowOnly.checked = Boolean(st.showOnlySelected);
    }

    const cbEnerginet = document.getElementById("rx-method-energinet");
    const cbIec = document.getElementById("rx-method-iec");
    const cbIecCollinear = document.getElementById("rx-iec-collinear");
    const t2 = document.getElementById("rx-t2");
    const t3 = document.getElementById("rx-t3");
    const t4 = document.getElementById("rx-t4");
    const enTopN = document.getElementById("rx-en-topn");
    const iecTopN = document.getElementById("rx-iec-topn");
    const note = document.getElementById("rx-method-note");
    const preselectionAvailable = Boolean(st.preselectionAvailable);

    if (cbEnerginet) {
      cbEnerginet.disabled = !preselectionAvailable;
      cbEnerginet.checked = preselectionAvailable && Boolean(st.energinetEnabled);
      const cnt = Number(st.energinetCandidateCount || 0);
      const lbl = document.getElementById("rx-method-energinet-label");
      if (lbl) lbl.textContent = `Energinet (${cnt})`;
    }
    if (cbIec) {
      cbIec.disabled = !preselectionAvailable;
      cbIec.checked = preselectionAvailable && Boolean(st.iecEnabled);
      const cnt = Number(st.iecCandidateCount || 0);
      const lbl = document.getElementById("rx-method-iec-label");
      if (lbl) lbl.textContent = `IEC (${cnt})`;
    }
    if (cbIecCollinear) {
      const iecOn = preselectionAvailable && Boolean(st.iecEnabled);
      cbIecCollinear.disabled = !iecOn;
      cbIecCollinear.checked = iecOn && Boolean(st.iecIncludeCollinear);
    }
    if (t2) {
      t2.disabled = !preselectionAvailable;
      setInputValueSafe(t2, Number(st.energinetT2 || 0));
    }
    if (t3) {
      t3.disabled = !preselectionAvailable;
      setInputValueSafe(t3, Number(st.energinetT3 || 0));
    }
    if (t4) {
      t4.disabled = !preselectionAvailable;
      setInputValueSafe(t4, Number(st.energinetT4 || 0));
    }
    if (enTopN) {
      enTopN.disabled = !preselectionAvailable;
      setInputValueSafe(enTopN, Number(st.energinetTopN || 0));
    }
    if (iecTopN) {
      iecTopN.disabled = !preselectionAvailable;
      setInputValueSafe(iecTopN, Number(st.iecTopN || 0));
    }
    if (note) {
      if (!preselectionAvailable) {
        const err = String(st.preselectionError || "");
        note.textContent = err ? `Preselection unavailable: ${err}` : "Preselection unavailable for this dataset.";
      } else {
        note.textContent = "";
      }
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

function pushEnerginetThresholds(stateKey) {
  const t2El = document.getElementById("rx-t2");
  const t3El = document.getElementById("rx-t3");
  const t4El = document.getElementById("rx-t4");
  const t2 = Number(t2El ? t2El.value : NaN);
  const t3 = Number(t3El ? t3El.value : NaN);
  const t4 = Number(t4El ? t4El.value : NaN);
  try {
    const api = getSelectionApi(stateKey);
    if (api && typeof api.setEnerginetThresholds === "function") {
      api.setEnerginetThresholds({ t2, t3, t4 });
    }
  } catch (e) {}
}

function pushEnerginetTopN(stateKey) {
  const inp = document.getElementById("rx-en-topn");
  const n = Number(inp ? inp.value : NaN);
  try {
    const api = getSelectionApi(stateKey);
    if (api && typeof api.setEnerginetTopN === "function") {
      api.setEnerginetTopN(n);
    }
  } catch (e) {}
}

function pushIecTopN(stateKey) {
  const inp = document.getElementById("rx-iec-topn");
  const n = Number(inp ? inp.value : NaN);
  try {
    const api = getSelectionApi(stateKey);
    if (api && typeof api.setIecTopN === "function") {
      api.setIecTopN(n);
    }
  } catch (e) {}
}

function pushIecCollinearMode(stateKey, flag) {
  try {
    const api = getSelectionApi(stateKey);
    if (api && typeof api.setIecIncludeCollinear === "function") {
      api.setIecIncludeCollinear(flag === true);
    }
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
        display: grid;
        gap: 6px;
        margin: 2px 0 6px 0;
        font-family: "Open Sans", verdana, arial, sans-serif;
      }
      .rx-row {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
      }
      .rx-btn {
        padding: 4px 10px;
        font-size: 12px;
        cursor: pointer;
        font-family: inherit;
        color: #222;
      }
      .rx-showonly,
      .rx-method-check {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #222;
      }
      .rx-th {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 12px;
        color: #222;
      }
      .rx-th input {
        width: 70px;
        font-size: 12px;
        padding: 2px 4px;
      }
      #rx-method-note {
        font-size: 11px;
        color: #7a2e2e;
        line-height: 1.25;
      }
    </style>
    <div id="rx-step-root">
      <div class="rx-row">
        <button id="rx-prev" type="button" class="rx-btn">&#8592; Prev frequency</button>
        <button id="rx-next" type="button" class="rx-btn">Next frequency &#8594;</button>
        <button id="rx-clear" type="button" class="rx-btn">Clear list</button>
        <button id="rx-csv" type="button" class="rx-btn">Download selected CSV</button>
        <label class="rx-showonly">
          <input id="rx-showonly" type="checkbox" />
          <span>Show only selected sweeps</span>
        </label>
      </div>
      <div class="rx-row">
        <label class="rx-method-check">
          <input id="rx-method-energinet" type="checkbox" />
          <span id="rx-method-energinet-label">Energinet</span>
        </label>
        <label class="rx-th">T2 <input id="rx-t2" type="number" min="1" step="1" /></label>
        <label class="rx-th">T3 <input id="rx-t3" type="number" min="1" step="1" /></label>
        <label class="rx-th">T4 <input id="rx-t4" type="number" min="1" step="1" /></label>
        <label class="rx-th">Top N <input id="rx-en-topn" type="number" min="0" step="1" value="0" /></label>
      </div>
      <div class="rx-row">
        <label class="rx-method-check">
          <input id="rx-method-iec" type="checkbox" />
          <span id="rx-method-iec-label">IEC</span>
        </label>
        <label class="rx-th">Top N <input id="rx-iec-topn" type="number" min="0" step="1" value="0" /></label>
        <label class="rx-method-check">
          <input id="rx-iec-collinear" type="checkbox" />
          <span>Include collinear boundary</span>
        </label>
      </div>
      <div id="rx-method-note"></div>
    </div>
  `;

  const prev = document.getElementById("rx-prev");
  const next = document.getElementById("rx-next");
  const showOnly = document.getElementById("rx-showonly");
  const clearBtn = document.getElementById("rx-clear");
  const csvBtn = document.getElementById("rx-csv");
  const energinetCb = document.getElementById("rx-method-energinet");
  const iecCb = document.getElementById("rx-method-iec");
  const iecCollinearCb = document.getElementById("rx-iec-collinear");
  const t2 = document.getElementById("rx-t2");
  const t3 = document.getElementById("rx-t3");
  const t4 = document.getElementById("rx-t4");
  const enTopN = document.getElementById("rx-en-topn");
  const iecTopN = document.getElementById("rx-iec-topn");

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
  if (energinetCb) {
    energinetCb.addEventListener("change", () => {
      try {
        const api = getSelectionApi(stateKey);
        if (api && typeof api.setEnerginetEnabled === "function") {
          api.setEnerginetEnabled(Boolean(energinetCb.checked));
          syncSelectionControls(stateKey);
        }
      } catch (e) {}
    });
  }
  if (iecCb) {
    iecCb.addEventListener("change", () => {
      try {
        const api = getSelectionApi(stateKey);
        if (api && typeof api.setIecEnabled === "function") {
          api.setIecEnabled(Boolean(iecCb.checked));
          syncSelectionControls(stateKey);
        }
      } catch (e) {}
    });
  }
  if (iecCollinearCb) {
    iecCollinearCb.addEventListener("change", () => {
      pushIecCollinearMode(stateKey, Boolean(iecCollinearCb.checked));
    });
  }

  const thresholdInputs = [t2, t3, t4].filter(Boolean);
  for (const inp of thresholdInputs) {
    inp.addEventListener("input", () => {
      try {
        if (thresholdDebounceTimer) clearTimeout(thresholdDebounceTimer);
        thresholdDebounceTimer = setTimeout(() => {
          pushEnerginetThresholds(stateKey);
        }, 220);
      } catch (e) {}
    });
    inp.addEventListener("change", () => {
      pushEnerginetThresholds(stateKey);
    });
  }

  const topNInputs = [
    { el: enTopN, push: () => pushEnerginetTopN(stateKey) },
    { el: iecTopN, push: () => pushIecTopN(stateKey) },
  ].filter((row) => Boolean(row.el));
  for (const row of topNInputs) {
    const el = row.el;
    el.addEventListener("input", () => {
      try {
        if (thresholdDebounceTimer) clearTimeout(thresholdDebounceTimer);
        thresholdDebounceTimer = setTimeout(() => {
          row.push();
        }, 220);
      } catch (e) {}
    });
    el.addEventListener("change", () => {
      row.push();
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
  sendToStreamlit("streamlit:setFrameHeight", { height: 190 });
}

window.addEventListener("message", (event) => {
  const data = event && event.data ? event.data : null;
  if (!data || data.type !== "streamlit:render") return;
  latestArgs = data.args || {};
  render();
});

sendToStreamlit("streamlit:componentReady", { apiVersion: 1 });
sendToStreamlit("streamlit:setFrameHeight", { height: 190 });
