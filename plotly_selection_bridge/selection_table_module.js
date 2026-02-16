// Selection-table and selection-API module.
// This file is loaded before listener.js and provides shared helpers
// used by the panel renderer and scatter toolbar bridge.

function tokenizeImportList(rawText) {
  const txt = String(rawText || "").trim();
  if (!txt) return [];
  const parts = txt.split(/[\s,;]+/g).map((x) => x.trim()).filter((x) => x.length > 0);
  const seen = new Set();
  const out = [];
  for (const p of parts) {
    const k = p.toLowerCase();
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(p);
  }
  return out;
}

function selectedRowsForTable(ctx, allowedSet) {
  const rows = [];
  for (const caseId of Array.from(ctx.state.selectedCases)) {
    const item = ctx.caseById.get(caseId);
    const displayCase = item ? String(item.display_case || caseId) : caseId;
    rows.push({
      caseId,
      displayCase,
      hidden: !allowedSet.has(caseId),
    });
  }
  rows.sort((a, b) => a.displayCase.localeCompare(b.displayCase));
  return rows;
}

function downloadSelectedCsv(rows) {
  const lines = ["case_id,display_case,status"];
  for (const r of rows) {
    const status = r.hidden ? "hidden_by_case_parts" : "visible";
    const caseIdEsc = String(r.caseId).split('"').join('""');
    const dispEsc = String(r.displayCase).split('"').join('""');
    lines.push(
      `"${caseIdEsc}","${dispEsc}","${status}"`,
    );
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = url;
    a.download = "selected_cases.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch (e) {}
  try {
    URL.revokeObjectURL(url);
  } catch (e) {}
}

function getApiStore() {
  const root = rootWindow();
  if (!root.__fsCaseUiApi || typeof root.__fsCaseUiApi !== "object") {
    root.__fsCaseUiApi = {};
  }
  return root.__fsCaseUiApi;
}

function applyStateToPlots() {
  try {
    if (typeof scheduleApplyPasses === "function") {
      scheduleApplyPasses();
      return;
    }
  } catch (e) {}
  try {
    applyAllPlots();
  } catch (e) {}
}

function registerExternalSelectionApi(ctx) {
  if (!ctx) return;
  const key = `${ctx.dataId}|${ctx.chartId}`;
  const store = getApiStore();
  pruneStoreByDataId(store, ctx.dataId);
  store[key] = {
    getState: () => {
      const nextCtx = ensureContext();
      if (!nextCtx) return null;
      const st = nextCtx.state || {};
      const energinetRawSet = st.methodEnerginetCasesRaw instanceof Set ? st.methodEnerginetCasesRaw : new Set();
      const iecRawSet = st.methodIecCasesRaw instanceof Set ? st.methodIecCasesRaw : new Set();
      const allowed = computeAllowedSet(nextCtx);
      const countVisible = (srcSet) => {
        let n = 0;
        for (const cid of srcSet) {
          if (allowed.has(cid)) n += 1;
        }
        return n;
      };
      const energinetVisible = countVisible(energinetRawSet);
      const iecVisible = countVisible(iecRawSet);
      const pre = getPreselectionPayload();
      return {
        showOnlySelected: st.showOnlySelected === true,
        selectedCount: Number(st.selectedCases instanceof Set ? st.selectedCases.size : 0),
        preselectionAvailable: Boolean(isPreselectionAvailable(nextCtx)),
        preselectionError: String(pre && pre.error ? pre.error : ""),
        limitationNote: String(pre && pre.limitation_note ? pre.limitation_note : ""),
        energinetEnabled: Boolean(st.methodEnerginetEnabled),
        iecEnabled: Boolean(st.methodIecEnabled),
        iecIncludeCollinear: Boolean(st.methodIecIncludeCollinear),
        energinetT2: Number(st.energinetT2 || 0),
        energinetT3: Number(st.energinetT3 || 0),
        energinetT4: Number(st.energinetT4 || 0),
        energinetTopN: Number(st.methodEnerginetTopN || 0),
        iecTopN: Number(st.methodIecTopN || 0),
        energinetCandidateCount: Number(energinetVisible),
        iecCandidateCount: Number(iecVisible),
        energinetCandidateCountTotal: Number(energinetRawSet.size),
        iecCandidateCountTotal: Number(iecRawSet.size),
      };
    },
    setShowOnlySelected: (flag) => {
      const nextCtx = ensureContext();
      if (!nextCtx) return false;
      nextCtx.state.showOnlySelected = flag === true;
      nextCtx.state.importStatus = "";
      recomputeSelectedCases(nextCtx);
      bumpStateVersion(nextCtx.state);
      renderPanel();
      applyStateToPlots();
      return true;
    },
    setEnerginetEnabled: (flag) => {
      const nextCtx = ensureContext();
      if (!nextCtx) return false;
      nextCtx.state.methodEnerginetEnabled = Boolean(flag);
      if (!nextCtx.state.methodEnerginetEnabled && !nextCtx.state.methodIecEnabled) {
        nextCtx.state.methodExcludedCases = new Set();
      }
      recomputeSelectedCases(nextCtx);
      nextCtx.state.importStatus = "";
      bumpStateVersion(nextCtx.state);
      renderPanel();
      applyStateToPlots();
      return true;
    },
    setIecEnabled: (flag) => {
      const nextCtx = ensureContext();
      if (!nextCtx) return false;
      nextCtx.state.methodIecEnabled = Boolean(flag);
      if (!nextCtx.state.methodIecEnabled) {
        nextCtx.state.methodIecIncludeCollinear = false;
      }
      if (!nextCtx.state.methodEnerginetEnabled && !nextCtx.state.methodIecEnabled) {
        nextCtx.state.methodExcludedCases = new Set();
      }
      recomputeSelectedCases(nextCtx);
      nextCtx.state.importStatus = "";
      bumpStateVersion(nextCtx.state);
      renderPanel();
      applyStateToPlots();
      return true;
    },
    setEnerginetThresholds: (thresholds) => {
      const nextCtx = ensureContext();
      if (!nextCtx) return false;
      const th = thresholds && typeof thresholds === "object" ? thresholds : {};
      const d = preselectionThresholdDefaults();
      const t2 = Number(th.t2 != null ? th.t2 : nextCtx.state.energinetT2);
      const t3 = Number(th.t3 != null ? th.t3 : nextCtx.state.energinetT3);
      const t4 = Number(th.t4 != null ? th.t4 : nextCtx.state.energinetT4);
      nextCtx.state.energinetT2 = Number.isFinite(t2) && t2 > 0 ? t2 : Number(d.t2);
      nextCtx.state.energinetT3 = Number.isFinite(t3) && t3 > 0 ? t3 : Number(d.t3);
      nextCtx.state.energinetT4 = Number.isFinite(t4) && t4 > 0 ? t4 : Number(d.t4);
      recomputeSelectedCases(nextCtx);
      nextCtx.state.importStatus = "";
      bumpStateVersion(nextCtx.state);
      renderPanel();
      applyStateToPlots();
      return true;
    },
    setEnerginetTopN: (rawVal) => {
      const nextCtx = ensureContext();
      if (!nextCtx) return false;
      nextCtx.state.methodEnerginetTopN = normalizeTopN(rawVal);
      recomputeSelectedCases(nextCtx);
      nextCtx.state.importStatus = "";
      bumpStateVersion(nextCtx.state);
      renderPanel();
      applyStateToPlots();
      return true;
    },
    setIecTopN: (rawVal) => {
      const nextCtx = ensureContext();
      if (!nextCtx) return false;
      nextCtx.state.methodIecTopN = normalizeTopN(rawVal);
      recomputeSelectedCases(nextCtx);
      nextCtx.state.importStatus = "";
      bumpStateVersion(nextCtx.state);
      renderPanel();
      applyStateToPlots();
      return true;
    },
    setIecIncludeCollinear: (flag) => {
      const nextCtx = ensureContext();
      if (!nextCtx) return false;
      if (!nextCtx.state.methodIecEnabled) {
        nextCtx.state.methodIecIncludeCollinear = false;
      } else {
        nextCtx.state.methodIecIncludeCollinear = flag === true;
      }
      recomputeSelectedCases(nextCtx);
      nextCtx.state.importStatus = "";
      bumpStateVersion(nextCtx.state);
      renderPanel();
      applyStateToPlots();
      return true;
    },
    clearSelection: () => {
      const nextCtx = ensureContext();
      if (!nextCtx) return false;
      nextCtx.state.manualSelectedCases = new Set();
      nextCtx.state.methodExcludedCases = new Set();
      nextCtx.state.methodEnerginetEnabled = false;
      nextCtx.state.methodIecEnabled = false;
      nextCtx.state.methodIecIncludeCollinear = false;
      recomputeSelectedCases(nextCtx);
      nextCtx.state.importStatus = "";
      bumpStateVersion(nextCtx.state);
      renderPanel();
      applyStateToPlots();
      return true;
    },
    downloadSelectedCsv: () => {
      const nextCtx = ensureContext();
      if (!nextCtx) return false;
      const allowed = computeAllowedSet(nextCtx);
      const rows = selectedRowsForTable(nextCtx, allowed);
      downloadSelectedCsv(rows);
      return true;
    },
  };
}
