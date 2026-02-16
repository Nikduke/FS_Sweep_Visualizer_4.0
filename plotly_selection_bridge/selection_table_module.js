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

function registerExternalSelectionApi(ctx) {
  if (!ctx) return;
  const key = `${ctx.dataId}|${ctx.chartId}`;
  const store = getApiStore();
  pruneStoreByDataId(store, ctx.dataId);
  store[key] = {
    getState: () => {
      const nextCtx = ensureContext();
      if (!nextCtx) return null;
      return {
        showOnlySelected: Boolean(nextCtx.state.showOnlySelected),
        selectedCount: Number(nextCtx.state.selectedCases.size || 0),
      };
    },
    setShowOnlySelected: (flag) => {
      const nextCtx = ensureContext();
      if (!nextCtx) return false;
      nextCtx.state.showOnlySelected = Boolean(flag);
      nextCtx.state.importStatus = "";
      bumpStateVersion(nextCtx.state);
      renderPanel();
      applyAllPlots();
      return true;
    },
    clearSelection: () => {
      const nextCtx = ensureContext();
      if (!nextCtx) return false;
      nextCtx.state.selectedCases = new Set();
      nextCtx.state.importStatus = "";
      bumpStateVersion(nextCtx.state);
      renderPanel();
      applyAllPlots();
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
