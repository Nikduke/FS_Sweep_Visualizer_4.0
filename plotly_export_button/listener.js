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

function asNumber(name, fallback) {
  const v = Number(latestArgs && latestArgs[name]);
  return Number.isFinite(v) ? v : Number(fallback);
}

function asString(name, fallback) {
  const v = latestArgs ? latestArgs[name] : null;
  if (v == null) return String(fallback);
  return String(v);
}

function render() {
  const root = document.getElementById("app");
  if (!root) return;
  const buttonLabel = asString("button_label", "Export");
  root.innerHTML = `
    <style>
      html, body {
        margin: 0;
        padding: 0;
        overflow: hidden;
        background: transparent;
      }
      #exp-root {
        width: 100%;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: flex-start;
      }
      #exp-btn {
        width: auto;
        min-width: 84px;
        height: 20px;
        padding: 0 10px;
        font-size: 12px;
        font-family: "Open Sans", verdana, arial, sans-serif;
        font-weight: 600;
        color: #18488f;
        background: #e8f1ff;
        border: 1px solid #91b2e5;
        border-radius: 8px;
        cursor: pointer;
        white-space: nowrap;
        line-height: 1.15;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-top: 8px;
      }
      #exp-btn:hover {
        background: #dcecff;
        border-color: #7ea5e0;
      }
      #exp-plot {
        width: 1px;
        height: 1px;
        position: absolute;
        left: -99999px;
        top: -99999px;
      }
    </style>
    <div id="exp-root">
      <button id="exp-btn" type="button">${buttonLabel}</button>
      <div id="exp-plot"></div>
    </div>
  `;
  const btn = document.getElementById("exp-btn");
  if (btn) {
    btn.addEventListener("click", () => {
      doExport();
    });
  }
  sendToStreamlit("streamlit:setFrameHeight", { height: 30 });
}

async function doExport() {
  const scale = Math.max(1, Math.round(asNumber("scale", 1)));
  const plotHeight = Math.max(1, Math.round(asNumber("plot_height", 400)));
  const topMargin = Math.round(asNumber("top_margin_px", 40));
  const bottomAxis = Math.round(asNumber("bottom_axis_px", 60));
  const legendPad = Math.round(asNumber("legend_padding_px", 18));
  const legendEntryWidth = Math.max(1, Math.round(asNumber("legend_entrywidth", 120)));
  const plotIndex = Math.max(0, Math.round(asNumber("plot_index", 0)));
  const filename = asString("filename", "plot.png");
  const fallbackLegendFontSize = Math.max(8, Math.round(asNumber("fallback_legend_font_size", 14)));
  const legendFontFamily = asString("legend_font_family", "Open Sans, verdana, arial, sans-serif");
  const legendFontColor = asString("legend_font_color", "#000000");
  const leftMarginBase = Math.max(1, Math.round(asNumber("left_margin_px", 60)));
  const rightMargin = Math.max(0, Math.round(asNumber("right_margin_px", 20)));
  const tickFontSize = Math.max(1, Math.round(asNumber("tick_font_size_px", 14)));
  const axisTitleFontSize = Math.max(1, Math.round(asNumber("axis_title_font_size_px", 16)));
  const leftMarginTickMult = asNumber("left_margin_tick_mult", 4.4);
  const leftMarginTitleMult = asNumber("left_margin_title_mult", 1.6);
  const rowHeightFactor = asNumber("export_legend_row_height_factor", 1.25);
  const sampleLineMinPx = Math.max(1, Math.round(asNumber("export_sample_line_min_px", 18)));
  const sampleLineMult = asNumber("export_sample_line_mult", 1.8);
  const sampleGapMinPx = Math.max(1, Math.round(asNumber("export_sample_gap_min_px", 6)));
  const sampleGapMult = asNumber("export_sample_gap_mult", 0.6);
  const textPadMinPx = Math.max(1, Math.round(asNumber("export_text_pad_min_px", 8)));
  const textPadMult = asNumber("export_text_pad_mult", 0.8);
  const tailFontMult = asNumber("export_legend_tail_font_mult", 0.35);
  const rowYOffset = asNumber("export_legend_row_y_offset", 0.6);
  const colPaddingMaxPx = Math.max(0, Math.round(asNumber("export_col_padding_max_px", 12)));
  const colPaddingFrac = asNumber("export_col_padding_frac", 0.06);
  const fallbackColor = asString("export_fallback_color", "#444");

  try {
    const Plotly = window.parent && window.parent.Plotly ? window.parent.Plotly : null;
    if (!Plotly) return;
    const plots = window.parent && window.parent.document ? window.parent.document.querySelectorAll("div.js-plotly-plot") : null;
    if (!plots || plots.length <= plotIndex) return;
    const gd = plots[plotIndex];
    if (!gd) return;

    const r = gd.getBoundingClientRect();
    const widthPx = Math.floor(r.width || 0);
    if (!widthPx) return;

    const legendFontSize = (
      (gd && gd._fullLayout && gd._fullLayout.legend && gd._fullLayout.legend.font && gd._fullLayout.legend.font.size) ||
      (gd && gd._fullLayout && gd._fullLayout.font && gd._fullLayout.font.size) ||
      fallbackLegendFontSize
    );
    const legendRowH = Math.ceil(Number(legendFontSize) * rowHeightFactor);
    const leftMarginPx = Math.max(
      leftMarginBase,
      Math.round(tickFontSize * leftMarginTickMult + axisTitleFontSize * leftMarginTitleMult),
    );

    const data = Array.isArray(gd.data) ? gd.data : [];
    const data2 = data.map((tr) => {
      const t = Object.assign({}, tr);
      if (t.type === "scattergl") t.type = "scatter";
      return t;
    });

    const legendItems = [];
    for (const tr of data2) {
      if (tr && tr.showlegend === false) continue;
      const name = tr && tr.name ? String(tr.name) : "";
      if (!name) continue;
      const color =
        (tr.line && tr.line.color) ? tr.line.color :
        (tr.marker && tr.marker.color) ? tr.marker.color :
        (tr.meta && tr.meta.legend_color) ? tr.meta.legend_color :
        fallbackColor;
      legendItems.push({ name, color });
    }

    const usableW = Math.max(1, widthPx - leftMarginPx - rightMargin);
    let maxTextW = 0;
    try {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.font = `${legendFontSize}px ${legendFontFamily}`;
        for (const it of legendItems) {
          const w = ctx.measureText(it.name).width || 0;
          if (w > maxTextW) maxTextW = w;
        }
      }
    } catch (e) {}

    const sampleLinePx = Math.max(sampleLineMinPx, Math.round(sampleLineMult * legendFontSize));
    const sampleGapPx = Math.max(sampleGapMinPx, Math.round(sampleGapMult * legendFontSize));
    const textPadPx = Math.max(textPadMinPx, Math.round(textPadMult * legendFontSize));
    const neededEntryPx = Math.ceil(sampleLinePx + sampleGapPx + maxTextW + textPadPx);
    const entryPx = Math.max(1, Math.max(legendEntryWidth, neededEntryPx));

    const cols = Math.max(1, Math.floor(usableW / entryPx));
    const rows = Math.ceil(legendItems.length / cols);
    const legendH = (rows * legendRowH) + legendPad + Math.ceil(tailFontMult * legendFontSize);
    const newHeight = plotHeight + topMargin + bottomAxis + legendH;
    const newMarginB = bottomAxis + legendH;

    const container = document.getElementById("exp-plot");
    if (!container) return;
    container.style.width = `${widthPx}px`;
    container.style.height = `${newHeight}px`;

    const baseLayout = Object.assign({}, gd.layout || {});
    baseLayout.width = widthPx;
    baseLayout.height = newHeight;
    baseLayout.autosize = false;
    baseLayout.margin = Object.assign({}, baseLayout.margin || {});
    baseLayout.margin.t = topMargin;
    baseLayout.margin.l = leftMarginPx;
    baseLayout.margin.r = rightMargin;
    baseLayout.margin.b = newMarginB;
    baseLayout.showlegend = false;
    for (const tr of data2) {
      tr.showlegend = false;
    }

    const ann = Array.isArray(baseLayout.annotations) ? baseLayout.annotations.slice() : [];
    const shp = Array.isArray(baseLayout.shapes) ? baseLayout.shapes.slice() : [];
    const colW = usableW / cols;
    const xPadPx = Math.max(0, Math.min(colPaddingMaxPx, Math.floor(colW * colPaddingFrac)));

    for (let i = 0; i < legendItems.length; i++) {
      const row = Math.floor(i / cols);
      const col = i % cols;
      const x0 = (col * colW + xPadPx) / usableW;
      const x1 = x0 + (sampleLinePx / usableW);
      const y = -(bottomAxis + legendPad + (row + rowYOffset) * legendRowH) / Math.max(1, plotHeight);
      shp.push({
        type: "line",
        xref: "paper",
        yref: "paper",
        x0, x1,
        y0: y, y1: y,
        line: { color: legendItems[i].color, width: 2 },
      });
      ann.push({
        xref: "paper",
        yref: "paper",
        x: x1 + (sampleGapPx / usableW),
        y,
        xanchor: "left",
        yanchor: "middle",
        showarrow: false,
        align: "left",
        text: legendItems[i].name,
        font: { size: legendFontSize, family: legendFontFamily, color: legendFontColor },
      });
    }

    baseLayout.annotations = ann;
    baseLayout.shapes = shp;

    await Plotly.newPlot(container, data2, baseLayout, { displayModeBar: false, staticPlot: true });
    const url = await Plotly.toImage(container, {
      format: "png",
      width: widthPx,
      height: newHeight,
      scale,
    });
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch (e) {
  } finally {
    try {
      const container = document.getElementById("exp-plot");
      if (container) {
        container.innerHTML = "";
        container.style.width = "1px";
        container.style.height = "1px";
      }
    } catch (e) {}
  }
}

window.addEventListener("message", (event) => {
  const data = event && event.data ? event.data : null;
  if (!data || data.type !== "streamlit:render") return;
  latestArgs = data.args || {};
  render();
});

sendToStreamlit("streamlit:componentReady", { apiVersion: 1 });
sendToStreamlit("streamlit:setFrameHeight", { height: 30 });
