from __future__ import annotations


def render_dashboard_html() -> str:
    return _render_shell(
        title="OCTTS Dashboard",
        page_title="OCTTS 智能趋势总览",
        page_subtitle="总览页聚合股票池信号、历史命中状态，并支持手动输入股票代码后立即触发一次分析。",
        content="""
      <section class="layout">
        <div class="main-column">
          <section class="summary-grid" id="summaryGrid"></section>
          <section class="cards" id="cards"></section>
        </div>
        <aside class="side-column">
          <section class="panel">
            <div class="section-title">手动分析</div>
            <form id="manualAnalyzeForm" class="stack">
              <label class="field">
                <span class="mini-label">股票代码</span>
                <input id="stockCodeInput" name="stock_code" type="text" placeholder="例如 600000.SH" autocomplete="off" />
              </label>
              <label class="field">
                <span class="mini-label">分析阶段</span>
                <select id="phaseSelect" name="phase">
                  <option value="morning">早盘</option>
                  <option value="afternoon">尾盘</option>
                  <option value="review" selected>复盘</option>
                </select>
              </label>
              <label class="checkbox-field">
                <input id="notifyToggle" name="notify" type="checkbox" checked />
                <span>分析完成后发送企业微信通知</span>
              </label>
              <label class="checkbox-field">
                <input id="persistToggle" name="persist" type="checkbox" />
                <span>把这只股票加入默认股票池</span>
              </label>
              <button id="manualAnalyzeButton" class="primary-button" type="submit">立即分析</button>
              <button id="defaultPoolAnalyzeButton" class="primary-button secondary-button" type="button">一键分析默认股票池</button>
              <div class="subtle">输入完整 Tushare 代码后即可发起单次分析，例如 `600000.SH` 或 `000001.SZ`。</div>
              <div id="manualAnalyzeStatus" class="subtle"></div>
            </form>
          </section>
          <section class="panel">
            <div class="section-title">默认股票池</div>
            <div id="defaultStockPool" class="stack"></div>
            <div id="defaultStockPoolStatus" class="subtle"></div>
          </section>
          <section class="panel automation-panel">
            <div class="section-title">OpenClaw 自动化</div>
            <div id="openclawStatus" class="stack"></div>
          </section>
          <section class="panel">
            <div class="section-title">数据管理</div>
            <div class="stack">
              <button id="clearAllDataButton" class="danger-button" type="button">清空全部分析结果</button>
              <div id="dataControlStatus" class="subtle">同一交易日同一阶段的重复分析会自动覆盖，不再重复累计。</div>
            </div>
          </section>
        </aside>
      </section>
    """,
        script=_overview_script(),
    )


def render_stock_detail_html(ts_code: str) -> str:
    return _render_shell(
        title=f"OCTTS {ts_code}",
        page_title=f"{ts_code} 单股详情",
        page_subtitle="单股详情页展示该股票最新结构化决策、验证状态、关键价位和完整历史时间线。",
        content=f"""
      <section class="detail-shell">
        <div class="toolbar">
          <a class="ghost-link" href="/dashboard">返回总览</a>
          <div class="row">
            <button id="clearSymbolDataButton" class="danger-button" type="button">清空这只股票的分析结果</button>
            <div id="detailGeneratedAt" class="subtle">等待加载数据...</div>
          </div>
        </div>
        <section class="layout detail-layout">
          <div class="main-column">
            <section class="hero-card" id="heroCard"></section>
            <section class="chart-and-summary">
              <div class="panel">
                <div class="section-title">Price Context</div>
                <div id="detailSparkline"></div>
              </div>
              <div class="panel">
                <div class="section-title">Validation Snapshot</div>
                <div id="validationSummary" class="summary-grid compact"></div>
              </div>
            </section>
            <section class="metrics-grid" id="detailMetrics"></section>
            <section class="panel">
              <div class="section-title">History Replay</div>
              <div id="detailHistory" class="timeline"></div>
            </section>
          </div>
          <aside class="side-column">
            <section class="panel automation-panel">
              <div class="section-title">OpenClaw 自动化</div>
              <div id="detailOpenclawStatus" class="stack"></div>
            </section>
            <section class="panel">
              <div class="section-title">决策依据</div>
              <div id="detailEvidence"></div>
            </section>
            <section class="panel">
              <div class="section-title">风险关注</div>
              <div id="detailRisks"></div>
            </section>
          </aside>
        </section>
      </section>
    """,
        script=_detail_script(ts_code),
    )


def _render_shell(*, title: str, page_title: str, page_subtitle: str, content: str, script: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #06101d;
      --panel: rgba(12, 23, 42, 0.82);
      --panel-strong: rgba(7, 18, 35, 0.94);
      --panel-border: rgba(148, 163, 184, 0.16);
      --text: #e5eefc;
      --muted: #93a4bf;
      --accent: #60a5fa;
      --good: #34d399;
      --bad: #f87171;
      --warn: #fbbf24;
      --shadow: 0 18px 60px rgba(2, 6, 23, 0.38);
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.25), transparent 35%),
        radial-gradient(circle at 80% 0%, rgba(16, 185, 129, 0.12), transparent 25%),
        linear-gradient(180deg, #030712 0%, #07111f 100%);
      color: var(--text);
    }}

    a {{
      color: inherit;
      text-decoration: none;
    }}

    .page {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px 24px 48px;
    }}

    .hero {{
      display: grid;
      gap: 16px;
      padding: 28px;
      margin-bottom: 18px;
      background: linear-gradient(145deg, rgba(10, 22, 43, 0.92), rgba(10, 18, 34, 0.82));
      border: 1px solid var(--panel-border);
      box-shadow: var(--shadow);
      border-radius: 24px;
      backdrop-filter: blur(14px);
    }}

    .hero h1 {{
      margin: 0;
      font-size: 30px;
      font-weight: 800;
      letter-spacing: -0.02em;
    }}

    .hero p, .subtle {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }}

    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 18px;
      align-items: start;
    }}

    .detail-layout {{
      grid-template-columns: minmax(0, 1fr) 340px;
    }}

    .main-column, .side-column {{
      display: grid;
      gap: 18px;
    }}

    .panel, .summary, .card, .hero-card {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      backdrop-filter: blur(14px);
      box-shadow: var(--shadow);
      border-radius: 22px;
    }}

    .panel {{
      padding: 20px;
    }}

    .field {{
      display: grid;
      gap: 8px;
    }}

    input, select, button {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      background: rgba(15, 23, 42, 0.72);
      color: var(--text);
      padding: 12px 14px;
      font: inherit;
    }}

    input::placeholder {{
      color: rgba(147, 164, 191, 0.7);
    }}

    .checkbox-field {{
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
    }}

    .checkbox-field input {{
      width: 16px;
      height: 16px;
      padding: 0;
      accent-color: var(--accent);
    }}

    .primary-button {{
      cursor: pointer;
      font-weight: 700;
      background: linear-gradient(135deg, rgba(37, 99, 235, 0.95), rgba(59, 130, 246, 0.88));
      border-color: rgba(96, 165, 250, 0.35);
      transition: transform 180ms ease, opacity 180ms ease;
    }}

    .primary-button:hover {{
      transform: translateY(-1px);
    }}

    .secondary-button {{
      background: rgba(15, 23, 42, 0.72);
      border-color: rgba(96, 165, 250, 0.25);
    }}

    .danger-button {{
      cursor: pointer;
      font-weight: 700;
      background: rgba(127, 29, 29, 0.28);
      border-color: rgba(248, 113, 113, 0.3);
      color: #fecaca;
    }}

    .primary-button:disabled {{
      cursor: wait;
      opacity: 0.7;
      transform: none;
    }}

    .chip-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.72);
      border: 1px solid rgba(148, 163, 184, 0.18);
    }}

    .chip-action {{
      width: auto;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid rgba(248, 113, 113, 0.22);
      background: rgba(127, 29, 29, 0.18);
      color: #fecaca;
      cursor: pointer;
    }}

    .hero-card {{
      padding: 24px;
      display: grid;
      gap: 18px;
      background: linear-gradient(145deg, rgba(9, 20, 38, 0.96), rgba(10, 18, 32, 0.88));
    }}

    .toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 18px;
    }}

    .toolbar button {{
      width: auto;
    }}

    .ghost-link {{
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      color: var(--muted);
      background: rgba(15, 23, 42, 0.38);
    }}

    .section-title {{
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 12px;
    }}

    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
    }}

    .summary-grid.compact {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}

    .summary {{
      padding: 18px;
    }}

    .summary .label, .mini-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .summary .value {{
      margin-top: 10px;
      font-size: 28px;
      font-weight: 800;
    }}

    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
    }}

    .card {{
      padding: 22px;
      display: grid;
      gap: 16px;
      transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }}

    .card:hover {{
      transform: translateY(-2px);
      border-color: rgba(96, 165, 250, 0.35);
      box-shadow: 0 24px 70px rgba(30, 41, 59, 0.45);
    }}

    .card-link {{
      display: contents;
    }}

    .row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}

    .stack {{
      display: grid;
      gap: 12px;
    }}

    .title {{
      font-size: 22px;
      font-weight: 800;
    }}

    .big-number {{
      font-size: 42px;
      font-weight: 800;
      letter-spacing: -0.03em;
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 7px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid rgba(255,255,255,0.12);
      text-transform: uppercase;
    }}

    .badge.buy, .badge.take_profit_hit {{ background: rgba(52, 211, 153, 0.14); color: var(--good); }}
    .badge.sell, .badge.stop_loss_hit {{ background: rgba(248, 113, 113, 0.14); color: var(--bad); }}
    .badge.hold, .badge.entered, .badge.connected {{ background: rgba(96, 165, 250, 0.14); color: var(--accent); }}
    .badge.reduce, .badge.watching_entry, .badge.expired, .badge.avoid, .badge.no_signal, .badge.disconnected {{ background: rgba(251, 191, 36, 0.14); color: var(--warn); }}

    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }}

    .metric {{
      padding: 16px;
      background: rgba(15, 23, 42, 0.56);
      border-radius: 18px;
      border: 1px solid rgba(148, 163, 184, 0.12);
      min-height: 110px;
    }}

    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .metric .value {{
      margin-top: 10px;
      font-size: 18px;
      font-weight: 700;
      line-height: 1.45;
      white-space: pre-wrap;
    }}

    .chart-and-summary {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(300px, 0.8fr);
      gap: 18px;
    }}

    .sparkline {{
      width: 100%;
      height: 86px;
      display: block;
      background: linear-gradient(180deg, rgba(96,165,250,0.12), rgba(96,165,250,0.03));
      border-radius: 16px;
    }}

    .timeline {{
      display: grid;
      gap: 12px;
    }}

    .timeline-item, .automation-item {{
      padding: 14px;
      background: rgba(15, 23, 42, 0.54);
      border-radius: 16px;
      border: 1px solid rgba(148, 163, 184, 0.1);
    }}

    .list {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.65;
    }}

    .empty {{
      padding: 24px;
      color: var(--muted);
      text-align: center;
      border: 1px dashed rgba(148, 163, 184, 0.2);
      border-radius: 18px;
    }}

    @media (max-width: 1100px) {{
      .layout, .detail-layout, .chart-and-summary {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{page_title}</h1>
      <p>{page_subtitle}</p>
      <div class="subtle" id="generatedAt">等待加载数据...</div>
    </section>
    {content}
  </div>
  <script>
    function formatValue(value, digits = 2) {{
      if (value === null || value === undefined || value === "") return "—";
      if (typeof value === "number") return value.toFixed(digits);
      return String(value);
    }}

    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function sparkline(records) {{
      const values = (records || [])
        .map(item => Number(item.close ?? item.close_price ?? item.price))
        .filter(value => Number.isFinite(value));
      if (!values.length) return '<div class="empty">暂无分时样本</div>';
      const max = Math.max(...values);
      const min = Math.min(...values);
      const points = values.map((value, index) => {{
        const x = values.length === 1 ? 200 : (index / (values.length - 1)) * 200;
        const y = max === min ? 40 : 72 - ((value - min) / (max - min)) * 58;
        return `${{x}},${{y}}`;
      }}).join(" ");
      return `<svg class="sparkline" viewBox="0 0 200 90" preserveAspectRatio="none">
        <polyline fill="none" stroke="#60a5fa" stroke-width="3" points="${{points}}" />
      </svg>`;
    }}

    function renderSummaryCard(label, value) {{
      return `<div class="summary"><div class="label">${{label}}</div><div class="value">${{value}}</div></div>`;
    }}

    function renderList(items) {{
      if (!items || !items.length) return '<div class="empty">暂无数据</div>';
      return `<ul class="list">${{items.map(item => `<li>${{escapeHtml(item)}}</li>`).join("")}}</ul>`;
    }}

    const PHASE_LABELS = {{
      morning: "早盘",
      afternoon: "尾盘",
      review: "复盘"
    }};

    const TREND_BIAS_LABELS = {{
      bullish: "看多",
      neutral: "中性",
      bearish: "看空"
    }};

    const SIGNAL_LABELS = {{
      buy: "买入",
      hold: "持有",
      reduce: "减仓",
      sell: "卖出",
      avoid: "观望"
    }};

    const VALIDATION_STATUS_LABELS = {{
      no_signal: "无交易信号",
      watching_entry: "等待入场",
      entered: "已进入区间",
      take_profit_hit: "已触发止盈",
      stop_loss_hit: "已触发止损",
      expired: "观点已过期"
    }};

    const PREVIOUS_VIEW_STATUS_LABELS = {{
      confirmed: "延续确认",
      weakened: "有所减弱",
      reversed: "观点反转",
      initial: "首次分析"
    }};

    const HOLDING_HORIZON_LABELS = {{
      intraday: "日内",
      swing: "波段",
      position: "中线"
    }};

    const PREDICTION_WINDOW_LABELS = {{
      next_1d: "未来1个交易日",
      next_3d: "未来3个交易日",
      next_5d: "未来5个交易日"
    }};

    function labelFor(mapping, value) {{
      if (value === null || value === undefined || value === "") return "—";
      return mapping[value] || String(value);
    }}

    function formatPhase(value) {{
      return labelFor(PHASE_LABELS, value);
    }}

    function formatTrendBias(value) {{
      return labelFor(TREND_BIAS_LABELS, value);
    }}

    function formatSignal(value) {{
      return labelFor(SIGNAL_LABELS, value);
    }}

    function formatValidationStatus(value) {{
      return labelFor(VALIDATION_STATUS_LABELS, value);
    }}

    function formatPreviousViewStatus(value) {{
      return labelFor(PREVIOUS_VIEW_STATUS_LABELS, value);
    }}

    function formatHoldingHorizon(value) {{
      return labelFor(HOLDING_HORIZON_LABELS, value);
    }}

    function formatPredictionWindow(value) {{
      return labelFor(PREDICTION_WINDOW_LABELS, value);
    }}

    function renderAutomationStatus(status) {{
      if (!status) return '<div class="empty">OpenClaw 状态暂不可用</div>';
      const connectivity = status.connected ? "connected" : "disconnected";
      return `
        <div class="automation-item">
          <div class="row">
            <strong>自动化状态</strong>
            <span class="badge ${{connectivity}}">${{status.automation_enabled ? "已启用" : "外部编排"}}</span>
          </div>
          <div class="subtle" style="margin-top:8px;">${{escapeHtml(status.automation_enabled ? `时区：${{status.automation_timezone}}` : (status.gateway_url || "尚未配置 OPENCLAW_GATEWAY_URL"))}}</div>
        </div>
        <div class="automation-item">
          <div class="mini-label">Agent ID</div>
          <div class="value">${{escapeHtml(status.agent_id || "octts")}}</div>
        </div>
        <div class="automation-item">
          <div class="mini-label">通知推送</div>
          <div class="value">${{status.automation_notify ? "已启用" : "已关闭"}}</div>
        </div>
        <div class="automation-item">
          <div class="mini-label">Hooks</div>
          <div class="value">${{status.hooks_enabled ? "已启用" : "已关闭"}}</div>
        </div>
        <div class="automation-item">
          <div class="mini-label">定时任务</div>
          <div class="stack">${{(status.automation_slots || []).map(slot => `
            <div class="row">
              <span>${{escapeHtml(slot.label)}}</span>
              <span class="subtle">${{escapeHtml(slot.time)}}</span>
            </div>
          `).join("")}}</div>
        </div>
        <div class="automation-item">
          <div class="mini-label">状态说明</div>
          <div class="value">${{escapeHtml(status.status_note)}}</div>
        </div>
      `;
    }}

    function renderHistoryTimeline(history, limit = history ? history.length : 0) {{
      if (!history || !history.length) return '<div class="empty">暂无历史记录</div>';
      return history.slice().reverse().slice(0, limit || history.length).map(item => `
        <div class="timeline-item">
          <div class="row">
            <strong>${{escapeHtml(formatPhase(item.report.phase))}} · ${{escapeHtml(formatSignal(item.report.decision.signal))}}</strong>
            <span class="badge ${{escapeHtml(item.validation.status)}}">${{escapeHtml(formatValidationStatus(item.validation.status))}}</span>
          </div>
          <div class="subtle" style="margin-top:6px;">${{escapeHtml(item.generated_at)}}</div>
          <div style="margin-top:8px;">${{escapeHtml(item.validation.note)}}</div>
        </div>
      `).join("");
    }}

    {script}
  </script>
</body>
</html>"""


def _overview_script() -> str:
    return """
    const summaryGrid = document.getElementById("summaryGrid");
    const cards = document.getElementById("cards");
    const generatedAt = document.getElementById("generatedAt");
    const openclawStatus = document.getElementById("openclawStatus");
    const manualAnalyzeForm = document.getElementById("manualAnalyzeForm");
    const stockCodeInput = document.getElementById("stockCodeInput");
    const phaseSelect = document.getElementById("phaseSelect");
    const notifyToggle = document.getElementById("notifyToggle");
    const persistToggle = document.getElementById("persistToggle");
    const manualAnalyzeButton = document.getElementById("manualAnalyzeButton");
    const defaultPoolAnalyzeButton = document.getElementById("defaultPoolAnalyzeButton");
    const manualAnalyzeStatus = document.getElementById("manualAnalyzeStatus");
    const defaultStockPool = document.getElementById("defaultStockPool");
    const defaultStockPoolStatus = document.getElementById("defaultStockPoolStatus");
    const clearAllDataButton = document.getElementById("clearAllDataButton");
    const dataControlStatus = document.getElementById("dataControlStatus");

    function renderSummary(payload) {
      const items = payload.cards || [];
      const statuses = payload.validation_summary || {};
      summaryGrid.innerHTML = [
        renderSummaryCard("跟踪股票数", items.length),
        renderSummaryCard("默认股票池", (payload.default_stock_pool || []).length),
        renderSummaryCard("止盈触发", statuses.take_profit_hit || 0),
        renderSummaryCard("止损触发", statuses.stop_loss_hit || 0),
        renderSummaryCard("等待入场", statuses.watching_entry || 0)
      ].join("");
    }

    function renderDefaultStockPool(stockPool) {
      if (!stockPool || !stockPool.length) {
        defaultStockPool.innerHTML = '<div class="empty">默认股票池为空。勾选上面的选项后分析一次，即可加入长期跟踪。</div>';
        return;
      }

      defaultStockPool.innerHTML = `<div class="chip-list">${stockPool.map(tsCode => `
        <div class="chip">
          <span>${escapeHtml(tsCode)}</span>
          <button class="chip-action" type="button" data-remove-ts-code="${escapeHtml(tsCode)}">移除</button>
        </div>
      `).join("")}</div>`;
    }

    function renderCard(item) {
      const zone = item.decision.entry_zone || {};
      const trendBreakdown = item.trend_breakdown || {};
      return `<a class="card-link" href="/stocks/${encodeURIComponent(item.ts_code)}">
        <article class="card">
          <div class="row">
            <div>
              <div class="title">${escapeHtml(item.ts_code)}${item.name ? " · " + escapeHtml(item.name) : ""}</div>
              <div class="subtle">${escapeHtml(item.generated_at)} · ${escapeHtml(formatPhase(item.phase))}</div>
            </div>
            <div class="row">
              <span class="badge ${escapeHtml(item.decision.signal)}">${escapeHtml(formatSignal(item.decision.signal))}</span>
              <span class="badge ${escapeHtml(item.validation.status)}">${escapeHtml(formatValidationStatus(item.validation.status))}</span>
            </div>
          </div>
          <div class="metrics-grid">
            <div class="metric"><div class="label">趋势判断</div><div class="value">${escapeHtml(item.trend_judgement)}</div></div>
            <div class="metric"><div class="label">短 / 中 / 长趋势</div><div class="value">${escapeHtml(`${formatTrendBias(trendBreakdown.short_term)} / ${formatTrendBias(trendBreakdown.mid_term)} / ${formatTrendBias(trendBreakdown.long_term)}`)}</div></div>
            <div class="metric"><div class="label">信心分数</div><div class="value">${Math.round((item.decision.confidence_score || 0) * 100)}%</div></div>
            <div class="metric"><div class="label">入场区间</div><div class="value">${formatValue(zone.low)} - ${formatValue(zone.high)}</div></div>
            <div class="metric"><div class="label">目标位</div><div class="value">${(item.decision.take_profit || []).map(v => formatValue(v)).join(" / ") || "—"}</div></div>
          </div>
          ${sparkline(item.snapshot.minute_summary)}
          <div class="metric">
            <div class="label">操作建议</div>
            <div class="value">${escapeHtml(item.operation_advice)}</div>
          </div>
          <div class="metric">
            <div class="label">验证说明</div>
            <div class="value">${escapeHtml(item.validation.note)}</div>
          </div>
        </article>
      </a>`;
    }

    async function loadDashboard() {
      const response = await fetch("/dashboard/data");
      const payload = await response.json();
      generatedAt.textContent = payload.generated_at ? `最近更新：${payload.generated_at}` : "暂无数据，请先触发一次分析。";
      renderSummary(payload);
      renderDefaultStockPool(payload.default_stock_pool || []);
      openclawStatus.innerHTML = renderAutomationStatus(payload.openclaw_status);
      if (!payload.cards || !payload.cards.length) {
        cards.innerHTML = '<div class="empty">还没有任何分析记录。先调用一次 /analyze，再回来查看趋势卡片与详情页入口。</div>';
        return;
      }
      cards.innerHTML = payload.cards.map(renderCard).join("");
    }

    async function addToDefaultStockPool(tsCode) {
      const response = await fetch("/stock-pool", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ts_code: tsCode })
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "默认股票池更新失败");
      }
      renderDefaultStockPool(payload.stock_pool || []);
      defaultStockPoolStatus.textContent = `${tsCode} 已加入默认股票池。`;
    }

    async function removeFromDefaultStockPool(tsCode) {
      const response = await fetch(`/stock-pool/${encodeURIComponent(tsCode)}`, {
        method: "DELETE"
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "默认股票池更新失败");
      }
      renderDefaultStockPool(payload.stock_pool || []);
      defaultStockPoolStatus.textContent = `${tsCode} 已从默认股票池移除。`;
      await loadDashboard();
    }

    async function triggerManualAnalysis(event) {
      event.preventDefault();
      const tsCode = (stockCodeInput.value || "").trim().toUpperCase();
      if (!tsCode) {
        manualAnalyzeStatus.textContent = "请先输入股票代码，例如 600000.SH。";
        stockCodeInput.focus();
        return;
      }

      manualAnalyzeButton.disabled = true;
      manualAnalyzeStatus.textContent = `正在分析 ${tsCode}...`;

      try {
        const response = await fetch("/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            phase: phaseSelect.value,
            stock_pool: [tsCode],
            notify: notifyToggle.checked
          })
        });

        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "请求失败");
        }

        if (persistToggle.checked) {
          await addToDefaultStockPool(tsCode);
        }

        manualAnalyzeStatus.textContent = `${tsCode} 分析完成，正在刷新页面...`;
        await loadDashboard();
        window.location.href = `/stocks/${encodeURIComponent(tsCode)}`;
      } catch (error) {
        manualAnalyzeStatus.textContent = `触发分析失败：${error.message}`;
      } finally {
        manualAnalyzeButton.disabled = false;
      }
    }

    async function triggerDefaultPoolAnalysis() {
      defaultPoolAnalyzeButton.disabled = true;
      manualAnalyzeStatus.textContent = "正在分析默认股票池...";
      try {
        const response = await fetch("/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            phase: phaseSelect.value,
            notify: notifyToggle.checked
          })
        });

        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "请求失败");
        }

        const successCount = (payload.reports || []).length;
        const failedItems = payload.errors || [];
        if (failedItems.length) {
          const failedSymbols = failedItems.map(item => item.ts_code).join(", ");
          manualAnalyzeStatus.textContent = `默认股票池分析完成：成功 ${successCount} 只，跳过 ${failedItems.length} 只（${failedSymbols}）。`;
        } else {
          manualAnalyzeStatus.textContent = `默认股票池分析完成，共处理 ${successCount} 只股票。`;
        }
        await loadDashboard();
      } catch (error) {
        manualAnalyzeStatus.textContent = `默认股票池分析失败：${error.message}`;
      } finally {
        defaultPoolAnalyzeButton.disabled = false;
      }
    }

    async function clearAllAnalysisData() {
      const confirmed = window.confirm("这会清空所有股票的历史记录和记忆摘要，确定继续吗？");
      if (!confirmed) return;

      clearAllDataButton.disabled = true;
      dataControlStatus.textContent = "正在清空全部分析结果...";
      try {
        const response = await fetch("/analysis-data", { method: "DELETE" });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "清空失败");
        }

        dataControlStatus.textContent = `已清空 ${payload.removed_records || 0} 条历史记录和 ${payload.removed_memory_items || 0} 条记忆摘要。`;
        manualAnalyzeStatus.textContent = "历史与记忆已清空，后续分析将从全新上下文开始。";
        await loadDashboard();
      } catch (error) {
        dataControlStatus.textContent = `清空失败：${error.message}`;
      } finally {
        clearAllDataButton.disabled = false;
      }
    }

    manualAnalyzeForm.addEventListener("submit", triggerManualAnalysis);
    defaultPoolAnalyzeButton.addEventListener("click", () => {
      triggerDefaultPoolAnalysis();
    });
    clearAllDataButton.addEventListener("click", () => {
      clearAllAnalysisData();
    });
    defaultStockPool.addEventListener("click", event => {
      const button = event.target.closest("[data-remove-ts-code]");
      if (!button) return;
      const tsCode = button.dataset.removeTsCode;
      removeFromDefaultStockPool(tsCode).catch(error => {
        defaultStockPoolStatus.textContent = `移除失败：${error.message}`;
      });
    });

    loadDashboard().catch(error => {
      cards.innerHTML = `<div class="empty">加载总览失败：${escapeHtml(error.message)}</div>`;
    });
    """


def _detail_script(ts_code: str) -> str:
    return f"""
    const detailGeneratedAt = document.getElementById("detailGeneratedAt");
    const generatedAt = detailGeneratedAt;
    const heroCard = document.getElementById("heroCard");
    const detailSparkline = document.getElementById("detailSparkline");
    const validationSummary = document.getElementById("validationSummary");
    const detailMetrics = document.getElementById("detailMetrics");
    const detailHistory = document.getElementById("detailHistory");
    const detailEvidence = document.getElementById("detailEvidence");
    const detailRisks = document.getElementById("detailRisks");
    const detailOpenclawStatus = document.getElementById("detailOpenclawStatus");
    const clearSymbolDataButton = document.getElementById("clearSymbolDataButton");

    function renderHero(symbol) {{
      return `
        <div class="row">
          <div>
            <div class="title">${{escapeHtml(symbol.ts_code)}}${{symbol.name ? " · " + escapeHtml(symbol.name) : ""}}</div>
            <div class="subtle">${{escapeHtml(symbol.generated_at)}} · ${{escapeHtml(formatPhase(symbol.phase))}}</div>
          </div>
          <div class="row">
            <span class="badge ${{escapeHtml(symbol.decision.signal)}}">${{escapeHtml(formatSignal(symbol.decision.signal))}}</span>
            <span class="badge ${{escapeHtml(symbol.validation.status)}}">${{escapeHtml(formatValidationStatus(symbol.validation.status))}}</span>
          </div>
        </div>
        <div class="row">
          <div>
            <div class="mini-label">趋势判断</div>
            <div class="big-number">${{escapeHtml(symbol.trend_judgement)}}</div>
          </div>
          <div style="text-align:right;">
            <div class="mini-label">操作建议</div>
            <div class="value">${{escapeHtml(symbol.operation_advice)}}</div>
          </div>
        </div>
      `;
    }}

    function renderValidationSummary(summary) {{
      validationSummary.innerHTML = [
        renderSummaryCard("止盈触发", summary.take_profit_hit || 0),
        renderSummaryCard("止损触发", summary.stop_loss_hit || 0),
        renderSummaryCard("等待入场", summary.watching_entry || 0),
        renderSummaryCard("已进入区间", summary.entered || 0)
      ].join("");
    }}

    function renderMetrics(symbol) {{
      const zone = symbol.decision.entry_zone || {{}};
      const trendBreakdown = symbol.trend_breakdown || {{}};
      const predictionText = (symbol.prediction_windows || []).map(item => {{
        const confidence = Math.round((item.confidence_score || 0) * 100);
        return `${{formatPredictionWindow(item.window)}}：${{formatTrendBias(item.bias)}} (${{confidence}}%)`;
      }}).join("\\n") || "—";
      detailMetrics.innerHTML = [
        ["入场区间", `${{formatValue(zone.low)}} - ${{formatValue(zone.high)}}`],
        ["止损位", formatValue(symbol.decision.stop_loss)],
        ["目标位", (symbol.decision.take_profit || []).map(v => formatValue(v)).join(" / ") || "—"],
        ["持有周期", escapeHtml(formatHoldingHorizon(symbol.decision.holding_horizon))],
        ["失效条件", escapeHtml(symbol.decision.invalidation_condition)],
        ["历史观点状态", escapeHtml(formatPreviousViewStatus(symbol.previous_view_status))],
        ["短线趋势", escapeHtml(formatTrendBias(trendBreakdown.short_term)) + "\\n" + escapeHtml(trendBreakdown.short_term_reason || "")],
        ["中线趋势", escapeHtml(formatTrendBias(trendBreakdown.mid_term)) + "\\n" + escapeHtml(trendBreakdown.mid_term_reason || "")],
        ["长线趋势", escapeHtml(formatTrendBias(trendBreakdown.long_term)) + "\\n" + escapeHtml(trendBreakdown.long_term_reason || "")],
        ["预测窗口", escapeHtml(predictionText)],
        ["分析摘要", escapeHtml(symbol.summary_markdown)],
        ["验证说明", escapeHtml(symbol.validation.note)]
      ].map(([label, value]) => `<div class="metric"><div class="label">${{label}}</div><div class="value">${{value}}</div></div>`).join("");
    }}

    async function loadDetail() {{
      const response = await fetch("/stocks/{ts_code}/data");
      if (response.status === 404) {{
        heroCard.innerHTML = '<div class="empty">该股票暂无历史记录。先触发一次分析再查看详情页。</div>';
        return;
      }}
      const payload = await response.json();
      const symbol = payload.symbol;
      generatedAt.textContent = payload.generated_at ? `最近更新：${{payload.generated_at}}` : "暂无数据";
      heroCard.innerHTML = renderHero(symbol);
      detailSparkline.innerHTML = sparkline(symbol.snapshot.minute_summary);
      renderValidationSummary(payload.validation_summary || {{}});
      renderMetrics(symbol);
      detailHistory.innerHTML = renderHistoryTimeline(symbol.history);
      detailEvidence.innerHTML = renderList(symbol.decision.evidence || []);
      detailRisks.innerHTML = renderList(symbol.memory.key_risks || []);
      detailOpenclawStatus.innerHTML = renderAutomationStatus(payload.openclaw_status);
    }}

    async function clearCurrentSymbol() {{
      const confirmed = window.confirm("这会清空当前股票的历史记录和记忆摘要，确定继续吗？");
      if (!confirmed) return;

      clearSymbolDataButton.disabled = true;
      detailGeneratedAt.textContent = "正在清空当前股票分析结果...";
      try {{
        const response = await fetch("/analysis-data/{ts_code}", {{ method: "DELETE" }});
        const payload = await response.json();
        if (!response.ok) {{
          throw new Error(payload.detail || "清空失败");
        }}

        detailGeneratedAt.textContent = `已清空 ${{(payload.cleared_symbols || []).join(", ")}} 的 ${{(payload.removed_records || 0)}} 条历史记录。`;
        heroCard.innerHTML = '<div class="empty">该股票历史已清空，可返回总览后重新触发分析。</div>';
        detailSparkline.innerHTML = '<div class="empty">暂无分时样本</div>';
        detailHistory.innerHTML = '<div class="empty">暂无历史记录</div>';
        detailMetrics.innerHTML = "";
        detailEvidence.innerHTML = '<div class="empty">暂无数据</div>';
        detailRisks.innerHTML = '<div class="empty">暂无数据</div>';
        validationSummary.innerHTML = "";
      }} catch (error) {{
        detailGeneratedAt.textContent = `清空失败：${{error.message}}`;
      }} finally {{
        clearSymbolDataButton.disabled = false;
      }}
    }}

    clearSymbolDataButton.addEventListener("click", clearCurrentSymbol);
    loadDetail().catch(error => {{
      heroCard.innerHTML = `<div class="empty">加载单股详情失败：${{escapeHtml(error.message)}}</div>`;
    }});
    """
