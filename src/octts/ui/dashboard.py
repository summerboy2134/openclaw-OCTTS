from __future__ import annotations

import json
from html import escape
from typing import Optional


def render_dashboard_html(
    data_payload: Optional[dict[str, object]] = None,
    *,
    stock_detail_href_prefix: str = "/stocks/",
    stock_detail_href_suffix: str = "",
    interactive: bool = True,
    intelligent_screening_href: Optional[str] = None,
) -> str:
    return _render_shell(
        title="OCTTS Dashboard",
        page_title="OCTTS 智能趋势总览",
        page_subtitle=(
            "总览页聚合股票池信号、历史命中状态，并支持手动输入股票代码后立即触发一次分析。"
            if interactive
            else "离线报告包已内嵌当前快照数据，打开本地 HTML 后仍可继续点击卡片进入单股详情。"
        ),
        content=_dashboard_content(interactive=interactive, intelligent_screening_href=intelligent_screening_href),
        script=_overview_script(
            initial_payload=data_payload,
            stock_detail_href_prefix=stock_detail_href_prefix,
            stock_detail_href_suffix=stock_detail_href_suffix,
            interactive=interactive,
        ),
    )


def render_stock_detail_html(
    ts_code: str,
    data_payload: Optional[dict[str, object]] = None,
    *,
    back_href: str = "/dashboard",
    interactive: bool = True,
) -> str:
    return _render_shell(
        title=f"OCTTS {ts_code}",
        page_title=f"{ts_code} 单股详情",
        page_subtitle=(
            "单股详情页展示该股票最新结构化决策、验证状态、关键价位和完整历史时间线。"
            if interactive
            else "离线详情页已内嵌该股票的最新数据和历史时间线，可在本地直接查看。"
        ),
        content=_detail_content(back_href=back_href, interactive=interactive),
        script=_detail_script(ts_code, initial_payload=data_payload, interactive=interactive),
    )


def _dashboard_content(*, interactive: bool, intelligent_screening_href: Optional[str] = None) -> str:
    if not interactive:
        intelligent_link = ""
        if intelligent_screening_href:
            intelligent_link = (
                '<div class="offline-intelligent-callout">'
                '<div>'
                '<div class="callout-title">智能选股系统已合并在本压缩包内</div>'
                '<div class="subtle">点击右侧按钮查看今日新闻、重点个股和推荐 Top3。</div>'
                '</div>'
                f'<a class="offline-primary-link" href="{escape(intelligent_screening_href)}">打开智能选股系统</a>'
                '</div>'
            )
        return f"""
      <section class="layout" style="grid-template-columns:minmax(0, 1fr);">
        <div class="main-column">
          <section class="panel">
            <div class="section-title">离线报告</div>
            <div class="subtle">当前压缩包包含总览与每只股票的详情页，点击下方卡片即可在本地继续跳转浏览。</div>
            {intelligent_link}
          </section>
          <section class="summary-grid" id="summaryGrid"></section>
          <section class="cards" id="cards"></section>
        </div>
      </section>
    """

    return """
      <section class="layout">
        <div class="main-column">
          <section class="summary-grid" id="summaryGrid"></section>
          <section class="panel">
            <div class="row">
              <div>
                <div class="section-title">智能选股中心</div>
                <div class="subtle">智能选股面向沪深 A 股，先跑技术筛选与新闻聚类，再对候选股做多维 AI 分析并生成报告</div>
              </div>
              <div class="button-row">
                <button id="runIntelligentScreeningButton" class="primary-button" type="button">运行智能选股</button>
                <a class="ghost-link" href="/intelligent-screening">展开完整智能选股页</a>
              </div>
            </div>
            <div id="intelligentScreeningStatus" class="subtle" style="margin-top:12px;"></div>
            <div id="intelligentScreeningSummary" class="summary-grid" style="margin-top:16px;"></div>
            <div class="metrics-grid" style="margin-top:16px;">
              <div class="metric">
                <div class="label">今日智能推荐 Top3</div>
                <div id="intelligentRecommendations"></div>
              </div>
              <div class="metric">
                <div class="label">智能报告摘要</div>
                <div id="intelligentReportSummary" class="value"></div>
              </div>
            </div>
          </section>
          <section class="panel">
            <div class="row">
              <div class="section-title">策略回测</div>
              <div class="subtle">最简版回测面板，直接调用 review 回测链路并展示结果。</div>
            </div>
            <div id="backtestResults" class="stack">
              <div class="empty">填写右侧参数后即可运行回测。结果会展示收益指标、资金曲线和成交明细。</div>
            </div>
          </section>
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
              <input id="phaseSelect" name="phase" type="hidden" value="review" />
              <input id="notifyToggle" name="notify" type="hidden" value="false" />
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
            <div class="section-title">回测参数</div>
            <form id="backtestForm" class="stack">
              <label class="field">
                <span class="mini-label">参数模板名称</span>
                <input
                  id="backtestTemplateNameInput"
                  name="template_name"
                  type="text"
                  placeholder="例如：复盘波段基线"
                  autocomplete="off"
                />
              </label>
              <div class="two-column-fields">
                <label class="field">
                  <span class="mini-label">已保存模板</span>
                  <select id="backtestTemplateSelect" name="template_select">
                    <option value="">选择一个模板</option>
                  </select>
                </label>
                <div class="field">
                  <span class="mini-label">模板操作</span>
                  <div class="button-row">
                    <button id="saveBacktestTemplateButton" class="secondary-button mini-button" type="button">保存</button>
                    <button id="applyBacktestTemplateButton" class="secondary-button mini-button" type="button">载入</button>
                    <button id="deleteBacktestTemplateButton" class="danger-button mini-button" type="button">删除</button>
                  </div>
                </div>
              </div>
              <label class="field">
                <span class="mini-label">股票池</span>
                <input
                  id="backtestStockPoolInput"
                  name="stock_pool"
                  type="text"
                  placeholder="留空则使用默认股票池；多个代码可用逗号分隔"
                  autocomplete="off"
                />
              </label>
              <div class="two-column-fields">
                <label class="field">
                  <span class="mini-label">开始日期</span>
                  <input id="backtestStartDateInput" name="start_date" type="date" />
                </label>
                <label class="field">
                  <span class="mini-label">结束日期</span>
                  <input id="backtestEndDateInput" name="end_date" type="date" />
                </label>
              </div>
              <div class="two-column-fields">
                <label class="field">
                  <span class="mini-label">初始资金</span>
                  <input id="backtestInitialCashInput" name="initial_cash" type="number" min="1" step="1000" value="100000" />
                </label>
                <label class="field">
                  <span class="mini-label">仓位比例</span>
                  <input id="backtestPositionSizeInput" name="position_size_pct" type="number" min="0.01" max="1" step="0.01" value="0.2" />
                </label>
              </div>
              <div class="two-column-fields">
                <label class="field">
                  <span class="mini-label">手续费率</span>
                  <input id="backtestCommissionInput" name="commission_rate" type="number" min="0" step="0.0001" value="0.0003" />
                </label>
                <label class="field">
                  <span class="mini-label">滑点率</span>
                  <input id="backtestSlippageInput" name="slippage_rate" type="number" min="0" step="0.0001" value="0.0005" />
                </label>
              </div>
              <button id="backtestButton" class="primary-button" type="submit">运行回测</button>
              <div class="subtle">当前只接入 `review` 阶段。日期会自动转换成 `YYYYMMDD` 后发送到 `/backtest`。</div>
              <div id="backtestTemplateStatus" class="subtle"></div>
              <div id="backtestStatus" class="subtle"></div>
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
    """


def _detail_content(*, back_href: str, interactive: bool) -> str:
    action_html = (
        '<button id="clearSymbolDataButton" class="danger-button" type="button">清空这只股票的分析结果</button>'
        if interactive
        else '<div class="subtle">离线详情页仅用于查看，已禁用在线清理操作。</div>'
    )
    position_panel_html = (
        """
            <section class="panel">
              <div class="section-title">持仓操作</div>
              <div class="stack">
                <label class="field">
                  <span class="mini-label">当前状态</span>
                  <select id="positionStatusSelect">
                    <option value="watching">未持有</option>
                    <option value="holding">已持有</option>
                  </select>
                </label>
                <button id="reanalyzeSymbolButton" class="primary-button" type="button">按当前状态重新分析</button>
                <div id="detailActionStatus" class="subtle">切换状态后会自动保存，重新分析会按当前状态生成建议。</div>
              </div>
            </section>
        """
        if interactive
        else """
            <section class="panel">
              <div class="section-title">持仓操作</div>
              <div class="subtle">离线详情页仅用于查看，已禁用持仓状态修改与重新分析。</div>
            </section>
        """
    )
    return f"""
      <section class="detail-shell">
        <div class="toolbar">
          <a class="ghost-link" href="{back_href}">返回总览</a>
          <div class="row">
            {action_html}
            <div id="detailGeneratedAt" class="subtle">等待加载数据...</div>
          </div>
        </div>
        <section class="layout detail-layout">
          <div class="main-column">
            <section class="hero-card" id="heroCard"></section>
            <section class="panel">
              <div class="section-title">完整分析</div>
              <div id="detailFullAnalysis" class="stack"></div>
            </section>
            <section class="metrics-grid" id="detailMetrics"></section>
            <section class="panel">
              <div class="section-title">History Replay</div>
              <div id="detailHistory" class="timeline"></div>
            </section>
          </div>
          <aside class="side-column">
            {position_panel_html}
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
    """


def _to_json_script_value(value: object) -> str:
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=False)


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
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px 20px 42px;
    }}

    .hero {{
      display: grid;
      gap: 12px;
      padding: 22px;
      margin-bottom: 16px;
      background: linear-gradient(145deg, rgba(10, 22, 43, 0.92), rgba(10, 18, 34, 0.82));
      border: 1px solid var(--panel-border);
      box-shadow: var(--shadow);
      border-radius: 24px;
      backdrop-filter: blur(14px);
    }}

    .hero h1 {{
      margin: 0;
      font-size: 26px;
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
      gap: 14px;
      align-items: start;
    }}

    .detail-layout {{
      grid-template-columns: minmax(0, 1fr) 340px;
    }}

    .main-column, .side-column {{
      display: grid;
      gap: 14px;
    }}

    .panel, .summary, .card, .hero-card {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      backdrop-filter: blur(14px);
      box-shadow: var(--shadow);
      border-radius: 22px;
    }}

    .panel {{
      padding: 16px;
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

    .recommendation-list {{
      display: grid;
      gap: 10px;
    }}

    .recommendation-item {{
      display: grid;
      gap: 6px;
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(15, 23, 42, 0.62);
      border: 1px solid rgba(148, 163, 184, 0.12);
    }}

    .recommendation-item:hover {{
      border-color: rgba(96, 165, 250, 0.3);
    }}

    .hero-card {{
      padding: 24px;
      display: grid;
      gap: 14px;
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

    .offline-intelligent-callout {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 14px;
      margin-top: 16px;
      padding: 16px;
      border-radius: 18px;
      border: 1px solid rgba(96, 165, 250, 0.28);
      background:
        radial-gradient(circle at 12% 20%, rgba(96, 165, 250, 0.16), transparent 32%),
        linear-gradient(135deg, rgba(15, 23, 42, 0.82), rgba(30, 41, 59, 0.62));
      box-shadow: 0 12px 30px rgba(2, 6, 23, 0.24);
    }}

    .callout-title {{
      margin-bottom: 6px;
      font-size: 16px;
      font-weight: 800;
      color: #f8fbff;
    }}

    .offline-primary-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 190px;
      padding: 12px 18px;
      border-radius: 999px;
      color: #dbeafe;
      background: linear-gradient(135deg, rgba(30, 64, 175, 0.82), rgba(37, 99, 235, 0.66));
      border: 1px solid rgba(147, 197, 253, 0.42);
      box-shadow: 0 10px 24px rgba(37, 99, 235, 0.18);
      font-weight: 800;
      letter-spacing: 0.01em;
    }}

    .offline-primary-link:hover {{
      transform: translateY(-1px);
      border-color: rgba(191, 219, 254, 0.62);
      box-shadow: 0 14px 28px rgba(37, 99, 235, 0.24);
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
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }}

    .summary-grid.compact {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}

    .summary {{
      padding: 14px;
    }}

    .summary .label, .mini-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .summary .value {{
      margin-top: 8px;
      font-size: 24px;
      font-weight: 800;
    }}

    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 14px;
    }}

    .card {{
      padding: 18px;
      display: grid;
      gap: 14px;
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

    .two-column-fields {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}

    .button-row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}

    .button-row button {{
      width: auto;
      flex: 1 1 0;
    }}

    .mini-button {{
      padding: 10px 12px;
      font-weight: 600;
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
    .badge.hold, .badge.entered, .badge.tracking_position, .badge.connected, .badge.position-holding {{ background: rgba(96, 165, 250, 0.14); color: var(--accent); }}
    .badge.reduce, .badge.watching_entry, .badge.watching_setup, .badge.expired, .badge.avoid, .badge.no_signal, .badge.disconnected, .badge.position-watching {{ background: rgba(251, 191, 36, 0.14); color: var(--warn); }}

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
      gap: 14px;
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
      line-height: 1.72;
    }}

    .analysis-summary-card {{
      display: grid;
      gap: 14px;
      padding: 18px;
      border-radius: 18px;
      background: linear-gradient(145deg, rgba(13, 25, 47, 0.92), rgba(11, 21, 38, 0.78));
      border: 1px solid rgba(96, 165, 250, 0.18);
    }}

    .analysis-summary-text {{
      font-size: 16px;
      line-height: 1.85;
      color: var(--text);
      white-space: pre-wrap;
    }}

    .analysis-meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}

    .analysis-meta-item {{
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(15, 23, 42, 0.54);
      border: 1px solid rgba(148, 163, 184, 0.12);
      display: grid;
      gap: 8px;
    }}

    .analysis-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(280px, 0.85fr);
      gap: 14px;
      align-items: start;
    }}

    .analysis-column {{
      display: grid;
      gap: 14px;
    }}

    .analysis-section {{
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(15, 23, 42, 0.52);
      border: 1px solid rgba(148, 163, 184, 0.12);
      display: grid;
      gap: 12px;
    }}

    .analysis-section-title {{
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}

    .analysis-section-text {{
      color: var(--text);
      line-height: 1.8;
      white-space: pre-wrap;
    }}

    .analysis-prediction-list {{
      display: grid;
      gap: 10px;
    }}

    .analysis-prediction-item {{
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(15, 23, 42, 0.54);
      border: 1px solid rgba(148, 163, 184, 0.1);
      display: grid;
      gap: 6px;
    }}

    .analysis-prediction-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      color: var(--text);
      font-weight: 700;
    }}

    .analysis-prediction-note {{
      color: var(--muted);
      line-height: 1.7;
    }}

    .analysis-empty {{
      padding: 16px 18px;
      border-radius: 16px;
      background: rgba(15, 23, 42, 0.42);
      border: 1px dashed rgba(148, 163, 184, 0.18);
      color: var(--muted);
    }}

    .empty {{
      padding: 24px;
      color: var(--muted);
      text-align: center;
      border: 1px dashed rgba(148, 163, 184, 0.2);
      border-radius: 18px;
    }}

    .table-shell {{
      overflow-x: auto;
      border: 1px solid rgba(148, 163, 184, 0.12);
      border-radius: 18px;
      background: rgba(15, 23, 42, 0.42);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
    }}

    th, td {{
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid rgba(148, 163, 184, 0.12);
      font-size: 14px;
    }}

    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: rgba(15, 23, 42, 0.76);
    }}

    tr:last-child td {{
      border-bottom: none;
    }}

    .text-good {{
      color: var(--good);
    }}

    .text-bad {{
      color: var(--bad);
    }}

    .mono {{
      font-family: ui-monospace, SFMono-Regular, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}

    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}

    .chart-shell {{
      padding: 16px;
      background: rgba(15, 23, 42, 0.56);
      border-radius: 18px;
      border: 1px solid rgba(148, 163, 184, 0.12);
      display: grid;
      gap: 10px;
    }}

    .chart-meta {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}

    .chart-svg {{
      width: 100%;
      height: 180px;
      display: block;
      border-radius: 14px;
      background: linear-gradient(180deg, rgba(148, 163, 184, 0.08), rgba(15, 23, 42, 0.04));
    }}

    .chart-caption {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }}

    @media (max-width: 1100px) {{
      .layout, .detail-layout, .chart-and-summary, .chart-grid, .analysis-grid {{
        grid-template-columns: 1fr;
      }}

      .two-column-fields {{
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

    function formatPriceZone(zone) {{
      if (!zone) return "—";
      const low = zone.low;
      const high = zone.high;
      if ((low === null || low === undefined || low === "") && (high === null || high === undefined || high === "")) {{
        return "—";
      }}
      return `${{formatValue(low)}} - ${{formatValue(high)}}`;
    }}

    function formatEntryZoneDisplay(signal, zone) {{
      const value = formatPriceZone(zone);
      if (signal === "avoid" && value === "—") return "观望";
      return value;
    }}

    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
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

    const POSITION_STATUS_LABELS = {{
      holding: "已持有",
      watching: "未持有"
    }};

    const VALIDATION_STATUS_LABELS = {{
      no_signal: "无交易信号",
      watching_setup: "等待条件成熟",
      watching_entry: "等待入场",
      tracking_position: "持仓跟踪",
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

    function formatPositionStatus(value) {{
      return labelFor(POSITION_STATUS_LABELS, value || "watching");
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

    function formatSnapshotTradeDate(snapshot) {{
      const tradeDate = snapshot && snapshot.trade_date;
      const text = String(tradeDate || "");
      if (text.length === 8) {{
        return `行情截至 ${{text.slice(0, 4)}}-${{text.slice(4, 6)}}-${{text.slice(6, 8)}}`;
      }}
      return `行情截至 ${{text || "—"}}`;
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
          <div class="subtle" style="margin-top:4px;">${{escapeHtml(formatSnapshotTradeDate(item.snapshot))}}</div>
          <div style="margin-top:8px;">${{escapeHtml(item.validation.note)}}</div>
        </div>
      `).join("");
    }}

    {script}
  </script>
</body>
</html>"""


def _overview_script(
    *,
    initial_payload: Optional[dict[str, object]] = None,
    stock_detail_href_prefix: str = "/stocks/",
    stock_detail_href_suffix: str = "",
    interactive: bool = True,
) -> str:
    prefix = f"""
    const IS_INTERACTIVE = {"true" if interactive else "false"};
    const INITIAL_DASHBOARD_PAYLOAD = {_to_json_script_value(initial_payload)};
    const STOCK_DETAIL_HREF_PREFIX = {json.dumps(stock_detail_href_prefix, ensure_ascii=False)};
    const STOCK_DETAIL_HREF_SUFFIX = {json.dumps(stock_detail_href_suffix, ensure_ascii=False)};
    """
    return prefix + """
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
    const runIntelligentScreeningButton = document.getElementById("runIntelligentScreeningButton");
    const intelligentScreeningStatus = document.getElementById("intelligentScreeningStatus");
    const intelligentScreeningSummary = document.getElementById("intelligentScreeningSummary");
    const intelligentRecommendations = document.getElementById("intelligentRecommendations");
    const intelligentReportSummary = document.getElementById("intelligentReportSummary");
    const backtestForm = document.getElementById("backtestForm");
    const backtestStockPoolInput = document.getElementById("backtestStockPoolInput");
    const backtestStartDateInput = document.getElementById("backtestStartDateInput");
    const backtestEndDateInput = document.getElementById("backtestEndDateInput");
    const backtestInitialCashInput = document.getElementById("backtestInitialCashInput");
    const backtestPositionSizeInput = document.getElementById("backtestPositionSizeInput");
    const backtestCommissionInput = document.getElementById("backtestCommissionInput");
    const backtestSlippageInput = document.getElementById("backtestSlippageInput");
    const backtestTemplateNameInput = document.getElementById("backtestTemplateNameInput");
    const backtestTemplateSelect = document.getElementById("backtestTemplateSelect");
    const saveBacktestTemplateButton = document.getElementById("saveBacktestTemplateButton");
    const applyBacktestTemplateButton = document.getElementById("applyBacktestTemplateButton");
    const deleteBacktestTemplateButton = document.getElementById("deleteBacktestTemplateButton");
    const backtestTemplateStatus = document.getElementById("backtestTemplateStatus");
    const backtestButton = document.getElementById("backtestButton");
    const backtestStatus = document.getElementById("backtestStatus");
    const backtestResults = document.getElementById("backtestResults");
    const BACKTEST_TEMPLATE_STORAGE_KEY = "octts.backtestTemplates.v1";

    function buildStockDetailHref(tsCode) {
      return `${STOCK_DETAIL_HREF_PREFIX}${encodeURIComponent(tsCode)}${STOCK_DETAIL_HREF_SUFFIX}`;
    }

    function renderSummary(payload) {
      const items = payload.cards || [];
      const statuses = payload.validation_summary || {};
      const intelligent = payload.intelligent_screening || {};
      summaryGrid.innerHTML = [
        renderSummaryCard("跟踪股票数", items.length),
        renderSummaryCard("默认股票池", (payload.default_stock_pool || []).length),
        renderSummaryCard("今日 Top3", intelligent.today_top_count || 0),
        renderSummaryCard("热点主题数", intelligent.news_cluster_count || 0),
        renderSummaryCard("止盈触发", statuses.take_profit_hit || 0),
        renderSummaryCard("止损触发", statuses.stop_loss_hit || 0),
        renderSummaryCard("等待条件成熟", statuses.watching_setup || 0),
        renderSummaryCard("等待入场", statuses.watching_entry || 0),
        renderSummaryCard("持仓跟踪", statuses.tracking_position || 0)
      ].join("");
    }

    function formatCurrency(value) {
      if (!Number.isFinite(Number(value))) return "—";
      return Number(value).toLocaleString("zh-CN", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      });
    }

    function formatPercent(value) {
      if (!Number.isFinite(Number(value))) return "—";
      return `${(Number(value) * 100).toFixed(2)}%`;
    }

    function formatDateLabel(value) {
      const text = String(value || "");
      if (text.length !== 8) return text || "—";
      return `${text.slice(0, 4)}-${text.slice(4, 6)}-${text.slice(6, 8)}`;
    }

    function getSignedClass(value) {
      if (!Number.isFinite(Number(value))) return "";
      if (Number(value) > 0) return "text-good";
      if (Number(value) < 0) return "text-bad";
      return "";
    }

    function toCompactDate(value) {
      return String(value || "").replaceAll("-", "");
    }

    function fromCompactDate(value) {
      if (!value || String(value).length !== 8) return "";
      return `${String(value).slice(0, 4)}-${String(value).slice(4, 6)}-${String(value).slice(6, 8)}`;
    }

    function shiftDate(baseDate, deltaDays) {
      const shifted = new Date(baseDate);
      shifted.setDate(shifted.getDate() + deltaDays);
      return shifted;
    }

    function formatDateInputValue(date) {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0");
      const day = String(date.getDate()).padStart(2, "0");
      return `${year}-${month}-${day}`;
    }

    function setBacktestDateDefaults() {
      const today = new Date();
      const startDate = shiftDate(today, -90);
      if (!backtestStartDateInput.value) {
        backtestStartDateInput.value = formatDateInputValue(startDate);
      }
      if (!backtestEndDateInput.value) {
        backtestEndDateInput.value = formatDateInputValue(today);
      }
    }

    function normalizeStockPoolInput(value) {
      return Array.from(
        new Set(
          String(value || "")
            .split(/[\\s,，;；]+/)
            .map(item => item.trim().toUpperCase())
            .filter(Boolean)
        )
      );
    }

    function formatConfidencePercent(value) {
      const number = Number(value);
      if (!Number.isFinite(number)) return "—";
      const normalized = number > 1 ? number : number * 100;
      return `${normalized.toFixed(1)}%`;
    }

    function readBacktestTemplates() {
      try {
        const payload = JSON.parse(window.localStorage.getItem(BACKTEST_TEMPLATE_STORAGE_KEY) || "[]");
        if (!Array.isArray(payload)) return [];
        return payload.filter(item => item && typeof item.name === "string" && item.name.trim());
      } catch (error) {
        return [];
      }
    }

    function writeBacktestTemplates(templates) {
      window.localStorage.setItem(BACKTEST_TEMPLATE_STORAGE_KEY, JSON.stringify(templates));
    }

    function getBacktestFormValues() {
      return {
        stock_pool: (backtestStockPoolInput.value || "").trim(),
        start_date: backtestStartDateInput.value,
        end_date: backtestEndDateInput.value,
        initial_cash: String(backtestInitialCashInput.value || ""),
        position_size_pct: String(backtestPositionSizeInput.value || ""),
        commission_rate: String(backtestCommissionInput.value || ""),
        slippage_rate: String(backtestSlippageInput.value || "")
      };
    }

    function applyBacktestTemplateValues(template) {
      if (!template) return;
      backtestTemplateNameInput.value = template.name || "";
      backtestStockPoolInput.value = template.stock_pool || "";
      backtestStartDateInput.value = template.start_date || "";
      backtestEndDateInput.value = template.end_date || "";
      backtestInitialCashInput.value = template.initial_cash || "100000";
      backtestPositionSizeInput.value = template.position_size_pct || "0.2";
      backtestCommissionInput.value = template.commission_rate || "0.0003";
      backtestSlippageInput.value = template.slippage_rate || "0.0005";
    }

    function renderBacktestTemplateOptions(selectedName = "") {
      const templates = readBacktestTemplates().sort((a, b) => a.name.localeCompare(b.name, "zh-CN"));
      backtestTemplateSelect.innerHTML = [
        '<option value="">选择一个模板</option>',
        ...templates.map(item => `<option value="${escapeHtml(item.name)}" ${item.name === selectedName ? "selected" : ""}>${escapeHtml(item.name)}</option>`)
      ].join("");
    }

    function loadSelectedBacktestTemplate() {
      const templateName = backtestTemplateSelect.value;
      const template = readBacktestTemplates().find(item => item.name === templateName);
      if (!template) {
        backtestTemplateStatus.textContent = "请选择一个已保存模板。";
        return;
      }
      applyBacktestTemplateValues(template);
      backtestTemplateStatus.textContent = `已载入模板：${template.name}`;
    }

    function saveCurrentBacktestTemplate() {
      const templateName = (backtestTemplateNameInput.value || "").trim();
      if (!templateName) {
        backtestTemplateStatus.textContent = "请先填写模板名称再保存。";
        backtestTemplateNameInput.focus();
        return;
      }

      const templates = readBacktestTemplates().filter(item => item.name !== templateName);
      templates.push({
        name: templateName,
        ...getBacktestFormValues(),
        saved_at: new Date().toISOString()
      });
      writeBacktestTemplates(templates);
      renderBacktestTemplateOptions(templateName);
      backtestTemplateStatus.textContent = `模板已保存：${templateName}`;
    }

    function deleteSelectedBacktestTemplate() {
      const templateName = backtestTemplateSelect.value || (backtestTemplateNameInput.value || "").trim();
      if (!templateName) {
        backtestTemplateStatus.textContent = "请先选择一个模板再删除。";
        return;
      }
      const templates = readBacktestTemplates();
      const nextTemplates = templates.filter(item => item.name !== templateName);
      if (nextTemplates.length === templates.length) {
        backtestTemplateStatus.textContent = `未找到模板：${templateName}`;
        return;
      }
      writeBacktestTemplates(nextTemplates);
      renderBacktestTemplateOptions("");
      if (backtestTemplateNameInput.value.trim() === templateName) {
        backtestTemplateNameInput.value = "";
      }
      backtestTemplateStatus.textContent = `模板已删除：${templateName}`;
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

    function renderIntelligentRecommendations(items) {
      if (!items || !items.length) {
        return '<div class="empty">还没有智能选股结果。点击上方按钮后，这里会显示推荐股票和智能摘要。</div>';
      }
      return `<div class="recommendation-list">${items.map(item => {
        const sourceTag = item.source_tag || "今日Top3";
        const repeatTag = item.is_repeat_pick ? ' · 连续入选' : '';
        return `
        <div class="recommendation-item">
          <div class="row">
            <strong>${escapeHtml(item.ts_code)}${item.name ? ` · ${escapeHtml(item.name)}` : ""}</strong>
            <span class="badge hold">${escapeHtml(Number(item.recommendation_score || item.score || 0).toFixed(1))} 分</span>
          </div>
          <div class="subtle">${escapeHtml(sourceTag + repeatTag)} · 技术信号：${escapeHtml(item.technical_signal || "信号待确认")} · 置信度 ${escapeHtml(formatConfidencePercent(item.confidence))}</div>
          <div>${escapeHtml(item.recommendation || "建议继续观察")}</div>
          <div class="subtle">${escapeHtml(item.summary || "")}</div>
        </div>`;
      }).join("")}</div>`;
    }

    function renderIntelligentOverview(payload) {
      if (!intelligentScreeningSummary || !intelligentRecommendations || !intelligentReportSummary) return;
      const intelligent = payload || {};
      intelligentScreeningSummary.innerHTML = [
        renderSummaryCard("执行策略数", intelligent.strategy_count || 0, "单只股票的策略数表示它被多少个筛选策略同时命中"),
        renderSummaryCard("候选股票数", intelligent.total_stocks || 0),
        renderSummaryCard("今日 Top3", intelligent.today_top_count || 0),
        renderSummaryCard("持仓复盘", intelligent.continuation_count || 0)
      ].join("");
      intelligentRecommendations.innerHTML = renderIntelligentRecommendations(intelligent.today_top3 || []);
      intelligentReportSummary.innerHTML = `
        <div><strong>${escapeHtml(intelligent.report_title || "智能选股报告")}</strong></div>
        <div class="subtle" style="margin-top:8px;">${escapeHtml(intelligent.generated_at ? `最近刷新：${intelligent.generated_at}` : "暂无最近运行时间")}</div>
        <div style="margin-top:10px;">${escapeHtml(intelligent.report_summary || "运行一次智能选股后，这里会显示最新摘要。")}</div>
      `;
    }

    function renderCard(item) {
      const zone = item.decision.entry_zone || {};
      const trendBreakdown = item.trend_breakdown || {};
      const positionStatus = item.position_status || "watching";
      const positionBadgeClass = positionStatus === "holding" ? "position-holding" : "position-watching";
      return `<a class="card-link" href="${buildStockDetailHref(item.ts_code)}">
        <article class="card">
          <div class="row">
            <div>
              <div class="title">${escapeHtml(item.ts_code)}${item.name ? " · " + escapeHtml(item.name) : ""}</div>
              <div class="subtle">${escapeHtml(item.generated_at)} · ${escapeHtml(formatPhase(item.phase))}</div>
              <div class="subtle">${escapeHtml(formatSnapshotTradeDate(item.snapshot))}</div>
            </div>
            <div class="row">
              <span class="badge ${escapeHtml(positionBadgeClass)}">${escapeHtml(formatPositionStatus(positionStatus))}</span>
              <span class="badge ${escapeHtml(item.decision.signal)}">${escapeHtml(formatSignal(item.decision.signal))}</span>
              <span class="badge ${escapeHtml(item.validation.status)}">${escapeHtml(formatValidationStatus(item.validation.status))}</span>
            </div>
          </div>
          <div class="metrics-grid">
            <div class="metric"><div class="label">趋势判断</div><div class="value">${escapeHtml(item.trend_judgement)}</div></div>
            <div class="metric"><div class="label">短 / 中 / 长趋势</div><div class="value">${escapeHtml(`${formatTrendBias(trendBreakdown.short_term)} / ${formatTrendBias(trendBreakdown.mid_term)} / ${formatTrendBias(trendBreakdown.long_term)}`)}</div></div>
            <div class="metric"><div class="label">信心分数</div><div class="value">${Math.round((item.decision.confidence_score || 0) * 100)}%</div></div>
            <div class="metric"><div class="label">入场区间</div><div class="value">${formatEntryZoneDisplay(item.decision.signal, zone)}</div></div>
            <div class="metric"><div class="label">目标位</div><div class="value">${(item.decision.take_profit || []).map(v => formatValue(v)).join(" / ") || "—"}</div></div>
          </div>
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

    const EXIT_REASON_LABELS = {
      stop_loss: "止损触发",
      take_profit: "止盈触发",
      horizon_exit: "持有周期到期"
    };

    function formatExitReason(value) {
      return labelFor(EXIT_REASON_LABELS, value);
    }

    function buildDrawdownSeries(dailyPositions) {
      let peak = 0;
      return (dailyPositions || []).map(item => {
        const equity = Number(item.equity);
        peak = Math.max(peak, equity);
        const drawdown = peak > 0 ? (peak - equity) / peak : 0;
        return {
          trade_date: item.trade_date,
          value: drawdown
        };
      });
    }

    function renderBacktestLineChart(series, options = {}) {
      if (!series || !series.length) {
        return '<div class="empty">暂无图表数据。</div>';
      }

      const width = 640;
      const height = 180;
      const paddingX = 18;
      const paddingY = 18;
      const values = series.map(item => Number(item.value)).filter(value => Number.isFinite(value));
      if (!values.length) {
        return '<div class="empty">暂无图表数据。</div>';
      }

      const min = options.minValue !== undefined ? Number(options.minValue) : Math.min(...values);
      const max = options.maxValue !== undefined ? Number(options.maxValue) : Math.max(...values);
      const range = max === min ? Math.abs(max || 1) : (max - min);
      const innerWidth = width - paddingX * 2;
      const innerHeight = height - paddingY * 2;
      const points = values.map((value, index) => {
        const x = series.length === 1 ? width / 2 : paddingX + (index / (series.length - 1)) * innerWidth;
        const y = paddingY + innerHeight - ((value - min) / range) * innerHeight;
        return { x, y };
      });
      const polyline = points.map(point => `${point.x},${point.y}`).join(" ");
      const areaPoints = `${paddingX},${height - paddingY} ${polyline} ${width - paddingX},${height - paddingY}`;
      const last = series[series.length - 1];
      const first = series[0];
      const lastPoint = points[points.length - 1];
      const stroke = options.stroke || "#60a5fa";
      const fill = options.fill || "rgba(96, 165, 250, 0.18)";
      const baselineValue = Number(options.baselineValue ?? min);
      const baselineY = paddingY + innerHeight - ((baselineValue - min) / range) * innerHeight;
      const yTopLabel = options.valueFormatter ? options.valueFormatter(max) : formatValue(max, 2);
      const yBottomLabel = options.valueFormatter ? options.valueFormatter(min) : formatValue(min, 2);
      const latestLabel = options.valueFormatter ? options.valueFormatter(last.value) : formatValue(last.value, 2);
      const startLabel = formatDateLabel(first.trade_date);
      const endLabel = formatDateLabel(last.trade_date);

      return `
        <div class="chart-shell">
          <div class="chart-meta">
            <div>
              <div class="section-title">${escapeHtml(options.title || "曲线")}</div>
              <div class="chart-caption">${escapeHtml(options.caption || "")}</div>
            </div>
            <div class="value ${getSignedClass(options.signed ? last.value : 0)}">${escapeHtml(latestLabel)}</div>
          </div>
          <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
            <line x1="${paddingX}" y1="${paddingY}" x2="${paddingX}" y2="${height - paddingY}" stroke="rgba(148, 163, 184, 0.18)" />
            <line x1="${paddingX}" y1="${height - paddingY}" x2="${width - paddingX}" y2="${height - paddingY}" stroke="rgba(148, 163, 184, 0.18)" />
            <line x1="${paddingX}" y1="${baselineY}" x2="${width - paddingX}" y2="${baselineY}" stroke="rgba(148, 163, 184, 0.18)" stroke-dasharray="4 4" />
            <polyline fill="${fill}" stroke="none" points="${areaPoints}" />
            <polyline fill="none" stroke="${stroke}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${polyline}" />
            <circle cx="${lastPoint.x}" cy="${lastPoint.y}" r="4" fill="${stroke}" />
            <text x="${paddingX}" y="14" fill="#93a4bf" font-size="11">${escapeHtml(yTopLabel)}</text>
            <text x="${paddingX}" y="${height - 4}" fill="#93a4bf" font-size="11">${escapeHtml(yBottomLabel)}</text>
            <text x="${paddingX}" y="${height - 4}" dx="40" fill="#93a4bf" font-size="11">${escapeHtml(startLabel)}</text>
            <text x="${width - paddingX}" y="${height - 4}" text-anchor="end" fill="#93a4bf" font-size="11">${escapeHtml(endLabel)}</text>
          </svg>
        </div>
      `;
    }

    function renderBacktestTrades(trades) {
      if (!trades || !trades.length) {
        return '<div class="empty">本次回测没有产生平仓交易，可能是区间内没有触发买点，或持仓尚未在区间内退出。</div>';
      }

      return `
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>股票</th>
                <th>信号日</th>
                <th>入场日</th>
                <th>出场日</th>
                <th>收益率</th>
                <th>PnL</th>
                <th>退出原因</th>
              </tr>
            </thead>
            <tbody>
              ${trades.map(item => `
                <tr>
                  <td class="mono">${escapeHtml(item.ts_code)}</td>
                  <td class="mono">${escapeHtml(item.signal_date)}</td>
                  <td class="mono">${escapeHtml(item.entry_date)}</td>
                  <td class="mono">${escapeHtml(item.exit_date)}</td>
                  <td class="${getSignedClass(item.return_pct)}">${escapeHtml(formatPercent(item.return_pct))}</td>
                  <td class="${getSignedClass(item.pnl)}">${escapeHtml(formatCurrency(item.pnl))}</td>
                  <td>${escapeHtml(formatExitReason(item.exit_reason))}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      `;
    }

    function renderBacktestDailyPositions(dailyPositions) {
      if (!dailyPositions || !dailyPositions.length) {
        return '<div class="empty">暂无每日权益数据。</div>';
      }

      const rows = dailyPositions.slice(-10).reverse();
      return `
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>交易日</th>
                <th>现金</th>
                <th>持仓市值</th>
                <th>总权益</th>
                <th>持仓数</th>
              </tr>
            </thead>
            <tbody>
              ${rows.map(item => `
                <tr>
                  <td class="mono">${escapeHtml(item.trade_date)}</td>
                  <td>${escapeHtml(formatCurrency(item.cash))}</td>
                  <td>${escapeHtml(formatCurrency(item.market_value))}</td>
                  <td>${escapeHtml(formatCurrency(item.equity))}</td>
                  <td>${escapeHtml(String(item.open_positions ?? 0))}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      `;
    }

    function renderBacktestResult(result) {
      const metrics = result.metrics || {};
      const dailyPositions = result.daily_positions || [];
      const stockPoolText = (result.stock_pool || []).join(", ") || "默认股票池";
      const equitySeries = dailyPositions.map(item => ({
        trade_date: item.trade_date,
        value: Number(item.equity)
      }));
      const drawdownSeries = buildDrawdownSeries(dailyPositions);

      backtestResults.innerHTML = `
        <div class="row">
          <div>
            <div class="title">Review 回测结果</div>
            <div class="subtle mono">${escapeHtml(result.start_date)} - ${escapeHtml(result.end_date)}</div>
          </div>
          <div class="subtle">${escapeHtml(stockPoolText)}</div>
        </div>
        <div class="metrics-grid">
          <div class="metric"><div class="label">初始资金</div><div class="value">${escapeHtml(formatCurrency(result.initial_cash))}</div></div>
          <div class="metric"><div class="label">期末权益</div><div class="value ${getSignedClass((result.ending_cash || 0) - (result.initial_cash || 0))}">${escapeHtml(formatCurrency(result.ending_cash))}</div></div>
          <div class="metric"><div class="label">总收益率</div><div class="value ${getSignedClass(metrics.total_return)}">${escapeHtml(formatPercent(metrics.total_return))}</div></div>
          <div class="metric"><div class="label">年化收益率</div><div class="value ${getSignedClass(metrics.annual_return)}">${escapeHtml(formatPercent(metrics.annual_return))}</div></div>
          <div class="metric"><div class="label">最大回撤</div><div class="value">${escapeHtml(formatPercent(metrics.max_drawdown))}</div></div>
          <div class="metric"><div class="label">胜率</div><div class="value">${escapeHtml(formatPercent(metrics.win_rate))}</div></div>
          <div class="metric"><div class="label">盈利因子</div><div class="value">${escapeHtml(formatValue(metrics.profit_factor, 2))}</div></div>
          <div class="metric"><div class="label">交易次数</div><div class="value">${escapeHtml(String(metrics.trade_count || 0))}</div></div>
        </div>
        <div class="chart-grid">
          ${renderBacktestLineChart(equitySeries, {
            title: "权益曲线",
            caption: "展示账户总权益随交易日变化，能更直观看到策略斜率与波动。",
            valueFormatter: formatCurrency,
            stroke: "#60a5fa",
            fill: "rgba(96, 165, 250, 0.16)"
          })}
          ${renderBacktestLineChart(drawdownSeries, {
            title: "回撤曲线",
            caption: "回撤越高代表离历史净值峰值越远，可快速定位风险集中区间。",
            valueFormatter: formatPercent,
            stroke: "#f87171",
            fill: "rgba(248, 113, 113, 0.16)",
            baselineValue: 0,
            minValue: 0
          })}
        </div>
        <div class="stack">
          <div>
            <div class="section-title">成交明细</div>
            ${renderBacktestTrades(result.trades || [])}
          </div>
          <div>
            <div class="section-title">最近 10 个交易日权益</div>
            ${renderBacktestDailyPositions(dailyPositions)}
          </div>
        </div>
      `;
    }

    async function triggerIntelligentScreening() {
      runIntelligentScreeningButton.disabled = true;
      intelligentScreeningStatus.textContent = "正在启动智能选股...";
      try {
        const response = await fetch("/screen/intelligent/jobs", {
          method: "POST"
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "启动智能选股失败");
        }
        intelligentScreeningStatus.textContent = "智能选股已启动，正在跳转...";
        const jobId = payload.job_id || payload.job?.job_id;
        if (jobId) {
          window.sessionStorage.setItem("octts:intelligent-screening:pending-job-id", jobId);
        }
        window.location.href = jobId ? `/intelligent-screening?job_id=${encodeURIComponent(jobId)}` : "/intelligent-screening";
      } catch (error) {
        intelligentScreeningStatus.textContent = `启动失败：${error.message}`;
        runIntelligentScreeningButton.disabled = false;
      }
    }

    async function loadDashboard() {
      const payload = INITIAL_DASHBOARD_PAYLOAD || await (async () => {
        const response = await fetch("/dashboard/data");
        return response.json();
      })();
      generatedAt.textContent = payload.generated_at ? `最近更新：${payload.generated_at}` : "暂无数据，请先触发一次分析。";
      renderSummary(payload);
      renderIntelligentOverview(payload.intelligent_screening || {});
      if (IS_INTERACTIVE && defaultStockPool) {
        renderDefaultStockPool(payload.default_stock_pool || []);
      }
      if (openclawStatus) {
        openclawStatus.innerHTML = renderAutomationStatus(payload.openclaw_status);
      }
      if (IS_INTERACTIVE && backtestStockPoolInput && !backtestStockPoolInput.value && (payload.default_stock_pool || []).length) {
        backtestStockPoolInput.placeholder = `留空则使用默认股票池：${payload.default_stock_pool.join(", ")}`;
      }
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

    function syncBacktestTemplateNameFromSelection() {
      if (backtestTemplateSelect.value) {
        backtestTemplateNameInput.value = backtestTemplateSelect.value;
      }
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
        window.location.href = buildStockDetailHref(tsCode);
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

    async function runBacktest(event) {
      event.preventDefault();
      const startDate = toCompactDate(backtestStartDateInput.value);
      const endDate = toCompactDate(backtestEndDateInput.value);
      const stockPool = normalizeStockPoolInput(backtestStockPoolInput.value);
      const initialCash = Number(backtestInitialCashInput.value);
      const positionSizePct = Number(backtestPositionSizeInput.value);
      const commissionRate = Number(backtestCommissionInput.value);
      const slippageRate = Number(backtestSlippageInput.value);

      if (!startDate || !endDate) {
        backtestStatus.textContent = "请先填写开始日期和结束日期。";
        return;
      }
      if (startDate > endDate) {
        backtestStatus.textContent = "开始日期不能晚于结束日期。";
        return;
      }

      backtestButton.disabled = true;
      backtestStatus.textContent = "正在运行回测...";

      try {
        const response = await fetch("/backtest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            phase: "review",
            start_date: startDate,
            end_date: endDate,
            stock_pool: stockPool.length ? stockPool : null,
            initial_cash: initialCash,
            position_size_pct: positionSizePct,
            commission_rate: commissionRate,
            slippage_rate: slippageRate
          })
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "回测请求失败");
        }

        backtestStatus.textContent = `回测完成：${payload.metrics?.trade_count || 0} 笔交易，期末权益 ${formatCurrency(payload.ending_cash)}。`;
        backtestStartDateInput.value = fromCompactDate(payload.start_date);
        backtestEndDateInput.value = fromCompactDate(payload.end_date);
        renderBacktestResult(payload);
      } catch (error) {
        backtestStatus.textContent = `回测失败：${error.message}`;
      } finally {
        backtestButton.disabled = false;
      }
    }

    if (IS_INTERACTIVE) {
      setBacktestDateDefaults();
      renderBacktestTemplateOptions();
      manualAnalyzeForm.addEventListener("submit", triggerManualAnalysis);
      backtestForm.addEventListener("submit", runBacktest);
      runIntelligentScreeningButton.addEventListener("click", () => {
        triggerIntelligentScreening();
      });
      saveBacktestTemplateButton.addEventListener("click", () => {
        saveCurrentBacktestTemplate();
      });
      applyBacktestTemplateButton.addEventListener("click", () => {
        loadSelectedBacktestTemplate();
      });
      deleteBacktestTemplateButton.addEventListener("click", () => {
        deleteSelectedBacktestTemplate();
      });
      backtestTemplateSelect.addEventListener("change", () => {
        syncBacktestTemplateNameFromSelection();
      });
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
    }

    loadDashboard().catch(error => {
      cards.innerHTML = `<div class="empty">加载总览失败：${escapeHtml(error.message)}</div>`;
    });
    """


def _detail_script(
    ts_code: str,
    *,
    initial_payload: Optional[dict[str, object]] = None,
    interactive: bool = True,
) -> str:
    prefix = f"""
    const IS_INTERACTIVE = {"true" if interactive else "false"};
    let initialDetailPayload = {_to_json_script_value(initial_payload)};
    """
    return prefix + f"""
    const detailGeneratedAt = document.getElementById("detailGeneratedAt");
    const generatedAt = detailGeneratedAt;
    const heroCard = document.getElementById("heroCard");
    const detailFullAnalysis = document.getElementById("detailFullAnalysis");
    const detailMetrics = document.getElementById("detailMetrics");
    const detailHistory = document.getElementById("detailHistory");
    const detailEvidence = document.getElementById("detailEvidence");
    const detailRisks = document.getElementById("detailRisks");
    const detailOpenclawStatus = document.getElementById("detailOpenclawStatus");
    const clearSymbolDataButton = document.getElementById("clearSymbolDataButton");
    const positionStatusSelect = document.getElementById("positionStatusSelect");
    const reanalyzeSymbolButton = document.getElementById("reanalyzeSymbolButton");
    const detailActionStatus = document.getElementById("detailActionStatus");
    let currentPositionStatus = "watching";
    let currentStockPool = [];

    function renderHero(symbol) {{
      return `
        <div class="row">
          <div>
            <div class="title">${{escapeHtml(symbol.ts_code)}}${{symbol.name ? " · " + escapeHtml(symbol.name) : ""}}</div>
            <div class="subtle">${{escapeHtml(symbol.generated_at)}} · ${{escapeHtml(formatPhase(symbol.phase))}}</div>
            <div class="subtle">${{escapeHtml(formatSnapshotTradeDate(symbol.snapshot))}}</div>
          </div>
          <div class="row">
            <span class="badge ${{escapeHtml(symbol.decision.signal)}}">${{escapeHtml(formatSignal(symbol.decision.signal))}}</span>
            <span class="badge ${{escapeHtml(symbol.validation.status)}}">${{escapeHtml(formatValidationStatus(symbol.validation.status))}}</span>
          </div>
        </div>
        <div class="stack">
          <div>
            <div class="mini-label">趋势判断</div>
            <div class="value">${{escapeHtml(symbol.trend_judgement || '暂无')}}</div>
          </div>
        </div>
      `;
    }}

    function formatSignedAmount(value) {{
      if (value === null || value === undefined || value === "") return "—";
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return "—";
      const sign = numeric > 0 ? "+" : "";
      const inYi = numeric / 10000;
      return `${{sign}}${{inYi.toFixed(2)}}亿`;
    }}

    function formatPercentValue(value, digits = 2) {{
      if (value === null || value === undefined || value === "") return "—";
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return "—";
      const sign = numeric > 0 ? "+" : "";
      return `${{sign}}${{numeric.toFixed(digits)}}%`;
    }}

    function renderPredictionWindows(predictionWindows) {{
      if (!predictionWindows || !predictionWindows.length) return '<div class="analysis-empty">暂无预测窗口说明</div>';
      return `
        <div class="analysis-prediction-list">
          ${{predictionWindows.map(item => {{
            const confidence = Math.round((item.confidence_score || 0) * 100);
            return `
              <div class="analysis-prediction-item">
                <div class="analysis-prediction-head">
                  <span>${{escapeHtml(formatPredictionWindow(item.window))}}：${{escapeHtml(formatTrendBias(item.bias))}}</span>
                  <span class="mini-label">置信度 ${{confidence}}%</span>
                </div>
                <div class="analysis-prediction-note">${{escapeHtml(item.rationale || '暂无补充说明')}}</div>
              </div>
            `;
          }}).join("")}}
        </div>
      `;
    }}

    function renderFullAnalysis(symbol) {{
      if (!detailFullAnalysis) return;
      const riskItems = (symbol.risk_warning || []).length ? symbol.risk_warning : ((symbol.memory || {{}}).key_risks || []);
      const observationItems = symbol.observation_points || [];
      detailFullAnalysis.innerHTML = `
        <div class="analysis-summary-card">
          <div class="analysis-section-title">分析摘要</div>
          <div class="analysis-summary-text">${{escapeHtml(symbol.summary_markdown || '暂无')}} </div>
          <div class="analysis-meta-grid">
            <div class="analysis-meta-item">
              <div class="analysis-section-title">操作建议</div>
              <div class="analysis-section-text">${{escapeHtml(symbol.operation_advice || '暂无')}}</div>
            </div>
            <div class="analysis-meta-item">
              <div class="analysis-section-title">验证说明</div>
              <div class="analysis-section-text">${{escapeHtml((symbol.validation || {{}}).note || '暂无')}}</div>
            </div>
          </div>
        </div>
        <div class="analysis-grid">
          <div class="analysis-column">
            <div class="analysis-section">
              <div class="analysis-section-title">未来 1/3/5 日判断</div>
              ${{renderPredictionWindows(symbol.prediction_windows || [])}}
            </div>
          </div>
          <div class="analysis-column">
            <div class="analysis-section">
              <div class="analysis-section-title">观察点</div>
              ${{renderList(observationItems)}}
            </div>
            <div class="analysis-section">
              <div class="analysis-section-title">风险提示</div>
              ${{renderList(riskItems)}}
            </div>
          </div>
        </div>
      `;
    }}

    function renderMetrics(symbol) {{
      const zone = symbol.decision.entry_zone || {{}};
      const trendBreakdown = symbol.trend_breakdown || {{}};
      const analysisContext = symbol.analysis_context || {{}};
      const technical = analysisContext.technical || {{}};
      const moneyflow = analysisContext.moneyflow || {{}};
      const largeOrderNet = (moneyflow.buy_lg_amount !== undefined && moneyflow.buy_lg_amount !== null && moneyflow.sell_lg_amount !== undefined && moneyflow.sell_lg_amount !== null)
        ? Number(moneyflow.buy_lg_amount) - Number(moneyflow.sell_lg_amount)
        : null;
      const superLargeOrderNet = (moneyflow.buy_elg_amount !== undefined && moneyflow.buy_elg_amount !== null && moneyflow.sell_elg_amount !== undefined && moneyflow.sell_elg_amount !== null)
        ? Number(moneyflow.buy_elg_amount) - Number(moneyflow.sell_elg_amount)
        : null;
      detailMetrics.innerHTML = [
        ["入场区间", formatEntryZoneDisplay(symbol.decision.signal, zone)],
        ["止损位", formatValue(symbol.decision.stop_loss)],
        ["目标位", (symbol.decision.take_profit || []).map(v => formatValue(v)).join(" / ") || "—"],
        ["持有周期", escapeHtml(formatHoldingHorizon(symbol.decision.holding_horizon))],
        ["失效条件", escapeHtml(symbol.decision.invalidation_condition)],
        ["历史观点状态", escapeHtml(formatPreviousViewStatus(symbol.previous_view_status))],
        ["MA20 / 偏离", `${{formatValue(technical.ma20)}} / ${{formatPercentValue(technical.distance_to_ma20_pct)}}`],
        ["量比 / 换手率", `${{formatValue(technical.vol_ratio)}} / ${{formatValue(technical.turnover_rate)}}`],
        ["近5日高低", `${{formatValue(technical.recent_5d_low)}} - ${{formatValue(technical.recent_5d_high)}}`],
        ["主力资金", formatSignedAmount(moneyflow.net_mf_amount)],
        ["大单 / 超大单", `${{formatSignedAmount(largeOrderNet)}} / ${{formatSignedAmount(superLargeOrderNet)}}`],
        ["中长线结构", escapeHtml(formatTrendBias(trendBreakdown.mid_term)) + "\\n" + escapeHtml(trendBreakdown.mid_term_reason || "") + "\\n\\n" + escapeHtml(formatTrendBias(trendBreakdown.long_term)) + "\\n" + escapeHtml(trendBreakdown.long_term_reason || "")]
      ].map(([label, value]) => `<div class="metric"><div class="label">${{label}}</div><div class="value">${{value}}</div></div>`).join("");
    }}

    async function savePositionStatus(nextStatus) {{
      if (!IS_INTERACTIVE || !positionStatusSelect) return;
      const previousStatus = currentPositionStatus;
      positionStatusSelect.disabled = true;
      if (detailActionStatus) {{
        detailActionStatus.textContent = "正在保存持仓状态...";
      }}
      try {{
        const response = await fetch("/positions/{ts_code}", {{
          method: "PUT",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ position_status: nextStatus }})
        }});
        const payload = await response.json();
        if (!response.ok) {{
          throw new Error(payload.detail || "保存失败");
        }}
        currentPositionStatus = payload.position_status || nextStatus;
        currentStockPool = payload.stock_pool || currentStockPool;
        positionStatusSelect.value = currentPositionStatus;
        if (detailActionStatus) {{
          detailActionStatus.textContent = currentPositionStatus === "holding"
            ? "已标记为持有，并自动加入默认股票池。"
            : `持仓状态已保存为${{currentPositionStatus === "holding" ? "已持有" : "未持有"}}。`;
        }}
      }} catch (error) {{
        positionStatusSelect.value = previousStatus;
        if (detailActionStatus) {{
          detailActionStatus.textContent = `保存失败：${{error.message}}`;
        }}
        throw error;
      }} finally {{
        positionStatusSelect.disabled = false;
      }}
    }}

    async function reanalyzeCurrentSymbol() {{
      if (!IS_INTERACTIVE || !reanalyzeSymbolButton) return;
      reanalyzeSymbolButton.disabled = true;
      if (positionStatusSelect) positionStatusSelect.disabled = true;
      if (detailActionStatus) {{
        detailActionStatus.textContent = "正在重新分析当前股票...";
      }}
      try {{
        if (positionStatusSelect && positionStatusSelect.value !== currentPositionStatus) {{
          await savePositionStatus(positionStatusSelect.value);
        }}
        const response = await fetch("/analyze", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            phase: "review",
            stock_pool: ["{ts_code}"],
            notify: false
          }})
        }});
        const payload = await response.json();
        if (!response.ok) {{
          throw new Error(payload.detail || "重新分析失败");
        }}
        await loadDetail(true);
        const signal = payload.reports && payload.reports[0] && payload.reports[0].decision ? payload.reports[0].decision.signal : null;
        if (detailActionStatus) {{
          detailActionStatus.textContent = signal ? `重新分析完成，最新信号：${{formatSignal(signal)}}。` : "重新分析完成。";
        }}
      }} catch (error) {{
        if (detailActionStatus) {{
          detailActionStatus.textContent = `重新分析失败：${{error.message}}`;
        }}
      }} finally {{
        reanalyzeSymbolButton.disabled = false;
        if (positionStatusSelect) positionStatusSelect.disabled = false;
      }}
    }}

    async function loadDetail(forceRefresh = false) {{
      const payload = (!forceRefresh && initialDetailPayload) || await (async () => {{
        const response = await fetch("/stocks/{ts_code}/data");
        if (response.status === 404) {{
          heroCard.innerHTML = '<div class="empty">该股票暂无历史记录。先触发一次分析再查看详情页。</div>';
          return null;
        }}
        return response.json();
      }})();
      if (!payload) {{
        return;
      }}
      initialDetailPayload = null;
      const symbol = payload.symbol;
      generatedAt.textContent = payload.generated_at ? `最近更新：${{payload.generated_at}}` : "暂无数据";
      heroCard.innerHTML = renderHero(symbol);
      renderFullAnalysis(symbol);
      renderMetrics(symbol);
      detailHistory.innerHTML = renderHistoryTimeline(symbol.history);
      detailEvidence.innerHTML = renderList(symbol.decision.evidence || []);
      detailRisks.innerHTML = renderList(symbol.memory.key_risks || []);
      detailOpenclawStatus.innerHTML = renderAutomationStatus(payload.openclaw_status);
      currentPositionStatus = payload.position_status || "watching";
      currentStockPool = payload.default_stock_pool || [];
      if (positionStatusSelect) {{
        positionStatusSelect.value = currentPositionStatus;
      }}
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
        if (detailFullAnalysis) {{
          detailFullAnalysis.innerHTML = '<div class="empty">暂无分析数据</div>';
        }}
        detailHistory.innerHTML = '<div class="empty">暂无历史记录</div>';
        detailMetrics.innerHTML = "";
        detailEvidence.innerHTML = '<div class="empty">暂无数据</div>';
        detailRisks.innerHTML = '<div class="empty">暂无数据</div>';
      }} catch (error) {{
        detailGeneratedAt.textContent = `清空失败：${{error.message}}`;
      }} finally {{
        clearSymbolDataButton.disabled = false;
      }}
    }}

    if (IS_INTERACTIVE && clearSymbolDataButton) {{
      clearSymbolDataButton.addEventListener("click", clearCurrentSymbol);
    }}
    if (IS_INTERACTIVE && positionStatusSelect) {{
      positionStatusSelect.addEventListener("change", event => {{
        savePositionStatus(event.target.value).catch(() => {{}});
      }});
    }}
    if (IS_INTERACTIVE && reanalyzeSymbolButton) {{
      reanalyzeSymbolButton.addEventListener("click", reanalyzeCurrentSymbol);
    }}
    loadDetail().catch(error => {{
      heroCard.innerHTML = `<div class="empty">加载单股详情失败：${{escapeHtml(error.message)}}</div>`;
    }});
    """
