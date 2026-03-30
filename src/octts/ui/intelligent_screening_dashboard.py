from __future__ import annotations

from html import escape
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _lookup(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _render_report_content(content: Any) -> str:
    text = _safe_text(content, "暂无内容")
    lines = [line.rstrip() for line in text.splitlines()]
    if not any(line.strip() for line in lines):
        return "暂无内容"
    return "<br>".join(escape(line) if line.strip() else "" for line in lines)


def _localize_technical_signal(value: Any) -> str:
    text = _safe_text(value)
    if not text:
        return "待确认"
    signal_labels = {
        "bullish": "看多",
        "neutral": "中性",
        "bearish": "看空",
        "bullish_rising": "多头增强",
        "bearish_improving": "空头钝化",
        "improving": "趋势改善",
        "mixed": "震荡整理",
        "weak_bearish": "偏弱看空",
    }
    lowered = text.lower()
    return signal_labels.get(lowered, text)


def _format_datetime_minute(value: Any) -> str:
    text = _safe_text(value)
    if not text:
        return "暂无最近运行时间"
    return escape(text[:16].replace("T", " "))


def _format_metric_value(value: Any, *, percent: bool = False) -> str:
    if value is None or value == "":
        return "--"
    if percent:
        return f"{_safe_float(value, 0.0) * 100:.1f}%"
    return escape(_safe_text(value, "--"))


def _render_summary_card(title: str, value: str, note: str = "") -> str:
    note_html = f'<div class="summary-note">{escape(note)}</div>' if note else ""
    return (
        '<div class="summary-card">'
        f'<div class="summary-label">{escape(title)}</div>'
        f'<div class="summary-value">{value}</div>'
        f'{note_html}'
        '</div>'
    )


def _format_industry_flow_text(item: Dict[str, Any]) -> str:
    bias = _safe_text(item.get("industry_flow_bias"), "中性")
    return f"板块氛围：{bias}"


def _build_stock_detail_href(ts_code: Any, *, prefix: str, suffix: str) -> str:
    return f"{prefix}{quote(_safe_text(ts_code))}{suffix}"


def _render_stock_cards(
    items: List[Dict[str, Any]],
    empty_text: str,
    *,
    stock_detail_href_prefix: str,
    stock_detail_href_suffix: str,
) -> str:
    if not items:
        return f'<div class="empty-state">{escape(empty_text)}</div>'
    html = []
    for item in items:
        code = escape(_safe_text(item.get("ts_code")))
        stock_href = escape(
            _build_stock_detail_href(
                item.get("ts_code"),
                prefix=stock_detail_href_prefix,
                suffix=stock_detail_href_suffix,
            )
        )
        name = escape(_safe_text(item.get("name"), _safe_text(item.get("ts_code"))))
        recommendation = escape(_safe_text(item.get("recommendation_text", item.get("recommendation")), "暂无推荐理由"))
        score = _safe_float(item.get("recommendation_score", item.get("recommend_score", item.get("score"))), 0.0)
        source_tag = escape(_safe_text(item.get("source_tag"), "候选池"))
        overall_score = _safe_float(item.get("overall_score", item.get("priority_score", item.get("score"))), 0.0)
        confidence = _safe_float(item.get("display_confidence", item.get("overall_confidence", item.get("ai_confidence"))), 0.0)
        technical_signal = escape(_localize_technical_signal(item.get("technical_signal")))
        industry_flow_text = escape(_format_industry_flow_text(item))
        html.append(
            f'''
            <div class="stock-item" onclick="window.location.href='{stock_href}'">
                <div class="stock-header"><div><span class="stock-code">{code}</span><span class="stock-name">{name}</span></div><span class="score-badge">推荐分数 {score:.1f}</span></div>
                <div class="detail-grid" style="margin-top:12px;">
                    <div class="detail-panel"><div class="detail-label">来源</div><div class="detail-value">{source_tag}</div></div>
                    <div class="detail-panel"><div class="detail-label">综合分</div><div class="detail-value">{overall_score:.1f}</div></div>
                    <div class="detail-panel"><div class="detail-label">置信度</div><div class="detail-value">{confidence:.2f}</div></div>
                </div>
                <div class="detail-value"><strong>技术信号：</strong>{technical_signal}</div>
                <div class="detail-value"><strong>{industry_flow_text}</strong></div>
                <div class="detail-value">{recommendation}</div>
            </div>
            '''
        )
    return "".join(html)


def _merge_card_items(primary_items: List[Dict[str, Any]], report_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    report_map = {item.get("ts_code"): item for item in report_items if isinstance(item, dict) and item.get("ts_code")}
    merged: List[Dict[str, Any]] = []
    for item in primary_items:
        if not isinstance(item, dict):
            continue
        code = _safe_text(item.get("ts_code"))
        report_item = report_map.get(code, {})
        combined = dict(item)
        for key, value in report_item.items():
            if key in {"ts_code", "source_tag", "recommendation_score", "recommend_rank", "overall_score", "priority_score"}:
                continue
            if value not in (None, "", [], {}):
                combined[key] = value
        action = dict(item.get("action_plan") or {})
        action.update(report_item.get("action_plan") or {})
        combined["action_plan"] = action
        merged.append(combined)
    return merged


def _render_today_top_cards(
    items: List[Dict[str, Any]],
    empty_text: str,
    *,
    stock_detail_href_prefix: str,
    stock_detail_href_suffix: str,
) -> str:
    if not items:
        return f'<div class="empty-state">{escape(empty_text)}</div>'
    html = []
    for item in items:
        action = item.get("action_plan") or {}
        code = escape(_safe_text(item.get("ts_code")))
        stock_href = escape(
            _build_stock_detail_href(
                item.get("ts_code"),
                prefix=stock_detail_href_prefix,
                suffix=stock_detail_href_suffix,
            )
        )
        name = escape(_safe_text(item.get("name"), _safe_text(item.get("ts_code"))))
        technical_signal = escape(_localize_technical_signal(item.get("technical_signal")))
        overall_assessment = escape(_safe_text(item.get("overall_assessment") or item.get("summary") or item.get("recommendation_text"), "暂无说明"))
        recommendation = escape(_safe_text(item.get("recommendation_text", item.get("recommendation")), "暂无推荐理由"))
        score = _safe_float(item.get("recommendation_score", item.get("recommend_score", item.get("score"))), 0.0)
        overall_score = _safe_float(item.get("overall_score", item.get("priority_score", item.get("score"))), 0.0)
        confidence = _safe_float(item.get("display_confidence", item.get("overall_confidence", item.get("ai_confidence"))), 0.0)
        industry_flow_text = escape(_format_industry_flow_text(item))
        html.append(
            f'''
            <div class="stock-item" onclick="window.location.href='{stock_href}'">
                <div class="stock-header"><div><span class="stock-code">{code}</span><span class="stock-name">{name}</span></div><span class="score-badge">先看买点</span></div>
                <div class="detail-grid" style="margin-top:12px;">
                    <div class="detail-panel"><div class="detail-label">买入区间</div><div class="detail-value">{escape(_safe_text(action.get('entry_zone'), '等待触发'))}</div></div>
                    <div class="detail-panel"><div class="detail-label">止损位</div><div class="detail-value">{escape(_safe_text(action.get('stop_loss'), '待观察'))}</div></div>
                    <div class="detail-panel"><div class="detail-label">第一止盈位</div><div class="detail-value">{escape(_safe_text(action.get('take_profit'), '待观察'))}</div></div>
                    <div class="detail-panel"><div class="detail-label">短期节奏</div><div class="detail-value">{escape(_safe_text(action.get('holding_horizon'), '1-5个交易日'))}</div></div>
                    <div class="detail-panel"><div class="detail-label">失效条件</div><div class="detail-value">{escape(_safe_text(action.get('invalid_condition'), '走势失真时离场'))}</div></div>
                    <div class="detail-panel"><div class="detail-label">辅助指标</div><div class="detail-value">推荐分 {score:.1f}（执行优先级）｜ 综合分 {overall_score:.1f}（AI基础判断）｜ 置信度 {confidence:.2f}</div></div>
                </div>
                <div class="detail-value"><strong>技术信号：</strong>{technical_signal}</div>
                <div class="detail-value"><strong>{industry_flow_text}</strong></div>
                <div class="detail-value"><strong>综合评估：</strong>{overall_assessment}</div>
                <div class="detail-value">{recommendation}</div>
            </div>
            '''
        )
    return "".join(html)


def _render_review_cards(
    items: List[Dict[str, Any]],
    empty_text: str,
    *,
    stock_detail_href_prefix: str,
    stock_detail_href_suffix: str,
) -> str:
    if not items:
        return f'<div class="empty-state">{escape(empty_text)}</div>'
    html = []
    for item in items:
        action = item.get("action_plan") or {}
        code = escape(_safe_text(item.get("ts_code")))
        stock_href = escape(
            _build_stock_detail_href(
                item.get("ts_code"),
                prefix=stock_detail_href_prefix,
                suffix=stock_detail_href_suffix,
            )
        )
        name = escape(_safe_text(item.get("name"), _safe_text(item.get("ts_code"))))
        verdict = escape(_safe_text(item.get("today_verdict"), _safe_text(item.get("status", item.get("review_status")), "待复评")))
        status = escape(_safe_text(item.get("status", item.get("review_status")), "待复评"))
        reason = escape(_safe_text(item.get("analysis") or item.get("yesterday_conclusion") or item.get("overall_assessment"), "暂无复盘说明"))
        risk = escape(_safe_text('；'.join(item.get('risk_warnings') or []), "暂无额外风险提示"))
        score_change_raw = item.get("score_change")
        score_change = _safe_float(score_change_raw, 0.0) if score_change_raw not in (None, "") else None
        score_change_text = f"{score_change:+.1f}" if score_change is not None else "--"
        industry_flow_text = escape(_format_industry_flow_text(item))
        html.append(
            f'''
            <div class="stock-item" onclick="window.location.href='{stock_href}'">
                <div class="stock-header"><div><span class="stock-code">{code}</span><span class="stock-name">{name}</span></div><span class="score-badge">昨日延续</span></div>
                <div class="detail-grid" style="margin-top:12px;">
                    <div class="detail-panel"><div class="detail-label">今日结论</div><div class="detail-value">{verdict}</div></div>
                    <div class="detail-panel"><div class="detail-label">跟踪状态</div><div class="detail-value">{status}</div></div>
                    <div class="detail-panel"><div class="detail-label">分数变化</div><div class="detail-value">{score_change_text}</div></div>
                    <div class="detail-panel"><div class="detail-label">继续持有看哪里止盈</div><div class="detail-value">{escape(_safe_text(action.get('take_profit'), '结合盘中强弱分批止盈'))}</div></div>
                    <div class="detail-panel"><div class="detail-label">转弱止损/离场</div><div class="detail-value">{escape(_safe_text(action.get('stop_loss'), '跌破关键位止损'))}</div></div>
                    <div class="detail-panel"><div class="detail-label">离场触发条件</div><div class="detail-value">{escape(_safe_text(action.get('invalid_condition'), '走势转弱时退出'))}</div></div>
                    <div class="detail-panel"><div class="detail-label">持有节奏</div><div class="detail-value">{escape(_safe_text(action.get('holding_horizon'), '1-5个交易日'))}</div></div>
                </div>
                <div class="detail-value"><strong>为什么继续/退出：</strong>{reason}</div>
                <div class="detail-value"><strong>{industry_flow_text}</strong></div>
                <div class="detail-value"><strong>昨日结论：</strong>{escape(_safe_text(item.get('yesterday_conclusion'), '昨日入选 Top3'))}</div>
                <div class="detail-value"><strong>风险提示：</strong>{risk}</div>
            </div>
            '''
        )
    return "".join(html)


def _render_focus_analysis(
    items: List[Dict[str, Any]],
    ai_analyses: Dict[str, Any],
    news_clusters: List[Any],
    *,
    stock_detail_href_prefix: str,
    stock_detail_href_suffix: str,
) -> str:
    if not items:
        return '<div class="empty-state">暂无重点个股分析</div>'
    html = []
    for stock in items:
        code = _safe_text(stock.get("ts_code"))
        stock_href = escape(
            _build_stock_detail_href(
                stock.get("ts_code"),
                prefix=stock_detail_href_prefix,
                suffix=stock_detail_href_suffix,
            )
        )
        payload = ai_analyses.get(code, {}) or {}
        sentiment_summary = _safe_text(payload.get("sentiment_summary"))
        news_fragments: List[str] = []
        news_summary = _safe_text(payload.get("news_summary"))
        if news_summary:
            news_fragments.append(news_summary)
        for cluster in news_clusters or []:
            if not isinstance(cluster, dict):
                continue
            if code not in (cluster.get("key_stocks") or []):
                continue
            theme = _safe_text(cluster.get("theme"))
            summary = _safe_text(cluster.get("summary"))
            if theme and summary:
                news_fragments.append(f"{theme}：{summary}")
            elif theme:
                news_fragments.append(theme)
            elif summary:
                news_fragments.append(summary)
        news_block = escape("<br>".join(dict.fromkeys(news_fragments)) or "暂无新闻/主题解读")
        key_points = escape("、".join(payload.get("key_points") or stock.get("core_highlights") or []) or "暂无")
        html.append(
            f'''
            <div class="stock-item focus-analysis-item" onclick="window.location.href='/stocks/{escape(code)}'">
                <div class="stock-header"><div><span class="stock-code">{escape(code)}</span><span class="stock-name">{escape(_safe_text(stock.get('name') or payload.get('name'), code))}</span></div><span class="score-badge">推荐分 {_safe_float(stock.get('recommendation_score', payload.get('recommendation_score')), 0.0):.1f}</span></div>
                <div class="detail-grid" style="margin-top:12px;">
                    <div class="detail-panel"><div class="detail-label">技术信号</div><div class="detail-value">{escape(_localize_technical_signal(stock.get('technical_signal') or payload.get('technical_signal')))}</div></div>
                    <div class="detail-panel"><div class="detail-label">推荐意见</div><div class="detail-value">{escape(_safe_text(stock.get('recommendation_text') or payload.get('recommendation') or payload.get('final_decision'), '暂无'))}</div></div>
                    <div class="detail-panel"><div class="detail-label">置信度</div><div class="detail-value">{_safe_float(stock.get('display_confidence', payload.get('overall_confidence', payload.get('ai_confidence'))), 0.0):.2f}</div></div>
                </div>
                <div class="analysis-grid">
                    <div class="detail-panel"><div class="detail-label">分析摘要</div><div class="detail-value">{escape(_safe_text(stock.get('overall_assessment') or payload.get('summary'), '暂无分析摘要'))}</div></div>
                    <div class="detail-panel"><div class="detail-label">技术面</div><div class="detail-value">{escape(_safe_text(payload.get('technical_summary'), '暂无'))}</div></div>
                    <div class="detail-panel"><div class="detail-label">基本面</div><div class="detail-value">{escape(_safe_text(payload.get('fundamental_summary'), '暂无'))}</div></div>
                    <div class="detail-panel"><div class="detail-label">情绪面</div><div class="detail-value">{escape(_safe_text(sentiment_summary, '暂无'))}</div></div>
                    <div class="detail-panel detail-panel-news"><div class="detail-label">新闻 / 主题</div><div class="detail-value">{news_block}</div></div>
                    <div class="detail-panel"><div class="detail-label">关键点</div><div class="detail-value">{key_points}</div></div>
                </div>
            </div>
            '''
        )
    return "".join(html)


def render_report_tab(report: Any) -> str:
    if not report:
        return '<p style="text-align: center; color: #86868b;">暂无报告数据</p>'

    title = escape(str(_lookup(report, "title", "智能报告")))
    summary = _render_report_content(_lookup(report, "summary", "暂无摘要"))
    sections = _lookup(report, "sections", []) or []
    blocks = _lookup(report, "blocks", {}) or {}

    html = f"""
    <h2 style="margin-bottom: 24px;">{title}</h2>
    <div class="report-section">
        <div class="report-title">摘要</div>
        <div class="report-content">{summary}</div>
    </div>
    """
    if blocks:
        return html + _render_structured_report_blocks(blocks)
    for section in sections:
        section_title = escape(str(_lookup(section, "title", "未命名章节")))
        section_content = _render_report_content(_lookup(section, "content", ""))
        html += f"""
        <div class="report-section">
            <div class="report-title">{section_title}</div>
            <div class="report-content">{section_content}</div>
        </div>
        """
    return html


def _render_news_cluster_block(clusters: List[Dict[str, Any]]) -> str:
    if not clusters:
        return ''
    html = '<div class="report-section"><div class="report-title">主题新闻聚焦</div><div class="news-cluster-list">'
    for cluster in clusters:
        summary = escape(_safe_text(cluster.get("summary"), "暂无主题摘要"))
        theme = escape(_safe_text(cluster.get("theme"), "未命名主题"))
        key_stocks = "、".join(cluster.get("key_stocks") or []) or "暂无"
        importance = cluster.get("importance")
        try:
            importance_text = f"{float(importance):.2f}"
        except Exception:
            importance_text = _safe_text(importance, "--") or "--"
        html += f"""
        <div class="news-cluster-card">
            <div class="news-cluster-head">
                <div class="news-cluster-title">{theme}</div>
                <span class="tag">重要度 {escape(importance_text)}</span>
            </div>
            <div class="news-cluster-meta"><strong>主题摘要：</strong>{summary}</div>
            <div class="news-cluster-meta"><strong>关联个股：</strong>{escape(key_stocks)}</div>
        </div>
        """
    return html + "</div></div>"


def _render_structured_report_blocks(blocks: Dict[str, Any]) -> str:
    return (
        _render_news_cluster_block(blocks.get("news_clusters") or [])
        + _render_focus_stock_block(blocks.get("focus_stocks") or [])
        + _render_review_block(blocks.get("yesterday_reviews") or [])
        + _render_comparison_block(blocks.get("comparison") or {})
        + _render_overall_action_block(blocks.get("overall_action") or {})
    )


def _render_focus_stock_block(items: List[Dict[str, Any]]) -> str:
    if not items:
        return '<div class="report-section"><div class="report-title">今日 Top3 重点点评</div><div class="report-content">暂无数据</div></div>'
    html = '<div class="report-section"><div class="report-title">今日 Top3 重点点评</div>'
    for item in items:
        action = item.get("action_plan") or {}
        highlights = "<br>".join(escape(_safe_text(v)) for v in (item.get("core_highlights") or [])) or "暂无"
        risks = "<br>".join(escape(_safe_text(v)) for v in (item.get("risk_warnings") or [])) or "暂无"
        html += f"""
        <div class="stock-item">
            <div class="stock-header"><div><span class="stock-code">{escape(_safe_text(item.get('ts_code')))}</span><span class="stock-name">{escape(_safe_text(item.get('name'), _safe_text(item.get('ts_code'))))}</span></div><span class="score-badge">{_safe_float(item.get('recommendation_score'), 0.0):.1f}分</span></div>
            <div class="detail-grid" style="margin-top:12px;">
                <div class="detail-panel"><div class="detail-label">综合分</div><div class="detail-value">{_safe_float(item.get('overall_score'), 0.0):.1f}</div></div>
                <div class="detail-panel"><div class="detail-label">置信度</div><div class="detail-value">{_safe_float(item.get('display_confidence', item.get('overall_confidence')), 0.0):.2f}</div></div>
                <div class="detail-panel"><div class="detail-label">来源</div><div class="detail-value">{escape(_safe_text(item.get('source_tag'), '--'))}</div></div>
            </div>
            <div class="report-content"><strong>核心亮点：</strong><br>{highlights}</div>
            <div class="report-content"><strong>风险提示：</strong><br>{risks}</div>
            <div class="report-content"><strong>综合评价：</strong>{escape(_safe_text(item.get('overall_assessment'), _safe_text(item.get('summary'), '暂无')))}</div>
            <div class="report-content"><strong>短线建议：</strong>{escape(_safe_text(action.get('action_bias'), '观察'))} ｜ 入场区间 {escape(_safe_text(action.get('entry_zone'), '待观察'))} ｜ 止盈 {escape(_safe_text(action.get('take_profit'), '待观察'))} ｜ 止损 {escape(_safe_text(action.get('stop_loss'), '待观察'))}</div>
        </div>
        """
    return html + "</div>"


def _render_review_block(items: List[Dict[str, Any]]) -> str:
    if not items:
        return '<div class="report-section"><div class="report-title">昨日 Top3 今日复盘</div><div class="report-content">暂无数据</div></div>'
    html = '<div class="report-section"><div class="report-title">昨日 Top3 今日复盘</div>'
    for item in items:
        html += f"""
        <div class="stock-item">
            <div class="stock-header"><div><span class="stock-code">{escape(_safe_text(item.get('ts_code')))}</span><span class="stock-name">{escape(_safe_text(item.get('name'), _safe_text(item.get('ts_code'))))}</span></div><span class="tag">{escape(_safe_text(item.get('status', item.get('today_verdict')), '观察'))}</span></div>
            <div class="report-content"><strong>昨日结论：</strong>{escape(_safe_text(item.get('yesterday_conclusion'), '暂无'))}</div>
            <div class="report-content"><strong>今日判断：</strong>{escape(_safe_text(item.get('today_verdict', item.get('status')), '暂无'))}</div>
            <div class="report-content"><strong>复盘说明：</strong>{escape(_safe_text(item.get('analysis'), _safe_text(item.get('absence_reason'), '暂无')))}</div>
            <div class="report-content"><strong>失误候选：</strong>{escape('、'.join(item.get('miss_reason_candidates') or []) or '暂无')}</div>
            <div class="report-content"><strong>缺失因子：</strong>{escape('、'.join(item.get('missing_factor_candidates') or []) or '暂无')}</div>
        </div>
        """
    return html + "</div>"


def _render_comparison_block(comparison: Dict[str, Any]) -> str:
    return f"""
    <div class="report-section">
        <div class="report-title">重点股票横向比较</div>
        <div class="report-content">
            <div><strong>基本面排序：</strong>{escape(' > '.join(comparison.get('basic_rank') or []) or '暂无')}</div>
            <div><strong>技术面排序：</strong>{escape(' > '.join(comparison.get('technical_rank') or []) or '暂无')}</div>
            <div><strong>风险排序：</strong>{escape(' > '.join(comparison.get('risk_rank') or []) or '暂无')}</div>
            <div><strong>短线交易性排序：</strong>{escape(' > '.join(comparison.get('trading_rank') or []) or '暂无')}</div>
            <div><strong>最适合短线：</strong>{escape(_safe_text(comparison.get('best_short_term'), '暂无'))}</div>
            <div><strong>最稳健：</strong>{escape(_safe_text(comparison.get('most_robust'), '暂无'))}</div>
            <div><strong>风险最高：</strong>{escape(_safe_text(comparison.get('highest_risk'), '暂无'))}</div>
        </div>
    </div>
    """


def _render_overall_action_block(overall: Dict[str, Any]) -> str:
    items = ''.join(f'<li>{escape(_safe_text(item))}</li>' for item in (overall.get('action_items') or [])) or '<li>暂无</li>'
    return f"""
    <div class="report-section">
        <div class="report-title">今日整体操作建议 / 风险总览</div>
        <div class="report-content">
            <div><strong>{escape(_safe_text(overall.get('headline'), '今日整体建议'))}</strong></div>
            <div><strong>市场判断：</strong>{escape(_safe_text(overall.get('market_view'), '暂无'))}</div>
            <div><strong>风险总览：</strong>{escape(_safe_text(overall.get('risk_summary'), '暂无'))}</div>
            <div style="margin-top:8px;"><strong>操作建议：</strong><ul>{items}</ul></div>
        </div>
    </div>
    """


def _build_enabled_strategies_list(methodology: Dict[str, Any]) -> str:
    strategies = methodology.get("strategies") or []
    if not strategies:
        return '<div class="empty-state">暂无已启用策略说明</div>'
    return '<ul class="bullet-list">' + ''.join(
        f'<li><strong>{escape(_safe_text(item.get("name"), "未命名策略"))}</strong>：{escape(_safe_text(item.get("description"), "暂无说明"))}</li>'
        for item in strategies[:10]
        if isinstance(item, dict)
    ) + '</ul>'


def _build_methodology_list(methodology: Dict[str, Any]) -> str:
    items = (methodology.get("candidate_selection") or []) + (methodology.get("score_formula") or []) + (methodology.get("tracking_metrics") or [])
    if not items:
        return '<div class="empty-state">暂无方法说明</div>'
    return '<ul class="bullet-list">' + ''.join(f'<li>{escape(_safe_text(item))}</li>' for item in items[:10]) + '</ul>'


def render_intelligent_screening_dashboard(
    *,
    screening_results: Dict[str, Any],
    recommendation_pool: Dict[str, Any],
    ai_analyses: Dict[str, Any],
    news_clusters: List[Any],
    intelligent_report: Any,
    recommendation_summary: Dict[str, Any],
    recommendation_methodology: Dict[str, Any],
    generated_at: Any = None,
    dashboard_href: str = "/dashboard",
    backtest_href: Optional[str] = "/backtest-page",
    refresh_href: str = "/intelligent-screening",
    jobs_api_base: str = "/screen/intelligent/jobs",
    autorun_enabled: bool = True,
    active_tab: str = "overview",
    stock_detail_href_prefix: str = "/stocks/",
    stock_detail_href_suffix: str = "",
) -> str:
    screening_results = screening_results or {}
    recommendation_pool = recommendation_pool or {}
    ai_analyses = ai_analyses or {}
    recommendation_summary = recommendation_summary or {}
    recommendation_methodology = recommendation_methodology or {}
    active_tab = active_tab if active_tab in {"overview", "report", "recommendations"} else "overview"
    report_html = render_report_tab(intelligent_report)
    generated_label = _format_datetime_minute(generated_at)
    stats = recommendation_summary.get("stats") or {}

    frontlist = recommendation_pool.get("frontlist") or []
    frontlist = sorted(
        frontlist,
        key=lambda item: (
            item.get("recommend_rank") or 9999,
            -_safe_float(item.get("recommendation_score", item.get("recommend_score", item.get("score"))), 0.0),
        ),
    )
    today_top = recommendation_pool.get("today_top") or [item for item in frontlist if item.get("source_tag") == "今日Top3"]
    today_top = sorted(
        today_top,
        key=lambda item: -_safe_float(item.get("recommendation_score", item.get("recommend_score", item.get("score"))), 0.0),
    )[:3]
    continuations = recommendation_pool.get("yesterday_continuations") or [item for item in frontlist if item.get("source_tag") == "昨日延续"]
    report_blocks = _lookup(intelligent_report, "blocks", {}) or {}
    today_top = _merge_card_items(today_top, report_blocks.get("focus_stocks") or [])
    continuations = _merge_card_items(continuations, report_blocks.get("yesterday_reviews") or [])

    strategy_count = screening_results.get("strategy_count")
    if strategy_count is None:
        strategy_count = recommendation_methodology.get("strategy_count")
    strategy_note = ""
    if screening_results.get("strategy_count") is None:
        strategy_note = "暂无本次运行快照，回退展示当前启用策略数"

    total_stocks = screening_results.get("total_stocks")
    total_stocks_value = _format_metric_value(total_stocks) if total_stocks is not None else "暂无本次运行数据"
    total_stocks_note = "本次所有策略命中的股票条目总数"
    if total_stocks is None:
        total_stocks_note = "未找到本次运行快照，因此不显示误导性的 0"

    win_rate_value = "--"
    tracked_count = stats.get("window_count")
    win_rate_note = "5日胜率为历史已验证推荐样本中 return_5d > 0 的占比"
    if stats.get("win_rate_5d") is not None and tracked_count:
        win_rate_value = _format_metric_value(stats.get("win_rate_5d"), percent=True)
    else:
        win_rate_note = "仅在开启数据库且存在已验证样本时展示"

    tab_labels = {
        "overview": "总览",
        "report": "AI智能早报",
        "recommendations": "推荐列表",
    }
    tab_links = []
    for tab_key in ("overview", "report", "recommendations"):
        href = f"{refresh_href}?{urlencode({'tab': tab_key})}"
        active_class = " is-active" if tab_key == active_tab else ""
        tab_links.append(
            f'<a class="tab-link{active_class}" href="{escape(href)}">{escape(tab_labels[tab_key])}</a>'
        )
    tabs_html = "".join(tab_links)

    overview_sections = f"""
      <section class="top-grid">
        <section class="panel panel-today">
          <div class="section-title">今日 Top3</div>
          <div class="subtle" style="margin-bottom:12px;">展示今日优先关注标的的买点、止损和短期节奏。</div>
          {_render_today_top_cards(today_top, '暂无今日 Top3', stock_detail_href_prefix=stock_detail_href_prefix, stock_detail_href_suffix=stock_detail_href_suffix)}
        </section>

        <section class="panel panel-review">
          <div class="section-title">昨日 Top3 今日复盘 / 昨日延续</div>
          <div class="subtle" style="margin-bottom:12px;">展示上一个交易日重点标的的今日复盘结论，以及继续持有还是离场的触发条件。</div>
          {_render_review_cards(continuations, '暂无昨日延续标的', stock_detail_href_prefix=stock_detail_href_prefix, stock_detail_href_suffix=stock_detail_href_suffix)}
        </section>
      </section>

      <section class="panel panel-focus">
        <div class="section-title">重点个股分析</div>
        <div class="subtle" style="margin-bottom:12px;">仅分析今日 Top3，并补充更完整的新闻与主题解读。</div>
        {_render_focus_analysis(today_top, ai_analyses, news_clusters, stock_detail_href_prefix=stock_detail_href_prefix, stock_detail_href_suffix=stock_detail_href_suffix)}
      </section>

      <section class="panel">
        <div class="section-title">历史统计摘要</div>
        <div class="metrics-grid">
          <div class="metric-card"><div class="mini-label">已验证样本数</div><div class="summary-value">{escape(_safe_text(tracked_count, '--'))}</div></div>
          <div class="metric-card"><div class="mini-label">平均 5 日收益</div><div class="summary-value">{escape(_safe_text(stats.get('average_return_5d'), '--'))}</div></div>
          <div class="metric-card"><div class="mini-label">新闻主题数</div><div class="summary-value">{escape(_safe_text(len(news_clusters), '0'))}</div></div>
        </div>
      </section>

      <section class="panel">
        <div class="section-title">方法说明</div>
        <div class="subtle" style="margin-bottom:12px;">当前启用策略数：{escape(_safe_text(recommendation_methodology.get('strategy_count'), '--'))}</div>
        <div class="report-content" style="margin-bottom:12px;"><strong>当前启用策略</strong></div>
        {_build_enabled_strategies_list(recommendation_methodology)}
        <div class="report-content" style="margin:14px 0 12px 0;"><strong>筛选与评分口径</strong></div>
        {_build_methodology_list(recommendation_methodology)}
      </section>
    """

    report_section = f"""
      <section class="panel panel-report">
        <div class="section-title">AI智能早报</div>
        <div class="subtle" style="margin-bottom:12px;">展示今日 Top3 点评、昨日复盘、横向比较和整体建议。</div>
        {report_html}
      </section>
    """

    recommendations_section = f"""
      <section class="panel panel-list">
        <div class="section-title">推荐列表</div>
        {_render_stock_cards(frontlist, '暂无推荐结果', stock_detail_href_prefix=stock_detail_href_prefix, stock_detail_href_suffix=stock_detail_href_suffix)}
      </section>
    """

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>智能选股</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --bg-accent: #eef4ff;
      --panel: #ffffff;
      --panel-border: #d9e2ef;
      --panel-soft: #fffaf0;
      --panel-softer: #fffdf7;
      --surface-soft: #f8fbff;
      --surface-muted: #eef3f8;
      --text: #1f2937;
      --text-strong: #162033;
      --text-soft: #526074;
      --muted: #6b7a90;
      --accent: #2563eb;
      --accent-soft: #e7f0ff;
      --accent-border: #bfd5ff;
      --shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
      --today-accent: #3b82f6;
      --today-soft: #eef5ff;
      --review-accent: #16a34a;
      --review-soft: #edf9f0;
      --focus-accent: #d97706;
      --focus-soft: #fff4e8;
      --report-accent: #9333ea;
      --report-soft: #f7efff;
      --list-accent: #0f766e;
      --list-soft: #ecfbf8;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: linear-gradient(180deg, var(--bg-accent) 0%, var(--bg) 220px); color: var(--text); }}
    a {{ color: inherit; }}
    .shell {{ max-width: 1480px; margin: 0 auto; padding: 28px 24px 48px; }}
    .hero {{ display:flex; justify-content:space-between; align-items:flex-start; gap:24px; margin-bottom:24px; padding: 24px 28px; background: rgba(255,255,255,0.78); border: 1px solid rgba(217, 226, 239, 0.9); border-radius: 24px; box-shadow: var(--shadow); backdrop-filter: blur(10px); }}
    .hero h1 {{ margin:0 0 10px; font-size:34px; font-weight:800; color: var(--text-strong); }}
    .hero p {{ margin:0; color: var(--muted); line-height:1.7; max-width:880px; }}
    .hero-actions {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
    .summary-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:14px; margin:18px 0; }}
    .summary-card, .panel, .stock-item, .report-section, .metric-card {{ background: var(--panel); border:1px solid var(--panel-border); border-radius:20px; box-shadow: var(--shadow); }}
    .summary-card {{ padding:18px; background: linear-gradient(180deg, #ffffff 0%, #fdfefe 100%); }}
    .summary-label, .mini-label {{ color: var(--muted); font-size:12px; }}
    .detail-label {{ color: var(--text-soft); font-size:12px; }}
    .summary-value {{ margin-top:8px; font-size:24px; font-weight:800; color: var(--text-strong); }}
    .summary-note {{ margin-top:8px; color: var(--muted); font-size:12px; line-height:1.6; }}
    .stack {{ display:grid; gap:18px; }}
    .top-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:18px; }}
    .panel {{ padding:18px; position: relative; overflow: hidden; background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%); }}
    .panel::before {{ content:""; position:absolute; inset:0 auto 0 0; width:4px; border-radius:20px 0 0 20px; background: #d7e1ee; }}
    .panel-today {{ background: linear-gradient(180deg, var(--today-soft), #ffffff 42%); }}
    .panel-today::before {{ background: var(--today-accent); }}
    .panel-review {{ background: linear-gradient(180deg, var(--review-soft), #ffffff 42%); }}
    .panel-review::before {{ background: var(--review-accent); }}
    .panel-focus {{ background: linear-gradient(180deg, var(--focus-soft), #ffffff 42%); }}
    .panel-focus::before {{ background: var(--focus-accent); }}
    .panel-report {{ background: linear-gradient(180deg, var(--report-soft), #ffffff 42%); }}
    .panel-report::before {{ background: var(--report-accent); }}
    .panel-list {{ background: linear-gradient(180deg, var(--list-soft), #ffffff 42%); }}
    .panel-list::before {{ background: var(--list-accent); }}
    .section-title, .report-title {{ font-size:18px; font-weight:800; margin-bottom:12px; color: var(--text-strong); }}
    .subtle {{ color: var(--muted); line-height:1.7; }}
    .tab-bar {{ display:flex; gap:12px; flex-wrap:wrap; margin:18px 0 22px; padding:14px 16px; background: rgba(255, 255, 255, 0.9); border:1px solid var(--panel-border); border-radius:18px; box-shadow: var(--shadow); }}
    .tab-link {{ display:inline-flex; align-items:center; justify-content:center; padding:10px 16px; border-radius:999px; text-decoration:none; font-weight:700; color: var(--text-soft); background: #f6f9fc; border:1px solid #d8e3f0; }}
    .tab-link.is-active {{ color: #1d4ed8; background: #eaf2ff; border-color: #b9d1ff; box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.08); }}
    .primary-button, .ghost-link {{ appearance:none; border:1px solid var(--accent-border); color: var(--text-strong); background: #f5f9ff; padding:12px 18px; border-radius:14px; font-weight:700; cursor:pointer; text-decoration:none; box-shadow: 0 6px 16px rgba(37, 99, 235, 0.08); }}
    .primary-button {{ background: linear-gradient(135deg, #edf4ff, #dbeafe); color: #1d4ed8; border-color: #bfd5ff; }}
    .primary-button:hover, .ghost-link:hover {{ transform: translateY(-1px); box-shadow: 0 10px 20px rgba(37, 99, 235, 0.12); }}
    .ghost-link {{ background: #ffffff; border:1px solid var(--panel-border); color: var(--text); }}
    .stock-item {{ padding:16px; margin-bottom:12px; cursor:pointer; background: linear-gradient(180deg, #ffffff 0%, #fcfdff 100%); transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease; }}
    .stock-item:hover {{ transform: translateY(-1px); box-shadow: 0 14px 28px rgba(15, 23, 42, 0.1); border-color: #c8d7ea; }}
    .stock-header {{ display:flex; justify-content:space-between; gap:12px; align-items:center; }}
    .stock-code {{ font-weight:800; margin-right:8px; color: var(--text-strong); }}
    .stock-name {{ color: var(--text-soft); }}
    .score-badge, .tag {{ padding:6px 10px; border-radius:999px; background: var(--accent-soft); color:#1d4ed8; border: 1px solid var(--accent-border); font-size:12px; font-weight: 700; }}
    .detail-grid {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:12px; }}
    .analysis-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; margin-top:12px; }}
    .detail-panel {{ background: linear-gradient(180deg, var(--panel-softer), var(--panel-soft)); border-radius:14px; padding:12px; border:1px solid rgba(210, 186, 154, 0.28); }}
    .detail-panel-news {{ grid-column: 1 / -1; }}
    .detail-value {{ margin-top:6px; line-height:1.75; color: var(--text-strong); }}
    .detail-value strong {{ color: var(--text-strong); }}
    .report-section {{ padding:18px; margin-bottom:16px; background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%); }}
    .panel .report-section:last-child, .panel .stock-item:last-child {{ margin-bottom: 0; }}
    .report-content {{ line-height:1.85; color: var(--text-strong); background: linear-gradient(180deg, var(--panel-softer), var(--panel-soft)); border:1px solid rgba(210, 186, 154, 0.28); border-radius:14px; padding:12px 14px; margin-top:12px; }}
    .progress-wrap {{ margin:12px 0 0; }}
    .progress-bar {{ height:8px; border-radius:999px; background:#e5edf6; overflow:hidden; border: 1px solid #d6e1ee; }}
    .progress-fill {{ width:0%; height:100%; background:linear-gradient(90deg,#3b82f6,#10b981); }}
    .empty-state {{ color: var(--muted); padding:24px; text-align:center; background: var(--surface-soft); border: 1px dashed #cfdae8; border-radius: 16px; }}
    .metrics-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:14px; }}
    .metric-card {{ padding:16px; background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%); }}
    .bullet-list {{ margin:0; padding-left:20px; color: var(--text-strong); line-height:1.9; }}
    .news-cluster-list {{ display:grid; gap:14px; margin-top:16px; }}
    .news-cluster-card {{ background: linear-gradient(180deg, var(--panel-softer), var(--panel-soft)); border:1px solid rgba(210, 186, 154, 0.28); border-radius:16px; padding:14px 16px; }}
    .news-cluster-head {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap; }}
    .news-cluster-title {{ font-size:16px; font-weight:800; color: var(--text-strong); }}
    .news-cluster-meta {{ margin-top:10px; color: var(--text-soft); line-height:1.75; }}
    @media (max-width: 960px) {{ .hero {{ flex-direction:column; padding: 20px; }} .top-grid {{ grid-template-columns: 1fr; }} .analysis-grid {{ grid-template-columns: 1fr; }} .detail-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <h1>智能选股</h1>
        <p>最近生成时间：{generated_label}</p>
      </div>
      <div class="hero-actions">
        <button id="runIntelligentScreeningPageButton" class="primary-button" type="button">运行智能选股</button>
        <a class="ghost-link" href="{escape(_safe_text(dashboard_href, '/dashboard'))}">返回总览</a>
      </div>
    </section>

    <section class="panel" style="margin-bottom:18px;">
      <div class="section-title">任务状态</div>
      <div id="runIntelligentScreeningStatus" class="subtle">准备就绪</div>
      <div id="runIntelligentScreeningStep" class="subtle">等待任务开始</div>
      <div class="progress-wrap">
        <div id="runIntelligentScreeningProgressBar" class="progress-bar"><div class="progress-fill"></div></div>
      </div>
      <div class="subtle" style="display:none;">{escape(_safe_text(jobs_api_base))}</div>
    </section>

    <section class="summary-grid">
      {_render_summary_card('策略数', _format_metric_value(strategy_count), strategy_note)}
      {_render_summary_card('筛选总数', total_stocks_value, total_stocks_note)}
      {_render_summary_card('前台推荐', _format_metric_value(screening_results.get('final_recommendations', len(frontlist))), '当前前台推荐池可见条目数')}
      {_render_summary_card('5日胜率', win_rate_value, win_rate_note)}
    </section>

    <nav class="tab-bar">
      {tabs_html}
    </nav>

    <section class="stack">
      <section class="panel" style="margin-bottom:18px;">
        <div class="section-title">运行状态与口径说明</div>
        <div class="subtle">5日胜率不是今日筛选的实时胜率，而是历史推荐样本的回看统计。若未开启数据库或暂无已验证样本，则显示 --。</div>
      </section>
      {overview_sections if active_tab == 'overview' else ''}
      {report_section if active_tab == 'report' else ''}
      {recommendations_section if active_tab == 'recommendations' else ''}
    </section>
  </div>
  <script>
    const jobsApiBase = "{escape(_safe_text(jobs_api_base))}";
    const statusEl = document.getElementById("runIntelligentScreeningStatus");
    const stepEl = document.getElementById("runIntelligentScreeningStep");
    const fillEl = document.querySelector("#runIntelligentScreeningProgressBar .progress-fill");
    const buttonEl = document.getElementById("runIntelligentScreeningPageButton");

    function setJobState(job) {{
      if (!job) {{
        if (buttonEl) {{
          buttonEl.disabled = false;
        }}
        if (statusEl) {{
          statusEl.textContent = "准备就绪";
        }}
        if (stepEl) {{
          stepEl.textContent = "等待任务开始";
        }}
        if (fillEl) {{
          fillEl.style.width = "0%";
        }}
        return;
      }}

      if (buttonEl) {{
        buttonEl.disabled = job.status === "queued" || job.status === "running";
      }}
      if (statusEl) {{
        if (job.status === "succeeded") {{
          statusEl.textContent = job.message || "智能选股已完成";
        }} else if (job.status === "failed") {{
          statusEl.textContent = job.error || job.message || "智能选股执行失败";
        }} else {{
          statusEl.textContent = job.message || "智能选股已启动，正在处理中...";
        }}
      }}
      if (stepEl) {{
        const stepName = job.step_name || "等待任务开始";
        const stepProgress = job.total_steps ? `（第 ${{job.current_step || 0}}/${{job.total_steps}} 步）` : "";
        stepEl.textContent = `${{stepName}}${{stepProgress}}`;
      }}
      if (fillEl) {{
        const percent = Number.isFinite(Number(job.progress_percent)) ? Number(job.progress_percent) : 0;
        fillEl.style.width = `${{Math.max(0, Math.min(100, percent))}}%`;
      }}
    }}

    async function fetchJob(jobId) {{
      const response = await fetch(`${{jobsApiBase}}/${{encodeURIComponent(jobId)}}`);
      if (!response.ok) {{
        throw new Error(`加载任务状态失败：${{response.status}}`);
      }}
      return await response.json();
    }}

    async function fetchActiveJob() {{
      const response = await fetch(`${{jobsApiBase}}/active`);
      if (!response.ok) {{
        throw new Error(`加载活动任务失败：${{response.status}}`);
      }}
      const payload = await response.json();
      return payload.job || null;
    }}

    async function syncJobState() {{
      let pendingJobId = "";
      try {{
        const params = new URLSearchParams(window.location.search);
        pendingJobId = params.get("job_id") || window.sessionStorage.getItem("octts:intelligent-screening:pending-job-id") || "";
        if (pendingJobId) {{
          window.sessionStorage.removeItem("octts:intelligent-screening:pending-job-id");
        }}
      }} catch (_) {{
      }}

      try {{
        const job = pendingJobId ? await fetchJob(pendingJobId) : await fetchActiveJob();
        setJobState(job);
      }} catch (error) {{
        if (statusEl) {{
          statusEl.textContent = error.message || "任务状态同步失败";
        }}
      }}
    }}

    syncJobState();
  </script>
</body>
</html>"""
