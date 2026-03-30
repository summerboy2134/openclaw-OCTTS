"""Simple web UI for backtesting."""


def render_backtest_page() -> str:
    """渲染回测页面"""
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OCTTS 策略回测</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background-color: #f5f5f7;
                color: #1d1d1f;
                padding: 24px;
            }

            .container {
                max-width: 1280px;
                margin: 0 auto;
            }

            .topbar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                margin-bottom: 18px;
                flex-wrap: wrap;
            }

            .nav-links {
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
            }

            .nav-link {
                display: inline-flex;
                align-items: center;
                padding: 10px 14px;
                border-radius: 999px;
                text-decoration: none;
                background: white;
                color: #111827;
                border: 1px solid #e5e7eb;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
                font-size: 14px;
                font-weight: 500;
            }

            .nav-link:hover {
                color: #005fcc;
                border-color: rgba(0, 122, 255, 0.28);
            }

            .header,
            .panel,
            .summary-card,
            .chart-card,
            .table-card {
                background: white;
                border-radius: 16px;
                box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
            }

            .header {
                padding: 24px 28px;
                margin-bottom: 24px;
            }

            .panel {
                padding: 24px;
                margin-bottom: 24px;
            }

            .subtle {
                color: #6b7280;
                font-size: 14px;
                line-height: 1.7;
            }

            h1 {
                font-size: 32px;
                margin-bottom: 8px;
            }

            h2 {
                font-size: 22px;
                margin-bottom: 18px;
            }

            h3 {
                font-size: 18px;
                margin-bottom: 14px;
            }

            .form-row {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 16px;
                margin-bottom: 16px;
            }

            .form-group {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }

            label {
                font-weight: 600;
                color: #111827;
                font-size: 14px;
            }

            input, select, textarea {
                width: 100%;
                padding: 12px 14px;
                border: 1px solid #d1d5db;
                border-radius: 10px;
                font-size: 15px;
                background: #fff;
            }

            textarea {
                min-height: 92px;
                resize: vertical;
                line-height: 1.6;
            }

            .strategy-list {
                display: grid;
                gap: 12px;
                margin-bottom: 20px;
            }

            .strategy-item {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 12px 14px;
                border-radius: 10px;
                background: #f9fafb;
                border: 1px solid #eef2f7;
            }

            .strategy-item input[type="checkbox"] {
                width: auto;
            }

            .btn-row {
                display: flex;
                align-items: center;
                gap: 12px;
                flex-wrap: wrap;
            }

            .btn {
                padding: 12px 24px;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .btn-primary {
                background: #007aff;
                color: white;
            }

            .btn-primary:hover {
                background: #005fcc;
            }

            .btn-primary:disabled {
                background: #9ca3af;
                cursor: not-allowed;
            }

            .tips {
                background: #fff8eb;
                border-left: 4px solid #f59e0b;
                padding: 16px;
                margin: 20px 0;
                border-radius: 0 10px 10px 0;
            }

            .tips h4 {
                margin-bottom: 8px;
                color: #92400e;
            }

            .tips ul {
                margin-left: 20px;
                color: #92400e;
                line-height: 1.8;
            }

            .loading {
                display: none;
                text-align: center;
                padding: 40px 16px;
            }

            .spinner {
                border: 3px solid #e5e7eb;
                border-top: 3px solid #007aff;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }

            .results {
                display: none;
            }

            .summary-grid,
            .chart-grid {
                display: grid;
                gap: 16px;
                margin-bottom: 20px;
            }

            .summary-grid {
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            }

            .chart-grid {
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            }

            .summary-card {
                padding: 18px 20px;
            }

            .summary-label {
                color: #6b7280;
                font-size: 13px;
                margin-bottom: 8px;
            }

            .summary-value {
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 6px;
            }

            .summary-note {
                color: #6b7280;
                font-size: 13px;
            }

            .metric-good {
                color: #16a34a;
                font-weight: 600;
            }

            .metric-bad {
                color: #dc2626;
                font-weight: 600;
            }

            .chart-card,
            .table-card {
                padding: 18px 20px;
            }

            .chart-shell {
                background: #f9fafb;
                border-radius: 12px;
                padding: 14px;
                border: 1px solid #eef2f7;
            }

            .chart-meta {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 12px;
                margin-bottom: 10px;
            }

            .chart-caption {
                color: #6b7280;
                font-size: 13px;
            }

            .chart-svg {
                width: 100%;
                height: 220px;
                display: block;
            }

            .empty {
                padding: 24px;
                text-align: center;
                color: #6b7280;
                background: #f9fafb;
                border-radius: 10px;
            }

            .table-shell {
                overflow-x: auto;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                min-width: 860px;
            }

            th {
                background: #f9fafb;
                padding: 12px;
                text-align: left;
                font-weight: 600;
                border-bottom: 2px solid #e5e7eb;
                font-size: 14px;
            }

            td {
                padding: 12px;
                border-bottom: 1px solid #e5e7eb;
                font-size: 14px;
                vertical-align: top;
            }

            .mono {
                font-family: ui-monospace, SFMono-Regular, SFMono-Regular, Menlo, monospace;
            }

            .badge {
                display: inline-flex;
                align-items: center;
                padding: 4px 10px;
                border-radius: 999px;
                background: #eef4ff;
                color: #1d4ed8;
                font-size: 12px;
                font-weight: 600;
            }

            .section-stack > * + * {
                margin-top: 20px;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            @media (max-width: 768px) {
                body {
                    padding: 16px;
                }

                .header,
                .panel,
                .summary-card,
                .chart-card,
                .table-card {
                    border-radius: 12px;
                }

                h1 {
                    font-size: 28px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="topbar">
                <div class="nav-links">
                    <a class="nav-link" href="/dashboard">返回总览仪表板</a>
                    <a class="nav-link" href="/intelligent-screening">查看智能选股</a>
                </div>
                <span class="subtle">这里回测的是“策略在历史上选出来的股票”，不是默认只回测某一只股票。</span>
            </div>

            <div class="header">
                <h1>📊 OCTTS 轻量级策略回测</h1>
                <p class="subtle">
                    快速验证选股策略的历史表现。你可以回测全市场策略命中结果，也可以限定在你输入的股票池内查看策略效果。
                </p>
            </div>

            <div class="panel">
                <h2>回测设置</h2>

                <div class="form-row">
                    <div class="form-group">
                        <label>开始日期</label>
                        <input type="date" id="start_date" value="2024-01-01">
                    </div>
                    <div class="form-group">
                        <label>结束日期</label>
                        <input type="date" id="end_date" value="2024-12-31">
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label>持有天数</label>
                        <select id="holding_days">
                            <option value="3">3天</option>
                            <option value="5" selected>5天（推荐）</option>
                            <option value="10">10天</option>
                            <option value="20">20天</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>每次选股数量</label>
                        <select id="top_n">
                            <option value="5">5只</option>
                            <option value="10" selected>10只（推荐）</option>
                            <option value="20">20只</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>手续费率</label>
                        <input type="number" id="commission_rate" min="0" step="0.0001" value="0.0003">
                    </div>
                    <div class="form-group">
                        <label>滑点率</label>
                        <input type="number" id="slippage_rate" min="0" step="0.0001" value="0.0005">
                    </div>
                </div>

                <div class="form-group">
                    <label>选择策略</label>
                    <div class="strategy-list">
                        <div class="strategy-item">
                            <input type="checkbox" id="oversold_bounce" checked>
                            <label for="oversold_bounce">超跌反弹 - 寻找RSI&lt;30的超跌股票</label>
                        </div>
                        <div class="strategy-item">
                            <input type="checkbox" id="volume_breakout" checked>
                            <label for="volume_breakout">放量突破 - 成交量放大且价格上涨</label>
                        </div>
                        <div class="strategy-item">
                            <input type="checkbox" id="golden_cross">
                            <label for="golden_cross">均线金叉 - 5日均线上穿20日均线</label>
                        </div>
                        <div class="strategy-item">
                            <input type="checkbox" id="small_cap_growth">
                            <label for="small_cap_growth">小盘成长 - 市值较小且上涨的股票</label>
                        </div>
                    </div>
                </div>

                <div class="form-group">
                    <label>可选股票池</label>
                    <textarea id="stock_pool" placeholder="留空表示回测全市场中被策略选中的股票；如需限定范围，可输入 600000.SH,000001.SZ"></textarea>
                    <div class="subtle">只有落在这个股票池里的策略命中标的才会纳入回测，所以这里能回答“到底回测哪只股票/哪几只股票”。</div>
                </div>

                <div class="tips">
                    <h4>💡 使用提示</h4>
                    <ul>
                        <li>本页默认回测“所选策略”在历史上的选股结果，不是固定单只股票。</li>
                        <li>如果你只想看某几只股票，请在“可选股票池”里填入代码，例如 `600000.SH,000001.SZ`。</li>
                        <li>默认手续费率 0.0003、滑点率 0.0005，可按需要调整。</li>
                        <li>建议选择至少 3 个月以上的时间段，以减少偶然性。</li>
                        <li>结果仅供研究参考，实盘请结合风控与仓位管理。</li>
                    </ul>
                </div>

                <div class="btn-row">
                    <button class="btn btn-primary" onclick="runBacktest()">开始策略回测</button>
                    <span class="subtle">结果将展示回测范围、最优策略、收益/回撤曲线和最近交易明细。</span>
                </div>
            </div>

            <div class="results" id="results">
                <div class="panel">
                    <h2>回测结果</h2>
                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <p class="subtle">正在回测中，请稍候...</p>
                    </div>
                    <div id="results-content" class="section-stack"></div>
                </div>
            </div>
        </div>

        <script>
            function escapeHtml(value) {
                return String(value ?? '')
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#39;');
            }

            function formatPercent(value, digits = 2) {
                const number = Number(value);
                if (!Number.isFinite(number)) return '--';
                return `${number.toFixed(digits)}%`;
            }

            function getSignedClass(value, reverse = false) {
                const number = Number(value);
                if (!Number.isFinite(number) || number === 0) return '';
                const positive = reverse ? number < 0 : number > 0;
                return positive ? 'metric-good' : 'metric-bad';
            }

            function renderMetricCard(label, value, note = '', className = '') {
                return `
                    <div class="summary-card">
                        <div class="summary-label">${escapeHtml(label)}</div>
                        <div class="summary-value ${className}">${escapeHtml(value)}</div>
                        <div class="summary-note">${escapeHtml(note)}</div>
                    </div>
                `;
            }

            function renderLineChart(series, options = {}) {
                if (!series || !series.length) {
                    return '<div class="empty">暂无图表数据。</div>';
                }

                const width = 720;
                const height = 220;
                const paddingX = 22;
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
                const polyline = points.map(point => `${point.x},${point.y}`).join(' ');
                const areaPoints = `${paddingX},${height - paddingY} ${polyline} ${width - paddingX},${height - paddingY}`;
                const last = series[series.length - 1];
                const lastPoint = points[points.length - 1];
                const first = series[0];
                const stroke = options.stroke || '#2563eb';
                const fill = options.fill || 'rgba(37, 99, 235, 0.12)';
                const baselineValue = Number(options.baselineValue ?? min);
                const baselineY = paddingY + innerHeight - ((baselineValue - min) / range) * innerHeight;
                const valueFormatter = options.valueFormatter || (value => Number(value).toFixed(2));

                return `
                    <div class="chart-shell">
                        <div class="chart-meta">
                            <div>
                                <div class="summary-label">${escapeHtml(options.title || '曲线')}</div>
                                <div class="chart-caption">${escapeHtml(options.caption || '')}</div>
                            </div>
                            <div class="summary-value ${getSignedClass(options.signed ? last.value : 0, options.reverseSigned)}">${escapeHtml(valueFormatter(last.value))}</div>
                        </div>
                        <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
                            <line x1="${paddingX}" y1="${paddingY}" x2="${paddingX}" y2="${height - paddingY}" stroke="rgba(148, 163, 184, 0.3)" />
                            <line x1="${paddingX}" y1="${height - paddingY}" x2="${width - paddingX}" y2="${height - paddingY}" stroke="rgba(148, 163, 184, 0.3)" />
                            <line x1="${paddingX}" y1="${baselineY}" x2="${width - paddingX}" y2="${baselineY}" stroke="rgba(148, 163, 184, 0.24)" stroke-dasharray="4 4" />
                            <polyline fill="${fill}" stroke="none" points="${areaPoints}" />
                            <polyline fill="none" stroke="${stroke}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${polyline}" />
                            <circle cx="${lastPoint.x}" cy="${lastPoint.y}" r="4" fill="${stroke}" />
                            <text x="${paddingX}" y="14" fill="#6b7280" font-size="11">${escapeHtml(valueFormatter(max))}</text>
                            <text x="${paddingX}" y="${height - 4}" fill="#6b7280" font-size="11">${escapeHtml(valueFormatter(min))}</text>
                            <text x="${paddingX}" y="${height - 4}" dx="52" fill="#6b7280" font-size="11">${escapeHtml(String(first.trade_date || ''))}</text>
                            <text x="${width - paddingX}" y="${height - 4}" text-anchor="end" fill="#6b7280" font-size="11">${escapeHtml(String(last.trade_date || ''))}</text>
                        </svg>
                    </div>
                `;
            }

            function renderStrategyTable(results) {
                const entries = Object.entries(results || {});
                if (!entries.length) {
                    return '<div class="empty">暂无策略结果。</div>';
                }

                return `
                    <div class="table-card">
                        <h3>策略对比</h3>
                        <div class="table-shell">
                            <table>
                                <thead>
                                    <tr>
                                        <th>策略名称</th>
                                        <th>总交易</th>
                                        <th>盈利 / 亏损</th>
                                        <th>胜率</th>
                                        <th>平均收益</th>
                                        <th>总收益</th>
                                        <th>最大回撤</th>
                                        <th>Sharpe</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${entries.map(([strategy, result]) => `
                                        <tr>
                                            <td><strong>${escapeHtml(strategy)}</strong></td>
                                            <td>${escapeHtml(String(result.total_trades ?? 0))}</td>
                                            <td>${escapeHtml(String(result.winning_trades ?? 0))} / ${escapeHtml(String(result.losing_trades ?? 0))}</td>
                                            <td class="${getSignedClass((Number(result.win_rate) - 0.5) * 100)}">${escapeHtml(formatPercent(Number(result.win_rate || 0) * 100))}</td>
                                            <td class="${getSignedClass(result.avg_return)}">${escapeHtml(formatPercent(result.avg_return))}</td>
                                            <td class="${getSignedClass(result.total_return)}">${escapeHtml(formatPercent(result.total_return))}</td>
                                            <td class="${getSignedClass(result.max_drawdown, true)}">${escapeHtml(formatPercent(result.max_drawdown))}</td>
                                            <td>${escapeHtml(Number(result.sharpe_ratio || 0).toFixed(2))}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                `;
            }

            function renderTradeDetails(strategyName, records) {
                if (!records || !records.length) {
                    return `
                        <div class="table-card">
                            <h3>最近交易明细</h3>
                            <div class="empty">${escapeHtml(strategyName)} 在当前区间没有产生交易。</div>
                        </div>
                    `;
                }

                const recent = records.slice(-8).reverse();
                return `
                    <div class="table-card">
                        <h3>最近交易明细 · ${escapeHtml(strategyName)}</h3>
                        <div class="table-shell">
                            <table>
                                <thead>
                                    <tr>
                                        <th>股票</th>
                                        <th>信号日</th>
                                        <th>入场</th>
                                        <th>出场</th>
                                        <th>成交字段</th>
                                        <th>毛收益</th>
                                        <th>净收益</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${recent.map(item => `
                                        <tr>
                                            <td class="mono">${escapeHtml(item.ts_code)}</td>
                                            <td class="mono">${escapeHtml(item.signal_date)}</td>
                                            <td>
                                                <div class="mono">${escapeHtml(item.entry_date)}</div>
                                                <div>${escapeHtml(String(item.entry_price))}</div>
                                            </td>
                                            <td>
                                                <div class="mono">${escapeHtml(item.exit_date)}</div>
                                                <div>${escapeHtml(String(item.exit_price))}</div>
                                            </td>
                                            <td>
                                                <span class="badge">open: ${escapeHtml(item.entry_price_field || '--')}</span>
                                                <br><br>
                                                <span class="badge">close: ${escapeHtml(item.exit_price_field || '--')}</span>
                                            </td>
                                            <td class="${getSignedClass(item.gross_return_pct)}">${escapeHtml(formatPercent(item.gross_return_pct))}</td>
                                            <td>
                                                <div class="${getSignedClass(item.return_pct)}">${escapeHtml(formatPercent(item.return_pct))}</div>
                                                <div class="subtle">费率 ${escapeHtml(formatPercent(Number(item.commission_rate || 0) * 100, 4))} / 滑点 ${escapeHtml(formatPercent(Number(item.slippage_rate || 0) * 100, 4))}</div>
                                            </td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                `;
            }

            function displayResults(data) {
                const entries = Object.entries(data.results || {});
                if (!entries.length) {
                    document.getElementById('results-content').innerHTML = '<div class="empty">没有返回任何回测结果。</div>';
                    return;
                }

                const [bestStrategyName, bestResult] = entries.reduce((best, current) => {
                    return Number(current[1].total_return || 0) > Number(best[1].total_return || 0) ? current : best;
                });

                const equitySeries = (bestResult.equity_curve || []).map(item => ({
                    trade_date: item.trade_date,
                    value: Number(item.value || 0)
                }));
                const drawdownSeries = (bestResult.equity_curve || []).map(item => ({
                    trade_date: item.trade_date,
                    value: Number(item.drawdown || 0)
                }));

                let html = '<div class="summary-grid">';
                html += renderMetricCard('回测范围', escapeHtml(data.summary?.stock_scope || '全市场策略命中股票'), '可通过股票池输入框限定范围');
                html += renderMetricCard('最优策略', escapeHtml(data.summary?.best_strategy || bestStrategyName), data.summary?.period || '');
                html += renderMetricCard('总收益', formatPercent(data.summary?.best_total_return ?? bestResult.total_return), '最佳策略累计表现', getSignedClass(data.summary?.best_total_return ?? bestResult.total_return));
                html += renderMetricCard('最大回撤', formatPercent(data.summary?.best_max_drawdown ?? bestResult.max_drawdown), '越低越稳健', getSignedClass(data.summary?.best_max_drawdown ?? bestResult.max_drawdown, true));
                html += renderMetricCard('Sharpe', Number(data.summary?.best_sharpe_ratio ?? bestResult.sharpe_ratio).toFixed(2), '风险调整后收益');
                html += '</div>';

                html += `
                    <div class="summary-card" style="margin-bottom:20px; padding:20px;">
                        <div class="summary-label">回测摘要</div>
                        <div class="subtle">${escapeHtml(data.summary?.recommendation || '')}</div>
                        <div class="subtle" style="margin-top:8px;">手续费率 ${escapeHtml(formatPercent(Number(data.summary?.commission_rate || 0) * 100, 4))}，滑点率 ${escapeHtml(formatPercent(Number(data.summary?.slippage_rate || 0) * 100, 4))}</div>
                    </div>
                `;

                html += renderStrategyTable(data.results || {});

                html += '<div class="chart-grid">';
                html += `
                    <div class="chart-card">
                        <h3>权益曲线 · ${escapeHtml(bestStrategyName)}</h3>
                        ${renderLineChart(equitySeries, {
                            title: '累计权益',
                            caption: '基于每笔净收益递推',
                            baselineValue: 1,
                            signed: true,
                            valueFormatter: value => `${Number(value).toFixed(2)}x`
                        })}
                    </div>
                `;
                html += `
                    <div class="chart-card">
                        <h3>回撤曲线 · ${escapeHtml(bestStrategyName)}</h3>
                        ${renderLineChart(drawdownSeries, {
                            title: '回撤走势',
                            caption: '按交易序列统计的阶段性回撤',
                            baselineValue: 0,
                            stroke: '#dc2626',
                            fill: 'rgba(220, 38, 38, 0.10)',
                            reverseSigned: true,
                            valueFormatter: value => `${Number(value).toFixed(2)}%`
                        })}
                    </div>
                `;
                html += '</div>';

                html += renderTradeDetails(bestStrategyName, bestResult.detail_records || []);

                document.getElementById('results-content').innerHTML = html;
            }

            async function runBacktest() {
                const startDate = document.getElementById('start_date').value.replace(/-/g, '');
                const endDate = document.getElementById('end_date').value.replace(/-/g, '');
                const holdingDays = Number(document.getElementById('holding_days').value);
                const topN = Number(document.getElementById('top_n').value);
                const commissionRate = Number(document.getElementById('commission_rate').value);
                const slippageRate = Number(document.getElementById('slippage_rate').value);
                const stockPoolRaw = document.getElementById('stock_pool').value;
                const stockPool = stockPoolRaw
                    .split(/[,\n]/)
                    .map(item => item.trim().toUpperCase())
                    .filter(Boolean);
                const button = document.querySelector('.btn-primary');

                if (!startDate || !endDate || startDate > endDate) {
                    alert('请输入有效的回测日期范围');
                    return;
                }
                if (!Number.isFinite(commissionRate) || commissionRate < 0 || !Number.isFinite(slippageRate) || slippageRate < 0) {
                    alert('请输入有效的手续费率和滑点率');
                    return;
                }

                const strategies = [];
                document.querySelectorAll('.strategy-item input:checked').forEach(input => {
                    strategies.push(input.id);
                });

                if (!strategies.length) {
                    alert('请至少选择一个策略');
                    return;
                }

                document.getElementById('results').style.display = 'block';
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results-content').innerHTML = '';
                button.disabled = true;

                try {
                    const response = await fetch('/api/backtest', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            start_date: startDate,
                            end_date: endDate,
                            holding_days: holdingDays,
                            top_n: topN,
                            commission_rate: commissionRate,
                            slippage_rate: slippageRate,
                            strategies: strategies,
                            stock_pool: stockPool
                        })
                    });

                    const data = await response.json();
                    if (!response.ok) {
                        throw new Error(data.detail || '回测请求失败');
                    }

                    displayResults(data);
                } catch (error) {
                    document.getElementById('results-content').innerHTML =
                        `<div class="tips"><h4>回测失败</h4><p>${escapeHtml(error.message)}</p></div>`;
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    button.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """
