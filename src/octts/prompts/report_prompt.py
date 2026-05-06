from __future__ import annotations

import json
from datetime import datetime

from typing import Optional

from octts.schemas.report import AnalysisPhase, HistoricalAnalysisRecord, MemorySummary, PositionStatus, PriceSnapshot


def _today_screening_hard_constraints() -> list[str]:
    return [
        "不能改写系统排序，也不能擅自调整推荐先后。",
        "只生成 focus_stocks、comparison、overall_action 三块，不要输出 yesterday_reviews。",
        "严禁把内部工程、训练或回补说明写入用户报告，例如“回补样本”“结构化特征近似”“训练集构造”“fallback”“backfill”等词一律不得输出。",
        "风险分为0或0.0时，不要写成核心亮点，也不要输出“风险分为0.0”；应解释为“结构化末端风险暂未触发明显异常，但仍需看资金承接和换手延续”。",
        "如果 recommendation_score 低于 overall_score，不得写“短线交易排序分相对综合分更占优”；应解释为“综合质量尚可，但风险/执行分被短期涨幅、位置或承接约束压低”。只有 recommendation_score 明显高于 overall_score 时，才可说交易性强于质量端。",
        "若输入存在 top_list/top_list_summary/limit_list/limit_status，要把龙虎榜资金方向、上榜原因、涨跌停强弱作为交易结构证据，但不得据此改写排序。",
        "若输入存在 earnings_forecast，要说明业绩预告对基本面判断的支持或拖累；没有则不要编造业绩预告。",
        "focus_stocks 必须逐只覆盖 today_top3；comparison 只基于 today_top3。",
        "若 today_top3_live_context 中存在对应个股的实时资讯，focus_analysis 必须优先引用其中的新闻、公告与发布时间线索；若为空，再回退到 news_clusters 与个股自身字段。",
        "若无法联网补充外部信息，就严格基于输入里的价格、资金、财务、主营摘要、主题线索做推演，不要编造缺失数据。",
        "comparison 必须包含 cross_stock_synthesis_view：横向比较三只股票谁更偏交易性、谁更偏基本面质量、谁风险更高、谁证据最完整。",
    ]


def _today_screening_writing_style_guidelines() -> list[str]:
    return [
        "不要逐字段罗列输入；必须先理解 evidence_digest，再归纳成结论、证据、反证和操作前提。报告应该像研究员写给交易员的判断，不像字段解释说明书。",
        "生成内容必须充分展开，today_top3 每只 focus_analysis 建议不少于 220 个中文字符，market_performance_view、catalyst_and_capital_view、trading_context_view 各输出 2-3 句；但不要空泛重复，也不要为了凑长度复述字段名。",
        "排序来源优先解释为：全市场模型排名、极端风险剔除后的Top3席位、资金/风险后的执行分、综合质量分。不要把 recommendation_score 直接称为最终推荐分；它更接近风险调整后的执行分/排序分。",
        "不要连续列举超过3个数字。数字只作为证据，不作为段落主语；先给判断，再用关键数字支撑。",
        "每只股票必须用至少一个非指标化判断句开头，例如“这是高位资金博弈票”“这是资金推动的修复票”“这是质量一般但情绪强的短线票”，然后再展开证据。",
        "focus_analysis 必须是完整段落，不要堆标签；应覆盖主结论、核心证据、反证风险、缺失信息和操作前提。",
        "focus_analysis 要优先按这个顺序组织：先给一句人话结论和交易画像，再写市场表现阶段（今日涨跌、日内强弱、当前处于启动/加速/高位分歧/调整/修复哪个阶段），再写主营逻辑或基本面质地，再写技术位置与量价结构、资金承接、板块与主题共振，最后写主要风险与操作前提。段落要真正围绕该股输入数据做判断，避免三只股票复用同样开头或同样结论句。",
        "focus_analysis 要尽量回答四个问题：这只股票今天是强还是弱、为什么走到当前位置、当前主要催化是什么、接下来最该防什么风险。",
        "短线建议必须基于输入中的 close、entry_price、技术描述、支撑阻力或总结信息；证据不足时只给观察区间。",
        "主题新闻输出优先概括市场总线、核心主线、风险扰动、观察线；overall_action 中的 market_view、risk_summary、action_items 要简明可执行。",
    ]


def build_today_screening_report_prompt(
    *,
    market_data: dict[str, object],
    news_clusters: list[dict[str, object]],
    screening_context: dict[str, object],
) -> tuple[str, str]:
    system_prompt = (
        "你是一名严格遵守排序约束的 A 股智能选股复盘分析助手。"
        "你的任务是解释系统已经给出的排序与分数，不允许改写排序，不允许擅自调整推荐先后。"
        "你只能输出一个合法 JSON 对象，不要输出 Markdown、代码块或额外解释。"
        "优先保证 JSON 完整闭合、字段齐全、内容收敛可读；证据不足时使用保守描述、null、空数组或观察区间，不要编造精准价位。"
    )

    payload = {
        "task": "基于今日 Top3 与主题上下文生成结构化报告",
        "market_data": market_data,
        "news_clusters": news_clusters,
        "screening_context": {
            "today_top3": screening_context.get("today_top3") or [],
            "today_top3_live_context": screening_context.get("today_top3_live_context") or [],
            "comparison_candidates": screening_context.get("comparison_candidates") or [],
            "cross_stock_synthesis": screening_context.get("cross_stock_synthesis") or {},
        },
        "field_semantics": {
            "model_rank": "预测模型在全市场Top100中的排序，Top3只在此排序基础上做极端风险剔除",
            "model_score": "预测模型原始分",
            "model_blend_score": "模型与轻量规则融合后的参考分，仅用于解释，不改写Top3",
            "overall_score": "多维综合分",
            "recommendation_score": "最终推荐排序分，解释排序时优先参考该字段",
            "risk_score": "统一风险分，优先来自分歧/派发风险分",
            "priority_score": "与综合分兼容的展示优先级字段",
            "overall_confidence": "分析置信度",
            "display_confidence": "前端展示置信度",
            "strategy_count": "命中策略数量",
            "news_mentioned": "是否命中新闻催化",
            "score_change": "与上一交易日推荐分变化",
            "score_snapshot": "统一后的分数字段快照，用于避免混用旧字段",
            "top_list": "龙虎榜明细，本轮只对今日Top3注入",
            "limit_list": "涨跌停/连板异动明细，本轮只对今日Top3注入",
            "earnings_forecast": "业绩预告摘要，来自Tushare forecast",
            "evidence_digest": "系统预先整理的证据摘要，包含正向证据、风险/反证、缺失数据和操作前提",
            "cross_stock_synthesis": "Top3横向综合任务，要求比较交易性、质量、风险和证据完整度",
            "source_tag": "来源标签，如今日Top3/今日候选/昨日复盘",
        },
        "hard_constraints": _today_screening_hard_constraints(),
        "writing_style_guidelines": _today_screening_writing_style_guidelines(),
        "instructions": [
            "先满足 hard_constraints，再尽量满足 writing_style_guidelines。",
            "若输入存在 distribution_risk_score、distribution_risk_flags、moneyflow_3d_value、turnover_spike_ratio、recent_runup_5d、late_stage_momentum_flag、industry_flow_bias、industry_heat_score，必须写出加分、扣分与风险含义；其中 distribution_risk_score 只是分歧/派发风险子分，不代表全部风险。",
            "若输入中存在 close/open/high/low/pct_change/turnover_rate/amount/amplitude/recent_runup_5d 等市场表现字段，要明确写出冲高回落、高位震荡、放量分歧、强势延续、回踩整理等阶段判断，不要只写笼统的技术面偏强/偏弱。",
            "若输入中存在 business_summary、latest_revenue_yoy、latest_profit_yoy、pe_ttm、industry_pe_median 等基本面或估值字段，要明确判断基本面是否支持当前涨幅、是否存在估值透支或基本面与股价背离。",
            "若输入中存在 catalyst_summary、main_fund_flow_1d、main_fund_flow_3d、main_fund_flow_10d、margin_balance_change_10d 等催化或资金字段，要区分主催化与次催化，并写出资金承接还是资金分歧。",
            "若大盘、板块广度、主题新闻簇等外部环境数据不足，不要写成‘缺乏实时市场数据支持’或暗示个股行情缺失；应明确表达为‘外部环境证据有限，当前判断主要依赖个股价格、量能、资金和风险结构’，并继续基于已有个股证据分析。",
            "focus_stocks 的 market_performance_view 与 catalyst_and_capital_view 应尽量输出 2-3 句短摘要，作为总览和重点分析之间的桥接层；证据不足时可保守但不要空泛重复。focus_analysis 不要机械重复这两个字段原句，而要在其基础上进一步归纳、串联和判断。",
            "focus_stocks 的 core_highlights 控制为 3-4 条，risk_warnings 控制为 2-3 条，overall_assessment 用 2-3 句给出完整结论。",
            "theme_focuses 若能稳定判断再输出；证据不足可返回空数组，不要编造。",
        ],
        "output_schema": {
            "focus_stocks": [
                {
                    "ts_code": "string",
                    "name": "string",
                    "score_rationale": "解释 recommendation_score / overall_score 的主要加分、减分来源",
                    "fundamental_view": "基本面、质地、业绩或行业逻辑判断",
                    "market_context_view": "大盘、板块、主题环境与个股匹配度判断；如果外部环境证据不足，应说明外部环境证据有限，不能写‘缺乏实时市场数据支持’",
                    "trading_context_view": "技术结构、位置、量能、节奏、承接情况",
                    "market_performance_view": "概括今日涨跌、日内走势、当前阶段判断",
                    "catalyst_and_capital_view": "概括主催化、次催化、资金承接或资金分歧",
                    "focus_analysis": "完整段落，优先包含主结论、市场表现、主营/基本面逻辑、催化与资金、反证风险、缺失信息、操作前提",
                    "evidence_based_view": "基于 evidence_digest 的证据归纳，不要逐字段复述",
                    "counter_evidence": "主要反证、风险或缺失数据",
                    "core_highlights": ["string"],
                    "risk_warnings": ["string"],
                    "overall_assessment": "string",
                    "action_plan": {
                        "action_bias": "买入|观察|减仓|不参与",
                        "entry_zone": "string|null",
                        "take_profit": "string|null",
                        "stop_loss": "string|null",
                        "holding_horizon": "string|null",
                        "invalid_condition": "string|null"
                    }
                }
            ],
            "comparison": {
                "basic_rank": ["ts_code"],
                "technical_rank": ["ts_code"],
                "risk_rank": ["ts_code"],
                "trading_rank": ["ts_code"],
                "best_short_term": "ts_code",
                "most_robust": "ts_code",
                "highest_risk": "ts_code",
                "cross_stock_synthesis_view": "横向综合比较段落"
            },
            "overall_action": {
                "headline": "string",
                "market_view": "string",
                "risk_summary": "string",
                "action_items": ["string"],
                "theme_focuses": [
                    {
                        "theme": "string",
                        "tier": "主线|次主线|观察|风险",
                        "summary": "string",
                        "continuity_view": "string",
                        "related_stocks": ["string"]
                    }
                ]
            }
        }
    }

    return system_prompt, json.dumps(payload, ensure_ascii=False, indent=2)


def build_today_screening_analysis_prompt(
    *,
    market_data: dict[str, object],
    news_clusters: list[dict[str, object]],
    screening_context: dict[str, object],
) -> tuple[str, str]:
    system_prompt = (
        "你是一名A股策略研究员，负责先做深度推理，不负责最终排版。"
        "你必须严格基于输入证据分析，不允许改写系统Top3排序，不允许编造缺失行情。"
        "输出一个合法JSON对象，不要Markdown，不要代码块。"
    )
    payload = {
        "task": "对今日Top3进行研究员级推理分析，生成中间分析草稿",
        "market_data": market_data,
        "news_clusters": news_clusters,
        "screening_context": {
            "today_top3": screening_context.get("today_top3") or [],
            "cross_stock_synthesis": screening_context.get("cross_stock_synthesis") or {},
        },
        "instructions": [
            "重点做归纳、推理、反证和取舍，不要逐字段复述。",
            "每只股票必须输出主论点、支持证据、反证/风险、缺失数据、操作逻辑，分析要足够展开，不能只给一句摘要。",
            "严禁输出内部工程、训练或回补说明，例如“回补样本”“结构化特征近似”“训练集构造”“fallback”“backfill”等。",
            "风险分为0或0.0时，只能理解为结构化末端风险暂未触发明显异常，不能当作核心亮点，也不能写成“风险分为0.0”。",
            "先给交易画像和判断，再用指标证明；不要以指标堆叠替代判断。",
            "如果 recommendation_score 低于 overall_score，不得写“交易排序分更占优”；应说明执行分被风险/位置/承接因素压低。",
            "必须横向比较Top3：谁更偏交易性、谁质量更稳、谁风险最高、谁证据最完整。",
            "若市场概览不可用，明确写不可用，不要假设指数涨跌或涨跌家数。",
            "可以充分展开分析，但所有结论都要能回溯到输入证据。",
        ],
        "output_schema": {
            "focus_reasoning": [
                {
                    "ts_code": "string",
                    "main_thesis": "核心判断",
                    "supporting_evidence": ["string"],
                    "counter_evidence": ["string"],
                    "missing_data": ["string"],
                    "risk_assessment": "风险评估",
                    "action_logic": "操作逻辑与触发/失效条件",
                    "confidence_view": "证据完整度与判断置信度",
                }
            ],
            "cross_stock_comparison": {
                "trading_preference": "谁更适合短线交易及原因",
                "quality_preference": "谁质量更稳及原因",
                "risk_order": "风险由高到低及原因",
                "evidence_quality": "三只股票证据完整度比较",
                "portfolio_view": "若只跟踪Top3，整体节奏和仓位思路",
            },
            "market_and_theme_reasoning": {
                "market_view": "市场概览可用则分析，不可用则标记不可用",
                "theme_view": "主题与新闻线索分析",
                "risk_view": "整体风险分析",
            },
        },
    }
    return system_prompt, json.dumps(payload, ensure_ascii=False, indent=2)


def build_today_screening_format_prompt(
    *,
    market_data: dict[str, object],
    news_clusters: list[dict[str, object]],
    screening_context: dict[str, object],
    reasoning_payload: dict[str, object],
) -> tuple[str, str]:
    system_prompt = (
        "你是一名A股智能选股报告编辑，负责把研究员分析草稿整理成稳定JSON报告。"
        "必须保留系统Top3排序，不允许新增推荐或改写排序。"
        "你只能输出一个合法JSON对象，不要Markdown、代码块或额外解释。"
    )
    base_system, base_payload_text = build_today_screening_report_prompt(
        market_data=market_data,
        news_clusters=news_clusters,
        screening_context=screening_context,
    )
    del base_system
    base_payload = json.loads(base_payload_text)
    base_payload["task"] = "基于研究员推理草稿生成最终结构化晨报JSON"
    base_payload["reasoning_payload"] = reasoning_payload
    base_payload["instructions"] = [
        "最终输出必须符合 output_schema。",
        "必须优先吸收 reasoning_payload 的主论点、反证、缺失数据和操作逻辑。",
        "先满足 hard_constraints，再尽量满足 writing_style_guidelines。",
        "如果某只股票的主分析仍然像字段说明书，要优先改写成研究员式点评，而不是继续补分数字段。",
        "comparison.cross_stock_synthesis_view 必须总结Top3横向取舍。",
    ]
    return system_prompt, json.dumps(base_payload, ensure_ascii=False, indent=2)


def build_focus_analysis_rewrite_prompt(
    *,
    stock_context: dict[str, object],
    reasoning_item: Optional[dict[str, object]],
    existing_focus_analysis: str,
) -> tuple[str, str]:
    system_prompt = (
        "你是一名A股智能选股报告编辑，只负责重写单只股票的主分析。"
        "你必须保留原始事实与排序背景，不能编造数据，不能改写推荐顺序。"
        "你只能输出一个合法JSON对象，不要Markdown、代码块或额外解释。"
    )
    payload = {
        "task": "重写单只重点个股的主分析，使其更像研究员点评而不是字段堆砌",
        "stock_context": stock_context,
        "reasoning_item": reasoning_item or {},
        "existing_focus_analysis": existing_focus_analysis,
        "hard_constraints": [
            "只能重写 focus_analysis，不要输出其他字段。",
            "不得改写排序、结论方向、关键价位和已有事实。",
            "不得编造市场指数、涨跌家数、外部新闻或缺失数据。",
            "不得出现内部工程、训练、fallback/backfill 等词。",
        ],
        "writing_style_guidelines": [
            "开头先给交易画像或一句人话判断，不要以上来就堆数字。",
            "优先写机会来源、当前阶段、核心反证和操作前提，不要逐字段解释 recommendation_score / overall_score。",
            "避免使用这些套话：它能进入今日重点名单、真正需要提防的不是普通波动、综合来看当前这只票、操作上先看。",
            "focus_analysis 保持单段落，控制在 220-360 个中文字符，像研究员写给交易员的盘前/复盘点评。",
        ],
        "output_schema": {
            "focus_analysis": "string",
        },
    }
    return system_prompt, json.dumps(payload, ensure_ascii=False, indent=2)


def build_yesterday_review_report_prompt(
    *,
    news_clusters: list[dict[str, object]],
    screening_context: dict[str, object],
) -> tuple[str, str]:
    system_prompt = (
        "你是一名严格遵守排序约束的 A 股智能选股复盘分析助手。"
        "你的任务是解释系统已经给出的复盘对象与结论，不允许擅自新增推荐。"
        "你只能输出一个合法 JSON 对象，不要输出 Markdown、代码块或额外解释。"
        "优先保证 JSON 完整闭合、字段齐全、内容收敛可读；证据不足时使用保守描述、null、空数组或观察区间，不要编造精准价位。"
    )

    payload = {
        "task": "基于昨日 Top3 复盘上下文生成结构化复盘",
        "news_clusters": news_clusters,
        "screening_context": {
            "yesterday_top3_review": screening_context.get("yesterday_top3_review") or [],
            "yesterday_top3_live_context": screening_context.get("yesterday_top3_live_context") or [],
        },
        "instructions": [
            "只生成 yesterday_reviews 一块，不要输出 focus_stocks、comparison、overall_action。",
            "yesterday_reviews 必须逐只覆盖 yesterday_top3_review。",
            "若 yesterday_top3_review 为空，必须返回空数组，不要自行编造昨日复盘内容。",
            "review_analysis 必须是完整段落，不要堆标签；可以充分展开，但不得空泛重复。",
            "昨日复盘重点写清：昨日逻辑今天是否兑现、强弱变化来自哪些因素、当前更适合继续跟踪/减仓/放弃/观察、下一步风险与操作。",
            "若个股未重新进入 today_top3 或今日候选，必须明确它只是复盘跟踪对象，不是今日继续推荐。",
            "review_analysis 必须明确回答四件事：昨日逻辑有没有兑现、今天变化来自哪里、当前状态该怎么处理、接下来重点观察什么延续/失效信号；避免只做泛化总结。",
            "若有新闻簇背景，可作为市场环境辅助；若证据不足，保守表达，不要编造。",
        ],
        "output_schema": {
            "yesterday_reviews": [
                {
                    "ts_code": "string",
                    "name": "string",
                    "yesterday_conclusion": "string",
                    "today_verdict": "一句明确结论，需说明当前处于继续强势/降级跟踪/转弱/失效中的哪种状态",
                    "status": "延续|转弱|失效|观察",
                    "strength_change": "相对昨日的强弱变化与来源",
                    "market_context_view": "今日市场环境是否仍支持原逻辑",
                    "review_analysis": "完整复盘段落",
                    "analysis": "不少于两句，需交代昨日逻辑今天是否兑现、当前风险和操作取向",
                    "miss_reason_candidates": ["string"],
                    "missing_factor_candidates": ["string"]
                }
            ]
        }
    }

    return system_prompt, json.dumps(payload, ensure_ascii=False, indent=2)


def build_intelligent_screening_report_prompt(
    *,
    market_data: dict[str, object],
    news_clusters: list[dict[str, object]],
    screening_context: dict[str, object],
) -> tuple[str, str]:
    return build_today_screening_report_prompt(
        market_data=market_data,
        news_clusters=news_clusters,
        screening_context=screening_context,
    )



def build_report_prompt(
    *,
    phase: AnalysisPhase,
    snapshot: PriceSnapshot,
    previous_memory: Optional[MemorySummary],
    previous_record: Optional[HistoricalAnalysisRecord] = None,
    market_context: Optional[dict[str, object]] = None,
    previous_trading_snapshot: Optional[dict[str, object]] = None,
    screening_context: Optional[dict[str, object]] = None,
    is_default_pool_symbol: bool = False,
    position_status: Optional[PositionStatus] = None,
) -> tuple[str, str]:
    system_prompt = (
        "你是一名严谨的 A 股量化复盘分析助手。"
        "你必须基于历史结论与当前数据做连续性判断，不允许忽略上次观点。"
        "你只能输出一个合法 JSON 对象。"
        "不要输出 Markdown，不要输出代码块，不要输出解释文字。"
        "所有 key 必须使用双引号。"
        "如果缺少信息，请使用 null、空数组或保守结论，不要省略必填字段。"
        "输出要清晰完整，在保证结构化的前提下适当展开关键信号、关键价位和主要依据。"
    )

    previous_payload = previous_memory.model_dump(mode="json") if previous_memory else {
        "status": "initial_analysis",
        "message": "No previous memory available.",
    }
    previous_record_payload = previous_record.model_dump(mode="json") if previous_record else None

    user_payload = {
        "task": "基于上一次判断和当前数据进行趋势延续/修正分析",
        "phase": phase,
        "snapshot": snapshot.model_dump(mode="json"),
        "previous_memory": previous_payload,
        "previous_record": previous_record_payload,
        "market_context": market_context,
        "previous_trading_snapshot": previous_trading_snapshot,
        "screening_context": screening_context,
        "symbol_context": {
            "is_default_pool_symbol": is_default_pool_symbol,
            "position_status": position_status,
        },
        "field_unit_hints": {
            "snapshot.amount": (
                "成交额，TuShare 日线字段原始单位为千元；如需换算为亿元，请使用 amount / 100000。"
                "例如 amount=128386.959 时，只能写成 1.284 亿或约 1.28 亿，不能写成 12.84 亿。"
            ),
            "market_context.current_daily_bar.amount": (
                "成交额，单位同 snapshot.amount，为千元；换算为亿元请除以 100000。"
                "例如 amount=128386.959 时，应写成 1.284 亿。"
            ),
            "market_context.previous_daily_bar.amount": (
                "成交额，单位同 snapshot.amount，为千元；换算为亿元请除以 100000。"
                "禁止把千元误当成万元或亿元。"
            ),
            "market_context.recent_daily_bars[].amount": (
                "成交额，单位同 snapshot.amount，为千元；换算为亿元请除以 100000。"
                "若文案写'成交额 X 亿'，必须先完成换算再输出。"
            ),
            "screening_context": (
                "智能选股系统辅助上下文，包含模型排序、推荐分、风险分、资金、行业热度和Top3相对位置。"
                "它用于补充多维判断，不能替代 snapshot 与 market_context 的实时行情结论。"
            ),
        },
        "time_context": _build_time_context(
            phase=phase,
            snapshot=snapshot,
            previous_memory=previous_memory,
            previous_record=previous_record,
            market_context=market_context,
            previous_trading_snapshot=previous_trading_snapshot,
        ),
        "output_schema": {
            "ts_code": "string",
            "phase": "morning|afternoon|review",
            "trend_judgement": "string",
            "trend_breakdown": {
                "short_term": "bullish|neutral|bearish",
                "mid_term": "bullish|neutral|bearish",
                "long_term": "bullish|neutral|bearish",
                "short_term_reason": "string",
                "mid_term_reason": "string",
                "long_term_reason": "string",
            },
            "previous_view_status": "confirmed|weakened|reversed|initial",
            "operation_advice": "string",
            "risk_warning": ["string"],
            "observation_points": ["string"],
            "summary_markdown": "string",
            "decision": {
                "signal": "buy|hold|reduce|sell|avoid",
                "rationale": "string",
                "entry_zone": {
                    "low": "number|null",
                    "high": "number|null",
                },
                "stop_loss": "number|null",
                "take_profit": ["number"],
                "invalidation_condition": "string",
                "holding_horizon": "intraday|swing|position",
                "confidence_score": "number between 0 and 1",
                "risk_reward_ratio": "number|null",
                "evidence": ["string"],
            },
            "prediction_windows": [
                {
                    "window": "next_1d|next_3d|next_5d",
                    "bias": "bullish|neutral|bearish",
                    "confidence_score": "number between 0 and 1",
                    "rationale": "string",
                }
            ],
            "memory": {
                "ts_code": "string",
                "phase": "morning|afternoon|review",
                "trend_bias": "bullish|neutral|bearish",
                "short_term_bias": "bullish|neutral|bearish|null",
                "mid_term_bias": "bullish|neutral|bearish|null",
                "long_term_bias": "bullish|neutral|bearish|null",
                "support_levels": ["number"],
                "resistance_levels": ["number"],
                "capital_flow_view": "string",
                "key_risks": ["string"],
                "next_checkpoints": ["string"],
                "confidence_score": "number between 0 and 1",
                "summary": "string",
            },
        },
        "analysis_instructions": [
            "明确判断上一次观点是延续、减弱、反转还是首次分析。",
            "当前分析所对应的日期只能以 snapshot.trade_date 与 time_context.current_trade_date 为准，不要根据系统当前时间、生成时间或 minute_summary 自行推断'今天'。",
            "历史观点状态只用于承接上一次分析结论，默认参考 previous_record 与 previous_memory，不要把它和上一交易日行情混为一谈。",
            "把 market_context 视为主要分析输入，其中 current_daily_bar、previous_daily_bar、recent_daily_bars、current_weekly_bar、previous_weekly_bar、recent_weekly_bars 为程序整理后的可靠行情上下文。",
            "如果 screening_context.data_available 为 true，必须吸收其中的 stock_context/latest_pool_state/latest_model_top3：说明该股是否进入最新智能选股池、全市场模型排序、Top3/候选状态、风险分与资金/行业证据；但不得因为模型排序单独给出买入结论。",
            "如果 screening_context.data_available 为 false 或 stock_context 缺失，要明确表示智能选股上下文不可用或该股未出现在最新模型池中，并回到行情、财务和历史观点进行判断。",
            "如果 screening_context.symbol_in_latest_pool 为 false 但存在 standalone_stock_context，要明确说明该股不在最新模型池中；同时吸收 standalone_stock_context 里的单股增强信息，包括 technical_snapshot、moneyflow_context、market_event_context、company_profile、earnings_context。若 technical_snapshot 中存在 ma20、distance_to_ma20_pct、risk_score、risk_flags、setup_notes，summary_markdown、trend_judgement 与 decision.rationale 至少要明确落地其中 2 项，不能只泛泛描述'震荡'或'转强'。",
            "如果 stock_context.standalone_score_context 为 true，stock_context.recommendation_score / overall_score 只是单股独立评估下的执行分与形态质量分，不代表全市场池内排序、Top100席位或今日Top3资格。",
            "对于默认跟踪池股票，若它不在最新智能选股Top100或Top3中，应把这作为相对强度不足的一个反证，而不是直接否定个股；仍需结合价格、量能、资金和持仓状态给出跟踪/观察/回避建议。",
            "若 screening_context 中存在 distribution_risk_score、distribution_risk_flags、top3_extreme_risk_blocked、top3_extreme_risk_reason、recent_runup_5d、turnover_spike_ratio，必须在风险提示或决策证据中解释其含义。",
            "若 screening_context 中存在 moneyflow_3d_value、recent_large_order_net_inflow、recent_super_large_order_net_inflow、industry_heat_score、industry_flow_bias，必须用于判断资金承接、行业共振或资金分歧。若 standalone_stock_context.moneyflow_context 存在 recent_3d_net_inflow、recent_large_order_net_inflow、recent_super_large_order_net_inflow，也必须在 summary_markdown、risk_warning 或 decision.evidence 中至少引用 1 项，说明是资金承接改善还是主力仍偏谨慎。",
            "若 screening_context 或 standalone_stock_context 中存在 top_list/top_list_summary/limit_list/limit_status，要把龙虎榜资金方向、上榜原因、涨跌停异动强弱写入交易结构判断；若存在 business_summary 或 earnings_forecast_summary，要补充主营逻辑、预告变化或业绩预期对当前走势的支持/拖累。",
            "凡是描述'收涨/收跌'、'放量/缩量'、'站上/跌破'、'今日/本次'相对变化时，优先基于 market_context.current_daily_bar 与 market_context.previous_daily_bar 做比较；仅在缺失时，才回退为当前 snapshot 截面描述。",
            "凡是引用 snapshot.amount 或 market_context 中的 amount 描述成交额时，必须按'千元'理解；若输出为'亿'，请用 amount / 100000 换算，禁止把千元误写成万元或亿元。",
            "成交额是高频易错项：若 snapshot.amount=128386.959，则成交额只能写成 1.284 亿、约 1.28 亿或 12838.6959 万，绝不能写成 12.84 亿。",
            "如果你在 trend_judgement、summary_markdown、operation_advice、risk_warning、observation_points、decision.rationale、decision.evidence、prediction_windows.rationale 或 memory.* 中写到'成交额 X 亿'，输出前必须逐项核对 X 是否等于 amount / 100000。",
            "若无法确认换算结果，宁可直接写'成交额放大/缩量'，也不要输出具体的'X 亿'数值。",
            "涉及历史比较时，优先引用 time_context 中的 previous_trade_date、previous_analysis_generated_at 与 current_trade_date。",
            "如果 market_context.previous_daily_bar 存在，优先引用其中的 trade_date 作为上一交易日，不要直接拿上一次分析日期代替上一交易日。",
            "默认不要使用'昨日'或'今日'这类相对时间词；只有在 current_trade_date 与上一交易日明确相邻时，才允许写'上一交易日'，否则统一写成具体日期或'上次分析'。",
            "跨周末或节假日时，禁止把上次分析直接表述为'昨日'。",
            "必须同时给出短线、中线、长线三层趋势判断，并分别解释依据。",
            "短线优先结合 minute_summary、market_context.current_daily_bar、market_context.previous_daily_bar 与 market_context.recent_daily_bars。",
            "若 snapshot.financial_indicators 或 snapshot.earnings_express 存在，必须吸收其中的最新盈利能力、增长、快报线索到趋势判断、操作建议与风险提示里，不能只看价格。",
            "当 snapshot.earnings_express 非空时，要优先参考最新快报中的营收、净利润、ROE、EPS 同比变化，判断基本面边际改善还是走弱。",
            "中线优先结合 market_context.recent_daily_bars 与关键价位变化。",
            "长线优先结合 market_context.current_weekly_bar、market_context.previous_weekly_bar 与 market_context.recent_weekly_bars。",
            "解释哪些价格、量能、资金流数据支持当前结论，描述尽量短句化。",
            "如果历史观点被推翻，指出被推翻的原因。",
            "必须给出 next_1d、next_3d、next_5d 三个预测窗口的方向判断与信心分数。",
            "输出结构化交易决策，包括信号、入场区、止损位、止盈位和观点失效条件。",
            "如果暂不适合买入或卖出，signal 应明确给出 hold 或 avoid。",
            "如果 signal 为 avoid，不强制提供 entry_zone、stop_loss、take_profit；当判断属于'等待入场'或'观察回踩/突破'时，可以给出一个参考 entry_zone 作为关注区间。",
            "对于 avoid，若当前结论只是继续观望且没有明确触发位，可以留空 entry_zone、stop_loss、take_profit。",
            "对于 buy、hold、reduce、sell，优先结合支撑/阻力/最近高低点给出可执行价位：entry_zone 尽量提供 low 和 high，stop_loss 尽量提供具体数值，take_profit 至少给出一个目标位。",
            "如果 symbol_context.is_default_pool_symbol 为 true，这只股票视为重点跟踪标的，优先给出可执行的仓位管理意见，不要轻易输出泛化观望。",
            "如果 symbol_context.position_status 为 holding，优先在 hold、reduce、sell 中做决策；只有明确适合继续加仓时才给 buy，并尽量提供 stop_loss、take_profit 与失效条件。",
            "如果 symbol_context.position_status 为 watching，优先在 buy 与 avoid 中做决策；若暂不入场但存在明确触发位，可使用 avoid 并给出 entry_zone 作为观察区间。",
            "如果 symbol_context.is_default_pool_symbol 为 true 且 position_status 为空，也要尽量给出可执行参考；只有在边界不清晰时才使用 avoid。",
            "给出可执行但克制的操作建议，不要承诺收益。",
            "trend_judgement 控制在 120 字内。",
            "trend_breakdown 中每个 reason 控制在 180 字内。",
            "operation_advice 控制在 140 字内。",
            "risk_warning 最多 5 条，每条控制在 90 字内。",
            "observation_points 最多 5 条，每条控制在 90 字内。",
            "decision.rationale 控制在 220 字内，decision.evidence 最多 6 条。",
            "prediction_windows 每条 rationale 控制在 140 字内。",
            "summary_markdown 作为详情页主要阅读内容，目标 260-420 个中文字符，最多 600 个中文字符；要覆盖趋势结论、关键价位/均线、量能或资金、主要风险与操作前提，不要只写一句泛化摘要。memory.summary 控制在 260 字内。",
        ],
    }

    return system_prompt, json.dumps(user_payload, ensure_ascii=False, indent=2)



def build_single_stock_main_analysis_rewrite_prompt(
    *,
    stock_payload: dict[str, object],
    screening_context: Optional[dict[str, object]] = None,
) -> tuple[str, str]:
    system_prompt = (
        "你是一名擅长 A 股盘后复盘的重点个股分析师。"
        "你的任务不是重复结构化字段，而是把技术结构、关键价位、资金承接、风险分歧、龙虎榜/异动、业绩催化串成一段自然、具体、可读的主分析。"
        "只输出纯文本，不要输出 JSON，不要输出 Markdown 标题，不要列表化堆砌。"
        "避免模板腔，避免机械重复‘综合来看’‘操作上先看’‘它能进入重点名单’这类话术。"
    )
    user_payload = {
        "task": "将单股结构化分析重写为一段更自然的重点个股主分析",
        "stock_payload": stock_payload,
        "screening_context": screening_context,
        "writing_requirements": [
            "输出 1 段主分析，目标长度 220 到 350 字。",
            "必须优先串联以下维度中的可用信息：技术结构、MA20 或关键支撑压力、资金流、风险分歧、龙虎榜/涨跌停异动、业绩催化。",
            "如果某个维度没有数据就跳过，不要硬编。",
            "不要简单复述 summary_markdown、operation_advice、decision.rationale 原句；要重新组织成更自然的分析段落。",
            "如果存在 ma20、distance_to_ma20_pct、recent_5d_high、recent_5d_low，要明确解释当前价格所处位置、回踩/追高风险或突破门槛。",
            "如果存在 recent_3d_net_inflow、recent_large_order_net_inflow、recent_super_large_order_net_inflow 或 moneyflow_summary，要明确说明资金承接改善、分歧扩大还是主力仍偏谨慎。",
            "如果存在 risk_flags、distribution_risk_flags、recent_runup_5d、turnover_spike_ratio，要写出风险来自哪里，而不是只说‘风险较高’。",
            "如果存在 top_list_summary、limit_status，要解释这是情绪强化、资金背书，还是高波动信号。",
            "如果存在 earnings_forecast_summary、financial_indicators、earnings_express，要说明业绩催化对走势的支持或拖累。",
            "结尾允许给一句执行提醒，但不要把整段写成条目式操作指南。",
            "禁止输出空泛套话，禁止只堆数据。",
        ],
    }
    return system_prompt, json.dumps(user_payload, ensure_ascii=False, indent=2)



def _build_time_context(
    *,
    phase: AnalysisPhase,
    snapshot: PriceSnapshot,
    previous_memory: Optional[MemorySummary],
    previous_record: Optional[HistoricalAnalysisRecord],
    market_context: Optional[dict[str, object]],
    previous_trading_snapshot: Optional[dict[str, object]],
) -> dict[str, object]:
    previous_trade_date = previous_record.snapshot.trade_date if previous_record else None
    previous_generated_at = previous_record.generated_at if previous_record else (
        previous_memory.generated_at if previous_memory else None
    )
    current_trade_date = snapshot.trade_date
    previous_market_trade_date = _extract_trade_date(
        (market_context or {}).get("previous_daily_bar") if isinstance(market_context, dict) else previous_trading_snapshot
    )

    return {
        "current_phase": phase,
        "current_trade_date": current_trade_date,
        "current_trade_date_label": _format_trade_date(current_trade_date),
        "previous_analysis_phase": (
            previous_record.report.phase if previous_record else (previous_memory.phase if previous_memory else None)
        ),
        "previous_trade_date": previous_trade_date,
        "previous_trade_date_label": _format_trade_date(previous_trade_date),
        "previous_market_trade_date": previous_market_trade_date,
        "previous_market_trade_date_label": _format_trade_date(previous_market_trade_date),
        "previous_analysis_generated_at": (
            previous_generated_at.isoformat() if isinstance(previous_generated_at, datetime) else None
        ),
        "calendar_day_gap": _calculate_calendar_day_gap(previous_trade_date, current_trade_date),
        "has_previous_analysis": bool(previous_memory or previous_record),
        "has_previous_trading_snapshot": bool(previous_market_trade_date),
    }


def _format_trade_date(value: Optional[str]) -> Optional[str]:
    if not value or len(value) != 8:
        return value
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def _calculate_calendar_day_gap(previous_trade_date: Optional[str], current_trade_date: Optional[str]) -> Optional[int]:
    if not previous_trade_date or not current_trade_date:
        return None
    try:
        previous_day = datetime.strptime(previous_trade_date, "%Y%m%d")
        current_day = datetime.strptime(current_trade_date, "%Y%m%d")
    except ValueError:
        return None
    return (current_day - previous_day).days


def _extract_trade_date(previous_trading_snapshot: Optional[dict[str, object]]) -> Optional[str]:
    if not isinstance(previous_trading_snapshot, dict):
        return None
    value = previous_trading_snapshot.get("trade_date")
    if value is None:
        return None
    text = str(value).strip()
    return text or None
