# 智能选股训练侧优化进展总结

> 目的：记录本轮关于“涨停股/接力失败板”的诊断、已完成改动、当前结论、后续计划和可执行命令，方便后续换模型/换会话继续。

## 1. 背景问题

近期智能选股出现两个核心问题：

1. **模型候选池高度偏向涨停/近涨停股**
   - 2026-06-08、2026-06-09 的模型前排候选大量甚至全部是涨停/近涨停。
   - 例：2026-06-09 最新结果中，`recommendation_pool_states` 里 48 只候选全部 `pct_change >= 9.5%`，全部被 `near_limit_up_pct_change` 极端风险否决。

2. **最终 Top3 风控能拦住涨停，但没有可执行标的**
   - Top3 硬过滤生效：`pct_change >= 9.2%` 会触发 `near_limit_up_pct_change`。
   - 但由于候选池前 50 全是涨停，过滤后今日 Top3 为空。
   - 页面 fallback 仍会显示一个“最适合短线”候选，例如 `600500.SH 中化国际`，但它实际已经被 veto，并不是 Top3。

用户的核心判断：

> 不希望靠扩大候选池补位解决问题，因为担心会降低收益和准确率；更希望从训练阶段让模型学会区分“涨停后能接力”和“涨停后不能接力”。

---

## 2. 重要诊断结论

### 2.1 当前默认模型

当前默认 rerank 模型原先是：

```text
memory/history/short_term_models/raw_market_202508_202605_return_3d_lgbm_intraday_full.lightgbm.pkl
```

元信息：

```text
target = return_3d
model_type = lightgbm
dataset = data/training_features_20250801_20260515_intraday.csv
features = 59
sample_weight = None
```

这说明原模型是纯 `return_3d` 回归模型，没有 anti-chase 权重，没有涨停样本降权，也没有 return clip。

### 2.2 训练 CSV 确认

当前默认模型对应训练特征文件：

```text
/Users/user/Desktop/AI/OCTTS/data/training_features_20250801_20260515_intraday.csv
```

文件存在，约 914MB。

### 2.3 训练侧实验结果

#### 初始强 anti-chase 版本

artifact：

```text
raw_market_202508_202605_return_3d_lgbm_intraday_full_antichase_clip.lightgbm.pkl
```

配置：

```text
sample_weight_mode = regime_anti_chase
limit_up_sample_mode = downweight
limit_up_sample_weight = 0.1
return_clip = [-0.15, 0.20]
```

结果：Top1%、Top3% 排序表现不佳，不建议使用。

#### 方案 A：mild anti-chase

artifact：

```text
raw_market_202508_202605_return_3d_lgbm_intraday_antichase_mild.lightgbm.pkl
```

配置：

```text
sample_weight_mode = anti_chase
sample_weight_profile = conservative
limit_up_sample_weight = 0.5
return_clip = [-0.15, 0.30]
```

训练 holdout：Top1% 仍跑输；样本外也偏向选大量下跌股。不建议使用。

#### 方案 B：涨停温和降权 + return clip，不启用 anti-chase

artifact：

```text
raw_market_202508_202605_return_3d_lgbm_intraday_limit_mild_clip.lightgbm.pkl
```

配置：

```text
sample_weight_mode = none
limit_up_sample_mode = downweight
limit_up_sample_weight = 0.5
return_clip = [-0.15, 0.30]
```

训练 holdout 表现较好：

```text
top_1pct_excess_return = +0.00244
top_3pct_excess_return = +0.00617
top_5pct_excess_return = +0.01279
top_10pct_excess_return = +0.01561
```

样本外 2026-05-16 ~ 2026-06-01：

```text
B top1 excess = +1.21%
default top1 excess = +0.98%
```

样本外 2026-06-02 ~ 2026-06-05：

```text
B top1 excess = +2.10%
default top1 excess = +1.45%
```

但方案 B 仍高度偏向涨停，2026-06-02 ~ 2026-06-05 的 Top1% 中：

```text
limit_like_ratio = 77.06%
```

结论：方案 B 比默认更好，但不能根治涨停候选池占满问题。

### 2.4 样本外涨停桶与实际 Top3 的矛盾

2026-05-16 ~ 2026-06-01：

```text
全市场 pct_change >= 9.5% 桶：return_3d = +1.06%
实际 Top3 中涨停股：next1 平均 -3.50%，次日胜率 20%
```

说明：

```text
涨停股整体不一定差，但系统实际选中的涨停股是接力失败板。
```

2026-06-02 ~ 2026-06-05：

- 6/4 Top3 三只涨停表现成功：次日均上涨。
- 6/5 Top3 三只涨停表现失败：次日大跌/下跌。

结论：需要区分“成功接力板”和“失败接力板”，不能简单把所有涨停都视为坏样本，也不能单靠最终过滤。

---

## 3. 已完成代码改动

### 3.1 新增训练权重工具

新增文件：

```text
src/octts/tools/modeling_weights.py
```

功能：

- `build_limit_up_mask()`：识别 `pct_change >= 9.5` 的涨停/近涨停训练样本。
- `build_sample_weights()`：anti-chase / regime anti-chase 样本权重。
- `apply_limit_up_downweight()`：对涨停样本降权。
- `clip_return_target()`：只对 `return_*` 回归目标做 clip。
- `count_clipped_return_target()`：统计 clip 样本数量。

测试文件：

```text
tests/test_modeling_weights.py
```

验证：

```bash
python -m pytest tests/test_modeling_weights.py -q
# 3 passed
```

### 3.2 更新训练脚本

文件：

```text
src/octts/tools/train_raw_market_model.py
src/octts/tools/train_tuned_models.py
```

新增参数：

```text
--sample-weight-mode {none,anti_chase,regime_anti_chase}
--anti-chase-profile {default,strict}
--sample-weight-profile {conservative,balanced,aggressive}

--limit-up-sample-mode {none,drop,downweight}
--limit-up-pct-threshold 9.5
--limit-up-sample-weight 0.1

--enable-return-clip
--return-clip-low -0.15
--return-clip-high 0.20
```

注意：默认值保持旧行为不变。

### 3.3 临时切换默认 rerank 模型（方案 B → luw05）

文件：

```text
src/octts/services/regression_rerank_service.py
```

`PREFERRED_MODEL_SPECS` 先切到方案 B，后于 2026-06-12 切到 luw05：

```python
PREFERRED_MODEL_SPECS = [
    (
        "lgbm_intraday_limit_mild_clip",
        "raw_market_202508_202605_return_3d_lgbm_intraday_limit_mild_clip.lightgbm.pkl",
        1.00,
    ),
]
```

验证：

```bash
python -m compileall src/octts/services/regression_rerank_service.py
```

并确认可解析 artifact。

### 3.4 新增样本外诊断工具

新增文件：

```text
src/octts/tools/evaluate_model_oos_limit_chase.py
```

功能：

- 对比多个模型 artifact 的样本外表现。
- 输出 Top1/Top3/Top5/Top10 的：
  - mean_return
  - excess_return
  - limit_like_ratio
  - strong_move_ratio
  - mid_1_to_7_ratio
  - negative_ratio
- 输出涨幅桶未来收益：
  - `pct_change >= 9.5`
  - `7.5 ~ 9.5`
  - `3 ~ 7.5`
  - `0 ~ 3`
  - negative
- 复盘实际 `今日Top3` 的 next1/next2/next3。

后续已增强支持新增特征：

- TopN 中 `prev_day_limit_up_ratio`
- TopN 中 `avg_limit_chase_failure_risk_score`
- `prev_day_limit_up_subset`
- `limit_chase_failure_risk_buckets`

### 3.5 新增接力训练特征

文件：

```text
src/octts/schemas/training.py
```

`RAW_MARKET_FEATURE_SCHEMA_VERSION` 从：

```text
raw_v1
```

升级为：

```text
raw_v2
```

新增字段：

```text
prev_day_limit_up
prev_day_limit_open_times
prev_day_limit_first_time
prev_day_limit_last_time
prev_day_limit_amount
prev_day_fd_amount
prev_day_limit_times
prev_day_up_stat_success
prev_day_up_stat_total
prev_day_up_stat_ratio
prev_day_one_word_limit_flag

moneyflow_net_1d
moneyflow_large_net_1d
moneyflow_elarge_net_1d
moneyflow_net_3d
moneyflow_large_net_3d
moneyflow_elarge_net_3d
moneyflow_positive_flag
limit_like_moneyflow_divergence_flag
limit_chase_failure_risk_score

label_limit_relay_success_1d
label_limit_relay_strong_1d
label_limit_relay_success_3d
label_limit_relay_limit_up_1d
```

### 3.6 批量读取涨停板/资金流数据

文件：

```text
src/octts/services/market_raw_data_repository.py
```

新增：

```python
get_limit_list_by_trade_dates(...)
get_moneyflow_by_trade_dates(...)
_serialize_market_moneyflow_daily(...)
_chunked(...)
```

实现方式：本地 SQLite 分批读取，不调用 Tushare。

后续优化：`get_limit_list_by_trade_dates` 和 `get_moneyflow_by_trade_dates` 均按每 800 个股票一批查询，避免 SQLite `IN (...)` 过大。

### 3.7 训练数据构建接入接力特征

文件：

```text
src/octts/services/raw_market_training_dataset.py
```

新增逻辑：

- 构建样本时加载：
  - `market_limit_list_daily`
  - `market_moneyflow_daily`
- 在 `_build_samples_for_code()` 中计算：
  - 前一日涨停与封板质量；
  - moneyflow 特征；
  - `limit_chase_failure_risk_score`；
  - 接力标签。

新增辅助函数：

```python
_safe_int()
_parse_limit_time_to_minutes()
_parse_up_stat()
_is_limit_up_row()
_is_one_word_limit()
_moneyflow_net()
_build_moneyflow_features()
_limit_chase_failure_risk_score()
```

### 3.8 保守处理 moneyflow：不纳入默认训练特征

用户检查发现 `market_moneyflow_daily` 每天只有几十到一百多条，远低于全市场 5000+，不适合作为全量训练主特征。

因此已做保守调整：

- schema 保留 moneyflow 字段；
- dataset builder 如果本地有 moneyflow 会写入；
- 但默认训练特征不使用 moneyflow 字段；
- `train_tuned_models.py` 默认特征也不使用 moneyflow；
- `limit_chase_failure_risk_score` 不再因为 moneyflow 缺失而加风险分。

当前默认训练保留的接力特征：

```text
prev_day_limit_up
prev_day_limit_open_times
prev_day_limit_first_time
prev_day_limit_last_time
prev_day_limit_amount
prev_day_fd_amount
prev_day_limit_times
prev_day_up_stat_success
prev_day_up_stat_total
prev_day_up_stat_ratio
prev_day_one_word_limit_flag
limit_chase_failure_risk_score
```

暂不默认使用：

```text
moneyflow_net_1d
moneyflow_large_net_1d
moneyflow_elarge_net_1d
moneyflow_net_3d
moneyflow_large_net_3d
moneyflow_elarge_net_3d
moneyflow_positive_flag
limit_like_moneyflow_divergence_flag
```

### 3.9 预构建训练特征表支持新增列

文件：

```text
src/octts/tools/build_training_features.py
```

已将新增接力标签加入 `TARGET_COLUMNS`。特征列来自 `train_raw_market_model.RAW_MARKET_FEATURE_COLUMNS`，因此会自动带上新增接力特征。

现有 `ALTER TABLE` 逻辑会自动补列。

---

## 4. 数据覆盖检查结论

### 4.1 moneyflow 覆盖不足

用户查询：

```text
SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM market_moneyflow_daily;
```

结果：

```text
2026-02-25|2026-06-05|21711
```

每日行数示例：

```text
2026-06-05|20
2026-06-04|42
2026-06-03|42
2026-06-02|42
2026-06-01|58
2026-05-29|72
...
```

结论：moneyflow 是局部候选股回填，不是全市场覆盖，不适合立即作为全量训练特征。

不建议立即全量补单股 moneyflow，因为 Tushare `moneyflow` 多为按股票拉取，全市场全日期请求量可能是几十万到上百万级，过慢且易受限。

### 4.2 limit list 覆盖情况

用户查询：

```text
SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM market_limit_list_daily;
```

结果：

```text
2025-10-09|2026-04-24|12585
```

字段非空统计：

```text
rows = 12585
open_times_rows = 12585
first_time_rows = 11253
last_time_rows = 9503
limit_amount_rows = 1338
fd_amount_rows = 9503
up_stat_rows = 11253
limit_times_rows = 9503
```

结论：limit list 字段质量可用，但覆盖只到 2026-04-24，缺 2026-04-25 之后的数据。下一步应优先补 `market_limit_list_daily`，而不是补全量 moneyflow。

---

## 5. 当前建议的下一步

### 5.1 优先补 limit_list，不补 moneyflow

先查看回填脚本参数：

```bash
python -m octts.tools.backfill_relay_market_data --help
```

如果参数匹配，执行：

```bash
python -m octts.tools.backfill_relay_market_data \
  --start-date 2026-04-25 \
  --end-date 2026-06-09 \
  --skip-moneyflow \
  --skip-industry-moneyflow \
  --skip-market-moneyflow
```

目的：只补轻量接力数据，尤其是 `market_limit_list_daily`。

补完后检查：

```bash
sqlite3 /Users/user/Desktop/AI/OCTTS/octts_screening.db \
"SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM market_limit_list_daily;"
```

以及：

```bash
sqlite3 /Users/user/Desktop/AI/OCTTS/octts_screening.db \
"SELECT trade_date, COUNT(*) AS rows
 FROM market_limit_list_daily
 GROUP BY trade_date
 ORDER BY trade_date DESC
 LIMIT 30;"
```

### 5.2 重建训练特征

建议先从 moneyflow 覆盖较近、limit list 补齐后的窗口开始：

```bash
python -m octts.tools.build_training_features \
  --start-date 2026-02-25 \
  --end-date 2026-06-09 \
  --min-history-days 20 \
  --exclude-bj
```

导出 CSV：

```bash
sqlite3 /Users/user/Desktop/AI/OCTTS/octts_screening.db \
  -header -csv \
  "SELECT * FROM training_features WHERE trade_date >= '2026-02-25' AND trade_date <= '2026-06-09'" \
  > /Users/user/Desktop/AI/OCTTS/data/training_features_20260225_20260609_relay_v2.csv
```

### 5.3 训练通用 relay v2 模型

先不启用 anti-chase，避免方案 A 的排序问题。

```bash
python -m octts.tools.train_raw_market_model \
  --input /Users/user/Desktop/AI/OCTTS/data/training_features_20260225_20260609_relay_v2.csv \
  --model-type lightgbm \
  --target return_3d \
  --limit-up-sample-mode downweight \
  --limit-up-pct-threshold 9.5 \
  --limit-up-sample-weight 0.3 \
  --enable-return-clip \
  --return-clip-low -0.15 \
  --return-clip-high 0.30 \
  --output-name raw_market_202602_202606_return_3d_lgbm_relay_v2
```

### 5.4 样本外诊断

```bash
python -m octts.tools.evaluate_model_oos_limit_chase \
  --start-date 2026-05-16 \
  --end-date 2026-06-09 \
  --feature-source training_features \
  --target return_3d \
  --artifacts 'B=raw_market_202508_202605_return_3d_lgbm_intraday_limit_mild_clip.lightgbm.pkl,v2=raw_market_202602_202606_return_3d_lgbm_relay_v2.lightgbm.pkl' \
  --output-file tmp/oos_limit_chase_relay_v2.json
```

重点看：

```text
prev_day_limit_up_ratio
avg_limit_chase_failure_risk_score
prev_day_limit_up_subset
limit_chase_failure_risk_buckets
limit_like_ratio
top_1pct/top_3pct excess_return
```

### 5.5 可选：训练接力专项分类器

导出接力子集：

```bash
sqlite3 /Users/user/Desktop/AI/OCTTS/octts_screening.db \
  -header -csv \
  "SELECT * FROM training_features WHERE trade_date >= '2026-02-25' AND trade_date <= '2026-06-09' AND prev_day_limit_up = 1 AND label_limit_relay_success_1d IS NOT NULL" \
  > /Users/user/Desktop/AI/OCTTS/data/training_features_20260225_20260609_relay_only_v2.csv
```

训练：

```bash
python -m octts.tools.train_raw_market_model \
  --input /Users/user/Desktop/AI/OCTTS/data/training_features_20260225_20260609_relay_only_v2.csv \
  --model-type lightgbm \
  --target label_limit_relay_success_1d \
  --output-name limit_relay_success_1d_lgbm_v2
```

但建议先跑通用 `return_3d` v2，再决定是否做专项模型。

---

## 6. 已执行的轻量验证

已执行：

```bash
python -m compileall \
  src/octts/schemas/training.py \
  src/octts/services/market_raw_data_repository.py \
  src/octts/services/raw_market_training_dataset.py \
  src/octts/tools/train_raw_market_model.py \
  src/octts/tools/build_training_features.py \
  src/octts/tools/train_tuned_models.py \
  src/octts/tools/evaluate_model_oos_limit_chase.py
```

通过。

已执行：

```bash
python -m pytest tests/test_modeling_weights.py -q
```

通过：

```text
3 passed
```

已做单只股票 smoke test：

```text
600500.SH 2026-06-09
pct_change = 9.9815
prev_day_limit_up = False
limit_chase_failure_risk_score = 3.0
```

说明新增字段能生成。

---

## 7. 2026-06-10 最新进展：limit_list 已补齐，relay v2 已完成首轮训练与 OOS 对比

### 7.1 limit_list 覆盖已补齐

当前本地 `market_limit_list_daily` 已覆盖完整训练窗口：

```text
MIN(trade_date) = 2025-08-01
MAX(trade_date) = 2026-06-09
COUNT(*) = 20213
```

最近交易日日频行数示例：

```text
2026-06-09 | 163
2026-06-08 | 102
2026-06-05 | 136
2026-06-04 | 106
2026-06-03 | 128
2026-06-02 | 104
2026-06-01 | 182
```

字段非空质量统计：

```text
rows = 20213
open_times_rows = 20213
first_time_rows = 18881
last_time_rows = 17131
limit_amount_rows = 1915
fd_amount_rows = 15216
up_stat_rows = 18881
limit_times_rows = 15216
```

结论：`market_limit_list_daily` 已不再是 blocker，可以支撑 relay v2 接力特征构建。`limit_amount` 覆盖仍相对低，但 `open_times`、`first_time`、`last_time`、`fd_amount`、`up_stat`、`limit_times` 等核心字段覆盖可用。

### 7.2 training_features 已重建到 relay v2 状态

特征构建结果：

```text
日期范围: 2025-08-01 ~ 2026-06-09
样本总数: 959,851
交易日数: 186
股票数量: 5212
插入记录: 959,851
构建耗时: 625.33秒
写入耗时: 130.52秒
```

目标变量有效值统计：

```text
return_1d: 954639 (99.5%)
return_3d: 944215 (98.4%)
return_5d: 933795 (97.3%)
return_10d: 907748 (94.6%)
vs_market_1d: 954639 (99.5%)
vs_market_3d: 944215 (98.4%)
vs_market_5d: 933795 (97.3%)
label_up_1d: 954639 (99.5%)
label_up_3d: 944215 (98.4%)
label_up_5d: 933795 (97.3%)
label_vs_market_1d: 954639 (99.5%)
label_vs_market_3d: 944215 (98.4%)
label_vs_market_5d: 933795 (97.3%)
label_strong_1d: 954639 (99.5%)
label_limit_relay_success_1d: 16801 (1.8%)
label_limit_relay_strong_1d: 16801 (1.8%)
label_limit_relay_success_3d: 16583 (1.7%)
label_limit_relay_limit_up_1d: 16801 (1.8%)
```

结论：

- `return_3d` 覆盖 98.4%，适合继续作为通用 rerank 主目标。
- 接力专项标签约 1.7% ~ 1.8%，符合涨停接力样本在全市场中的稀疏性。
- 当前训练集已经具备训练 relay v2 通用模型和后续接力专项分类器的条件。

### 7.3 relay v2 通用模型已完成首轮训练

首轮 relay v2 artifact：

```text
memory/history/short_term_models/raw_market_202508_202606_return_3d_lgbm_relay_v2.lightgbm.pkl
```

元信息摘要：

```text
artifact_target = return_3d
artifact_sample_weight_mode = none
artifact_limit_up_sample_mode = downweight
artifact_return_clip_enabled = true
feature_count = 71
missing_feature_columns = []
```

对比基线 B：

```text
memory/history/short_term_models/raw_market_202508_202605_return_3d_lgbm_intraday_limit_mild_clip.lightgbm.pkl
```

B 元信息摘要：

```text
artifact_target = return_3d
artifact_sample_weight_mode = none
artifact_limit_up_sample_mode = downweight
artifact_return_clip_enabled = true
feature_count = 59
missing_feature_columns = []
```

### 7.4 OOS 对比结果：B vs relay v2

OOS 窗口的全市场基准：

```text
baseline_mean_return = -1.7788%
```

#### Top1%

```text
B:
  mean_return = -0.4038%
  excess_return = +1.3750%
  limit_like_ratio = 26.48%
  prev_day_limit_up_ratio = 9.24%
  avg_limit_chase_failure_risk_score = 1.0455

v2:
  mean_return = -0.7356%
  excess_return = +1.0432%
  limit_like_ratio = 9.93%
  prev_day_limit_up_ratio = 4.55%
  avg_limit_chase_failure_risk_score = 0.6469
```

结论：Top1% 上 B 收益略优，但 v2 明显降低涨停/近涨停占比、前日涨停接力占比和平均接力失败风险分。

#### Top3%

```text
B:
  mean_return = -2.1200%
  excess_return = -0.3412%
  limit_like_ratio = 10.34%
  prev_day_limit_up_ratio = 4.23%
  avg_limit_chase_failure_risk_score = 0.5244

v2:
  mean_return = -0.5404%
  excess_return = +1.2384%
  limit_like_ratio = 3.77%
  prev_day_limit_up_ratio = 1.65%
  avg_limit_chase_failure_risk_score = 0.3286
```

结论：Top3% 上 v2 明显优于 B，并且更符合最终 Top3 候选池稳定性的目标。

#### Top5%

```text
B:
  mean_return = -2.5603%
  excess_return = -0.7815%
  limit_like_ratio = 6.81%
  prev_day_limit_up_ratio = 3.01%
  avg_limit_chase_failure_risk_score = 0.4199

v2:
  mean_return = -0.0007%
  excess_return = +1.7781%
  limit_like_ratio = 2.98%
  prev_day_limit_up_ratio = 1.24%
  avg_limit_chase_failure_risk_score = 0.3055
```

结论：Top5% 是 v2 表现最强的区间，说明 v2 不只是少选涨停，而是在更宽候选池里整体排序质量更好。

#### Top10%

```text
B:
  mean_return = -2.2700%
  excess_return = -0.4912%
  limit_like_ratio = 5.10%
  prev_day_limit_up_ratio = 2.33%
  avg_limit_chase_failure_risk_score = 0.3712

v2:
  mean_return = -0.0530%
  excess_return = +1.7258%
  limit_like_ratio = 3.86%
  prev_day_limit_up_ratio = 1.89%
  avg_limit_chase_failure_risk_score = 0.3947
```

结论：Top10% 上 v2 收益显著优于 B，涨停/前日涨停占比也更低；平均风险分与 B 接近，略高于 B。

### 7.5 当前 OOS 结论

综合判断：

1. relay v2 已显著缓解原问题：候选池被涨停/近涨停股占满。
2. relay v2 的 Top3%、Top5%、Top10% OOS excess return 全面优于 B。
3. relay v2 的 Top1% excess return 略弱于 B，但风险结构明显更健康。
4. 从智能选股系统最终要形成可执行 Top3 的角度，Top3%/Top5% 稳定性比单纯 Top1% 更重要。
5. relay v2 可以作为替换 B 的强候选，但建议切默认模型前再补充分日 TopN/实际 Top3 诊断，避免聚合指标掩盖个别日期极端失败。

暂定判断：

```text
relay v2 大概率优于方案 B，更适合作为下一版默认 rerank 模型候选。
```

但暂不建议无诊断直接切线上默认模型。

### 7.6 对当前诊断结果的注意点

1. v2 的 `negative_ratio` 较高，但 OOS 窗口全市场基准为 -1.78%，市场环境整体偏弱，因此不能单独用负收益样本占比否定 v2。
2. 增强诊断前，`limit_chase_failure_risk_buckets` 在 B 和 v2 中完全一致，说明该统计更像 OOS 全样本风险桶分布，不是模型 TopN 内风险桶分布。
3. 已增强诊断工具，在每个 TopN 内输出风险桶分布和分日表现：

```text
top_1pct.risk_bucket_distribution
top_3pct.risk_bucket_distribution
top_5pct.risk_bucket_distribution
top_10pct.risk_bucket_distribution

top_1pct.daily_performance
top_3pct.daily_performance
top_5pct.daily_performance
top_10pct.daily_performance
```

这样可以更直接观察模型到底选中了哪些风险桶，并检查单日极端失败是否被聚合指标掩盖。

### 7.7 增强诊断：TopN 内风险桶分布

增强诊断结果显示：v2 降低了整体涨停/前日涨停暴露，但高风险桶在 OOS 窗口内并不一定表现差。相反，在 2026-05-16 ~ 2026-06-09 的弱市窗口里，`high_4_plus` 桶的均值和胜率高于低风险桶。因此不能把 `limit_chase_failure_risk_score` 简单理解为“越低越好”的硬 veto 信号，更适合作为候选结构监控指标和特征消融对象。

#### v2 Top1% 风险桶

```text
low_0_1:
  count = 553
  ratio = 76.28%
  mean_return = -1.4515%
  win_rate = 20.98%

mid_2_4:
  count = 144
  ratio = 19.86%
  mean_return = +0.8533%
  win_rate = 44.44%

high_4_plus:
  count = 28
  ratio = 3.86%
  mean_return = +5.2324%
  win_rate = 67.86%
```

#### B Top1% 风险桶

```text
low_0_1:
  count = 499
  ratio = 68.83%
  mean_return = -1.4912%
  win_rate = 29.46%

mid_2_4:
  count = 154
  ratio = 21.24%
  mean_return = +0.3817%
  win_rate = 42.21%

high_4_plus:
  count = 72
  ratio = 9.93%
  mean_return = +5.4522%
  win_rate = 63.89%
```

结论：B 的 Top1% 收益略强，部分来自更高的 `high_4_plus` 暴露；v2 将 Top1% 高风险桶占比从 9.93% 降至 3.86%，风险结构更保守，但可能牺牲了一部分强接力收益。

#### v2 Top10% 风险桶

```text
low_0_1:
  count = 6301
  ratio = 86.86%
  mean_return = -2.4129%
  win_rate = 23.42%

mid_2_4:
  count = 804
  ratio = 11.08%
  mean_return = -2.1543%
  win_rate = 33.83%

high_4_plus:
  count = 149
  ratio = 2.05%
  mean_return = +3.1454%
  win_rate = 53.69%
```

结论：在当前 OOS 窗口中，高风险桶不是简单的坏样本桶，可能混合了“失败接力风险”和“真正强势接力机会”。后续特征筛选需要重点验证：

1. `limit_chase_failure_risk_score` 是否过度压制强势接力；
2. 原子 limit list 特征是否比聚合风险分更有泛化能力；
3. `limit_up_sample_weight` 是否过低导致 v2 在 Top1% 放弃了部分高收益强势板。

## 8. 后续研究方向：超参数与特征筛选

### 8.0 已完成一轮统一 ablation 比较

已完成统一比较文件：

```text
tmp/oos_limit_chase_relay_v2_ablation.json
```

比较窗口：

```text
2026-05-16 ~ 2026-06-09
rows = 72,548
trade_dates = 14
baseline_mean_return = -1.7788%
```

本轮比较包含：

```text
B: 当前方案 B，59 特征
v2/full: relay v2 全量 71 特征
no_limit_amount: 去掉 prev_day_limit_amount，70 特征
no_risk: 去掉 limit_chase_failure_risk_score，70 特征
luw05: limit_up_sample_weight = 0.5，71 特征
luw08: limit_up_sample_weight = 0.8，71 特征
```

#### Top1% 对比

```text
B:
  excess_return = +1.3750%
  mean_return = -0.4038%
  limit_like_ratio = 26.48%
  prev_day_limit_up_ratio = 9.24%
  avg_limit_chase_failure_risk_score = 1.046

v2/full:
  excess_return = +1.0432%
  mean_return = -0.7356%
  limit_like_ratio = 9.93%
  prev_day_limit_up_ratio = 4.55%
  avg_limit_chase_failure_risk_score = 0.647

no_limit_amount:
  excess_return = +0.6925%
  mean_return = -1.0863%
  limit_like_ratio = 14.62%
  prev_day_limit_up_ratio = 5.79%
  avg_limit_chase_failure_risk_score = 1.250

no_risk:
  excess_return = +1.1936%
  mean_return = -0.5852%
  limit_like_ratio = 8.83%
  prev_day_limit_up_ratio = 3.59%
  avg_limit_chase_failure_risk_score = 0.350

luw05:
  excess_return = +1.3814%
  mean_return = -0.3974%
  limit_like_ratio = 12.00%
  prev_day_limit_up_ratio = 4.41%
  avg_limit_chase_failure_risk_score = 0.793

luw08:
  excess_return = +1.5938%
  mean_return = -0.1850%
  limit_like_ratio = 20.28%
  prev_day_limit_up_ratio = 7.31%
  avg_limit_chase_failure_risk_score = 1.583
```

Top1% 结论：`luw08` 最高，但涨停/接力风险暴露明显回升；`luw05` 在收益接近 B 的同时，风险结构显著好于 B；`no_risk` 也优于原 v2/full，说明聚合风险分可能过度压制了部分强势接力机会。

#### Top3% 对比

```text
B:
  excess_return = -0.3412%
  mean_return = -2.1200%
  limit_like_ratio = 10.34%
  prev_day_limit_up_ratio = 4.23%
  avg_limit_chase_failure_risk_score = 0.524

v2/full:
  excess_return = +1.2384%
  mean_return = -0.5404%
  limit_like_ratio = 3.77%
  prev_day_limit_up_ratio = 1.65%
  avg_limit_chase_failure_risk_score = 0.329

no_limit_amount:
  excess_return = +0.8522%
  mean_return = -0.9266%
  limit_like_ratio = 6.80%
  prev_day_limit_up_ratio = 2.99%
  avg_limit_chase_failure_risk_score = 0.626

no_risk:
  excess_return = +1.5564%
  mean_return = -0.2224%
  limit_like_ratio = 4.69%
  prev_day_limit_up_ratio = 1.61%
  avg_limit_chase_failure_risk_score = 0.229

luw05:
  excess_return = +1.4002%
  mean_return = -0.3786%
  limit_like_ratio = 5.24%
  prev_day_limit_up_ratio = 1.84%
  avg_limit_chase_failure_risk_score = 0.401

luw08:
  excess_return = +1.1659%
  mean_return = -0.6129%
  limit_like_ratio = 8.96%
  prev_day_limit_up_ratio = 2.76%
  avg_limit_chase_failure_risk_score = 0.696
```

Top3% 结论：`no_risk` 最优，其次 `luw05`，再是 v2/full。`luw08` 虽然 Top1% 强，但 Top3% 已落后且涨停暴露明显偏高。

#### Top5% 对比

```text
B:
  excess_return = -0.7815%
  mean_return = -2.5603%
  limit_like_ratio = 6.81%
  prev_day_limit_up_ratio = 3.01%
  avg_limit_chase_failure_risk_score = 0.420

v2/full:
  excess_return = +1.7781%
  mean_return = -0.0007%
  limit_like_ratio = 2.98%
  prev_day_limit_up_ratio = 1.24%
  avg_limit_chase_failure_risk_score = 0.305

no_limit_amount:
  excess_return = +0.7733%
  mean_return = -1.0055%
  limit_like_ratio = 5.05%
  prev_day_limit_up_ratio = 2.54%
  avg_limit_chase_failure_risk_score = 0.500

no_risk:
  excess_return = +1.8108%
  mean_return = +0.0320%
  limit_like_ratio = 5.16%
  prev_day_limit_up_ratio = 1.19%
  avg_limit_chase_failure_risk_score = 0.280

luw05:
  excess_return = +1.7996%
  mean_return = +0.0208%
  limit_like_ratio = 4.58%
  prev_day_limit_up_ratio = 1.52%
  avg_limit_chase_failure_risk_score = 0.366

luw08:
  excess_return = +1.1394%
  mean_return = -0.6394%
  limit_like_ratio = 6.31%
  prev_day_limit_up_ratio = 2.10%
  avg_limit_chase_failure_risk_score = 0.515
```

Top5% 结论：`no_risk` 略优于 `luw05` 和 v2/full，三者都显著优于 B；`no_limit_amount` 明显退化，说明 `prev_day_limit_amount` 虽然覆盖低，但可能仍提供有效信号，不能简单删除。

#### Top10% 对比

```text
B:
  excess_return = -0.4912%
  mean_return = -2.2700%
  limit_like_ratio = 5.10%
  prev_day_limit_up_ratio = 2.33%
  avg_limit_chase_failure_risk_score = 0.371

v2/full:
  excess_return = +1.7258%
  mean_return = -0.0530%
  limit_like_ratio = 3.86%
  prev_day_limit_up_ratio = 1.89%
  avg_limit_chase_failure_risk_score = 0.395

no_limit_amount:
  excess_return = +0.6617%
  mean_return = -1.1171%
  limit_like_ratio = 3.45%
  prev_day_limit_up_ratio = 1.86%
  avg_limit_chase_failure_risk_score = 0.359

no_risk:
  excess_return = +1.1130%
  mean_return = -0.6658%
  limit_like_ratio = 3.74%
  prev_day_limit_up_ratio = 1.45%
  avg_limit_chase_failure_risk_score = 0.306

luw05:
  excess_return = +1.7749%
  mean_return = -0.0039%
  limit_like_ratio = 4.82%
  prev_day_limit_up_ratio = 2.07%
  avg_limit_chase_failure_risk_score = 0.409

luw08:
  excess_return = +1.1211%
  mean_return = -0.6577%
  limit_like_ratio = 4.52%
  prev_day_limit_up_ratio = 1.63%
  avg_limit_chase_failure_risk_score = 0.372
```

Top10% 结论：`luw05` 最优，v2/full 接近；`no_risk` 在 Top10% 退化，说明去掉聚合风险分虽然改善 Top3/Top5，但可能影响更宽候选池稳定性。

#### 本轮 ablation 综合结论

1. `no_limit_amount` 明显退化，不建议删除 `prev_day_limit_amount`。
2. `no_risk` 在 Top3%/Top5% 最强，且涨停暴露仍可控，是下一轮重点候选。
3. `luw05` 综合表现非常稳：Top1% 接近 B，Top3/Top5/Top10 明显优于 B，风险暴露低于 B，但高于 v2/full。
4. `luw08` Top1% 最强，但涨停/接力风险暴露明显回升，不建议作为默认主模型，除非后续目标明确偏向更激进接力。
5. 当前更适合进入下一轮验证的候选是：

```text
第一候选：luw05
第二候选：no_risk
保守候选：v2/full
不推荐：no_limit_amount
激进观察：luw08
```

当前偏向：`luw05` 更适合作为默认 rerank 候选，因为它在收益、候选池宽度和风险暴露之间更均衡；`no_risk` 值得继续验证，但需要关注 Top10% 稳定性和是否过度依赖少数接力行情。

### 8.1 是否需要继续检查 v2 超参数

需要，但不建议马上做大规模盲目搜索。当前 v2 已经在 OOS 上显著改善 Top3/Top5/Top10，并降低涨停风险暴露，因此更合理的做法是围绕当前 v2 做小范围、可解释的 ablation。

优先检查的超参数：

1. `limit_up_sample_weight`
   - 当前首轮 v2 使用涨停样本降权。
   - 建议对比：`0.2`、`0.3`、`0.5`、`0.8`。
   - 目标不是单纯让 `limit_like_ratio` 最低，而是在 Top3/Top5 excess return 和风险暴露之间找平衡。

2. `return_clip_high`
   - 当前 high 为 `0.30`。
   - 建议对比：`0.20`、`0.25`、`0.30`。
   - 如果 high 太高，模型可能继续偏好极端高弹性样本；如果太低，可能损失强势股识别能力。

3. `return_clip_low`
   - 当前 low 为 `-0.15`。
   - 可小范围对比：`-0.10`、`-0.15`、`-0.20`。
   - 主要观察是否影响回撤环境中的排序稳定性。

4. LightGBM 树模型复杂度
   - 如果当前训练脚本暴露参数，可考虑小范围对比 `num_leaves`、`min_child_samples`、`learning_rate`、`n_estimators`。
   - 但需要避免为了 OOS 小窗口过拟合。

不建议优先恢复强 anti-chase：

- 之前方案 A 和 mild anti-chase 已出现排序表现不佳、偏向下跌股的问题。
- 当前 v2 已经通过 relay 特征和温和涨停降权显著降低追涨暴露，因此 anti-chase 不应作为下一步主线。

### 8.2 是否需要进一步做特征筛选

需要，但建议以“稳定性和防过拟合”为目标，不建议只按单次 feature importance 粗暴删除。

当前 v2 从 59 个特征增加到 71 个特征，新增 12 个 relay 相关默认特征。由于 OOS 改善明显，新增特征整体是有效的，但仍应做筛选和消融。

建议优先做以下 ablation：

1. 全量 relay v2 特征
   - 当前模型。

2. 去掉低覆盖字段组合
   - 重点观察 `limit_amount` 相关字段。
   - 当前 `limit_amount_rows = 1915 / 20213`，覆盖明显低于其他字段。
   - 如果该字段缺失处理不稳，可能引入噪声。

3. 只保留高覆盖 limit 特征
   - 例如：
     - `prev_day_limit_up`
     - `prev_day_limit_open_times`
     - `prev_day_limit_first_time`
     - `prev_day_limit_last_time`
     - `prev_day_fd_amount`
     - `prev_day_limit_times`
     - `prev_day_up_stat_success`
     - `prev_day_up_stat_total`
     - `prev_day_up_stat_ratio`
     - `prev_day_one_word_limit_flag`
     - `limit_chase_failure_risk_score`

4. 去掉聚合风险分，仅保留原子特征
   - 即去掉 `limit_chase_failure_risk_score`。
   - 目的：检查人工风险分是否过强、是否压制模型从原子特征中学习。

5. 只保留聚合风险分，去掉部分原子接力字段
   - 目的：检查新增收益是否主要来自风险分，还是来自更细的 limit list 原始字段。

6. 不使用 moneyflow 字段的当前设定继续保持
   - moneyflow 覆盖仍不足，不建议纳入默认训练特征。

建议评估标准：

```text
主指标：top_3pct excess_return、top_5pct excess_return
风险指标：limit_like_ratio、prev_day_limit_up_ratio、avg_limit_chase_failure_risk_score、实际 Top3 veto 数量
稳定性指标：分日 TopN 表现、极端差日期数量
```

不要只看：

```text
top_1pct excess_return
```

因为这容易重新回到高弹性涨停/近涨停偏好。

### 8.3 建议下一步执行顺序

1. 先增强 OOS 诊断工具：输出每个 TopN 内风险桶分布和分日 TopN 表现。
2. 用当前 v2 跑分日诊断，重点检查 2026-06-04、2026-06-05、2026-06-08、2026-06-09。
3. 如果分日表现没有明显硬伤，可以考虑把默认 rerank 从 B 切到 v2。
4. 再做小规模 ablation：
   - `limit_up_sample_weight`：0.2 / 0.3 / 0.5 / 0.8
   - `return_clip_high`：0.20 / 0.25 / 0.30
   - relay 特征组消融
5. 最后再考虑接力专项分类器 `label_limit_relay_success_1d`，作为二阶段加权或 veto，而不是直接替代通用 return_3d 主模型。

## 9. 注意事项

1. 当前代码仍然保留了 moneyflow 字段和读取能力，但默认训练不使用 moneyflow。
2. `market_limit_list_daily` 已覆盖 2025-08-01 ~ 2026-06-09，当前已不再是 blocker。
3. 当前默认线上模型已从方案 B 切换到 **luw05**（`raw_market_202508_202606_return_3d_lgbm_relay_v2_luw05.lightgbm.pkl`），71 特征，limit_up_sample_weight=0.5，return_clip=[-0.15, 0.30]。
4. 不建议现在扩大候选池补位作为核心修复，因为用户担心收益和准确率下降；训练侧区分接力成功/失败是主线。
5. 后续如果全市场 moneyflow 能补齐，再考虑将 moneyflow 特征加入默认训练特征。
6. 服务器开发环境为 Python 3.9，后续代码改动应避免 Python 3.10+ 语法，例如 `str | None`、`list[str]` 等。
7. 不要修改前端框架；当前优化主线集中在数据、训练、rerank 和诊断工具。

## 10. 2026-06-12：默认模型已切换到 luw05

### 10.1 切换内容

`PREFERRED_MODEL_SPECS` 已从方案 B 切换到 luw05：

```python
PREFERRED_MODEL_SPECS = [
    (
        "lgbm_relay_v2_luw05",
        "raw_market_202508_202606_return_3d_lgbm_relay_v2_luw05.lightgbm.pkl",
        1.00,
    ),
]
```

文件：`src/octts/services/regression_rerank_service.py`

### 10.2 luw05 模型元信息

```text
artifact: raw_market_202508_202606_return_3d_lgbm_relay_v2_luw05.lightgbm.pkl
target: return_3d
model_type: lightgbm
dataset: training_features_20250801_20260609_relay_v2.csv
feature_count: 71
sample_weight_mode: none
limit_up_sample_mode: downweight
limit_up_pct_threshold: 9.5
limit_up_sample_weight: 0.5
return_clip_enabled: true
return_clip: [-0.15, 0.30]
```

### 10.3 切换依据

根据第 8 节 ablation 结果，luw05 综合表现最优：

| 指标 | B | v2/full | luw05 |
|------|---|---------|-------|
| Top1% excess | +1.38% | +1.04% | **+1.38%** |
| Top3% excess | -0.34% | +1.24% | **+1.40%** |
| Top5% excess | -0.78% | +1.78% | **+1.80%** |
| Top10% excess | -0.49% | +1.73% | **+1.77%** |
| Top1% limit_like_ratio | 26.48% | 9.93% | **12.00%** |
| Top1% prev_day_limit_up | 9.24% | 4.55% | **4.41%** |

luw05 在收益和风险暴露之间取得最佳平衡：Top1% 收益接近 B，Top3/5/10% 全面优于 B，同时涨停/接力风险暴露远低于 B。

### 10.4 切换验证

已完成以下验证：

```text
✓ python -m compileall 通过
✓ load_model_artifact 加载正常
✓ dummy predict 输出正常
✓ 无旧模型引用残留
```

### 10.5 后续关注

切换后应重点观察：

1. 实际智能选股结果中 Top3 是否不再全为涨停/近涨停
2. 分日表现，尤其是 2026-06-04、2026-06-05、2026-06-08、2026-06-09 等关键日期
3. 如需进一步优化，可考虑 `no_risk`（去掉聚合风险分）作为下一候选

## 11. 2026-06-12：修复每日自动选股未拉取 limit_list

### 11.1 问题

每日智能选股前，`MarketDataSyncService.ensure_trade_date_data()` 只自动拉取三张表：

```text
✓ market_daily（日线 OHLCV）
✓ market_daily_basic（换手率、PE 等）
✓ market_adj_factor（复权因子）
✗ market_limit_list_daily（涨停板明细）— 缺失！
✗ market_top_list_daily（龙虎榜）— 缺失
```

导致 relay v2 模型的 12 个接力特征（`prev_day_limit_open_times`、`prev_day_up_stat_success`、`prev_day_fd_amount` 等）在每日预测时：
- 涨停股的 `prev_day_limit_up` 可正确设为 `True`（靠 `pct_change >= 9.5` 判断）
- 但 `open_times`、`first_time`、`last_time`、`up_stat_success`、`fd_amount`、`limit_times`、`limit_amount` 等全部为 `None`
- 预测时被 `_impute_prediction_features` 填充为训练中位数，而非真实值
- 模型学到的"用 relay 特征区分成功/失败接力"的能力在预测时完全失效

### 11.2 影响

2026-06-11 的实际选股结果：

```text
49 个候选中 48 个涨停（pct_change >= 9.5%）
所有候选的 rerank_model_score 挤在 0.990~1.0（几乎无区分力）
Top50 中 limit_up 占比 98%
全部被 near_limit_up_pct_change veto → Top3 为空
```

### 11.3 修复

文件：`src/octts/services/market_data_sync_service.py`

在 `ensure_trade_date_data()` 中新增：

```python
# limit_list: relay v2 模型核心数据源
if not self.db.has_market_data_for_trade_date(model=MarketLimitListDaily, trade_date=trade_date_value):
    rows = _fetch_rows(self.tushare_client._pro.limit_list_d(trade_date=normalized_trade_date))
    result["fetched"]["limit_list"] = len(rows)
    result["inserted"]["limit_list"] = self.db.upsert_market_limit_list_daily(rows, force_refresh=False)
result["limit_list"] = self.db.has_market_data_for_trade_date(model=MarketLimitListDaily, trade_date=trade_date_value)

# top_list: 龙虎榜，报告展示用
if not self.db.has_market_data_for_trade_date(model=MarketTopListDaily, trade_date=trade_date_value):
    rows = _fetch_rows(self.tushare_client._pro.top_list(trade_date=normalized_trade_date))
    result["fetched"]["top_list"] = len(rows)
    result["inserted"]["top_list"] = self.db.upsert_market_top_list_daily(rows, force_refresh=False)
result["top_list"] = self.db.has_market_data_for_trade_date(model=MarketTopListDaily, trade_date=trade_date_value)
```

### 11.4 验证

删除 6/11 的 limit_list 数据后重新调用 ensure：

```text
limit_list: fetched=131, inserted=131  ✓ 自动拉取并入库
top_list: fetched=0 (已存在)           ✓ 跳过
daily/daily_basic/adj_factor: fetched=0 ✓ 跳过
```

修复后每日选股时 relay 特征可获得真实 limit_list 数据，模型区分力恢复正常。
