# 回归报告（LangGraph 1.0.10 升级后）

- 生成时间：2026-03-10 16:47:15
- 说明：本报告由 `scripts/regression/run_full_regression.py` 自动生成。

## 1. 运行环境版本

- `langchain`: `1.2.10`
- `langgraph`: `1.0.10`
- `langchain-core`: `1.2.18`
- `langchain-openai`: `1.1.11`
- `langsmith`: `0.7.16`
- `openai`: `2.26.0`

## 2. 路由回归（100 条）

- Domain 命中率：`96.00%`
- Intent 命中率：`98.00%`
- 全链路命中率：`94.00%`
- 路由耗时：Domain p50=`0ms` / p95=`0ms`，Intent p50=`0ms` / p95=`0ms`

### 2.1 主要误差样本（Top 20）

- 输入：`晚上出门合适吗` | 期望 `WEB_SEARCH/CHAT`，实际 `GENERAL/CHAT`
- 输入：`天气也这么不好，老板还让我查询一下holter最近的数据` | 期望 `YUNYOU_DB/CHAT`，实际 `YUNYOU_DB/yunyou_agent`
- 输入：`今天郑州东站有什么活动，再帮我查holter最近5条` | 期望 `YUNYOU_DB/CHAT`，实际 `YUNYOU_DB/yunyou_agent`
- 输入：`先查郑州天气，再查本地库用户最近10条` | 期望 `LOCAL_DB/CHAT`，实际 `WEB_SEARCH/CHAT`
- 输入：`我先看看今天天气，然后帮我查云柚holter记录` | 期望 `YUNYOU_DB/CHAT`，实际 `WEB_SEARCH/CHAT`
- 输入：`查下附近活动并且查询holter最近使用用户` | 期望 `YUNYOU_DB/CHAT`，实际 `WEB_SEARCH/CHAT`

## 3. 结构化路由烟测（LLM fallback 开启）

- 通过：`10/10`

## 4. 日志时延统计（app.log）

- `domain_router_ms`: count=13, p50=0ms, p95=1ms, max=1ms
- `intent_router_ms`: count=13, p50=0ms, p95=0ms, max=0ms
- `chat_node_ms`: count=3, p50=158124ms, p95=234333ms, max=234333ms
- `planner_ms`: count=0, p50=0ms, p95=0ms, max=0ms
- `aggregator_ms`: count=0, p50=0ms, p95=0ms, max=0ms
- `worker_ms`: count=4, p50=530ms, p95=62295ms, max=62295ms

## 5. 三链路端到端离线回归

- 成功率：`100.00%` （3/3）
- 首包耗时：p50=`36ms` / p95=`167ms`
- 总耗时：p50=`37ms` / p95=`168ms`

- `weather_chain`: status=`ok`, first_stream=`167ms`, total=`168ms`, preview=`郑州今天天气偏冷且有霾，建议减少长时间户外活动。`
- `search_chain`: status=`ok`, first_stream=`36ms`, total=`37ms`, preview=`已为你整理附近可选活动：商场、书店、室内运动馆。`
- `yunyou_chain`: status=`ok`, first_stream=`33ms`, total=`33ms`, preview=`✅ 已直接查询 Holter 数据。
- 时间范围：`2016-03-13` ~ `2026-03-10`
- 排序：`id DESC`
- 返回：`2` 条（上限 `5`）
- 说明：未提供明确时间范围，已按默认最近 3650 天查询。

| 用户ID | 使用日期 | 报告状态 | Holter类型 |
| --- | --- | --- | --- |`

## 6. 结论与建议

- 路由层面：规则优先 + 结构化输出已可稳定运行，核心路径耗时很低。
- 性能瓶颈：从日志看，`chat_node` 仍存在高耗时长尾（秒级到分钟级），建议继续做模型超时/降级策略。
- 后续动作：建议接入真实在线 E2E 压测（同样脚本框架，替换离线模型为在线模型）。
