# 提速验收报告（P0/P1/P2）

- 生成时间：2026-03-30 09:22:11

## 1. HTTP SSE 指标

- 已跳过：--skip-http enabled

## 2. 审批查询热路径探针

- 非 RESUME 调用次数：`0` (轮数=`30`)
- RESUME 调用次数：`10` (轮数=`10`)
- 相比旧模型预期减少：`30` 次

## 3. Baseline 对比

- 未提供 baseline，未做首包下降比例判定。

## 4. 验收门槛

- 非 RESUME 不触发审批查询：`True`
- RESUME 行为保留：`True`
- thinking 有但正文长时间空白：`None`
- 首包 p95 降幅门槛：`None`
