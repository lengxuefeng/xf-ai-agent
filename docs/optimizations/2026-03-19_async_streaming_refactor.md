# 异步流式响应全链路改造

- **日期**：2026-03-19
- **版本**：LangGraph 1.0 / FastAPI / Python 3.11+

---

## 背景与问题

| # | 问题 | 现象 |
|---|------|------|
| 1 | 路由层阻塞 | `def` 路由放入线程池，`StreamingResponse` 整体完成后才发送 |
| 2 | 事件循环堵塞 | 同步 `Generator` 无法在 asyncio 中真正并发 |
| 3 | 首次编译延迟 | 新会话 `create_supervisor_graph()` 耗时 300ms~2s |
| 4 | 重复拓扑样板 | `WeatherAgent`/`SearchAgent` 各自重复 60+ 行 `StateGraph` 代码 |
| 5 | 同步工具阻塞 | `ThreadPoolExecutor` 同步调用浪费线程资源 |
| 6 | 取消感知延迟 | 客户端断开只能在 `GeneratorExit` 时感知 |

---

## 阶段一：修复流式响应

**改动文件**

| 文件 | 改动摘要 |
|------|----------|
| `app/api/v1/chat_api.py` | 路由改为 `async def`，调用 `process_stream_chat_async` |
| `app/services/chat_service.py` | 新增 async 版生成器；阻塞 DB 操作通过 `asyncio.to_thread` 下沉 |
| `app/agent/graph_runner.py` | `stream_run` 改为 `AsyncGenerator`；用 `asyncio.Queue + loop.call_soon_threadsafe` 桥接后台线程；`_consume_event_queue_async` 用 `await asyncio.wait_for` 消费 |

**核心架构**

```
[FastAPI async route]
        │ await
        ▼
[chat_service async generator]  ← asyncio.to_thread (DB)
        │ async for
        ▼
[graph_runner.stream_run]  ← AsyncGenerator
        ├─ asyncio.Queue  ←── call_soon_threadsafe ──┐
        │  await queue.get()              [_graph_worker 后台线程]
        ▼                                 graph.stream() 同步执行
   SSE chunks → 前端
```

**效果**：首包延迟从"整体完成后推送"降为"每个 chunk 立即推送"。

---

## 阶段二：Session 预热池（SessionPool）

**改动文件**

| 文件 | 改动摘要 |
|------|----------|
| `app/config/runtime_settings.py` | 新增 `SessionPoolConfig` + `SESSION_POOL_CONFIG` |
| `app/services/session_pool.py` | 新建 `SessionPool`：LIFO 借取 + 过期淘汰 + 后台 refill |
| `app/agent/graph_runner.py` | `_get_or_create_supervisor`：本地缓存 → 池借取 → 按需编译 |
| `app/main.py` | startup 异步启动池；shutdown 停止池 |

**配置参数**

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `SESSION_POOL_ENABLED` | `true` | 是否启用 |
| `SESSION_POOL_SIZE` | `4` | 预热实例数 |
| `SESSION_POOL_MAX_IDLE_SECONDS` | `300` | 最大空闲时间 |
| `SESSION_POOL_REFILL_INTERVAL_SECONDS` | `60` | 补充间隔 |
| `SESSION_POOL_BORROW_TIMEOUT_SECONDS` | `0.1` | 借取超时 |

**效果**：新会话首包延迟从 300ms~2s 降低至 <10ms（命中池时）。

---

## 阶段三：简化 Agent 图拓扑

**改动文件**

| 文件 | 改动摘要 |
|------|----------|
| `app/agent/base.py` | 新增 `_build_react_graph()` 通用 ReAct 工厂 |
| `app/agent/agents/weather_agent.py` | 改用工厂，删除 ~60 行重复样板 |
| `app/agent/agents/search_agent.py` | 改用工厂，删除 ~20 行重复样板 |

**核心**：子类只提供 `model_node_fn` + `tools` + `max_tool_loops`，拓扑由基类统一管理。

`SqlAgent`/`CodeAgent`/`MedicalAgent` 为线性或含审批图，保持原样。

---

## 阶段四：原生异步工具

**改动文件**

| 文件 | 改动摘要 |
|------|----------|
| `app/agent/tools/search_tools.py` | `tavily_search_tool` 改 `async def`；`asyncio.wait_for + to_thread` 替代 `ThreadPoolExecutor` |
| `app/agent/tools/weather_tools.py` | `get_weathers` 改 `async def`；`asyncio.gather + to_thread` 替代 `ThreadPoolExecutor.map` |
| `app/agent/graph_runner.py` | `_graph_worker` 启动时 `asyncio.new_event_loop()`，退出时 `loop.close()` |

**效果**：工具并发不再占用额外线程；代码量缩减约 40%。

---

## 阶段五：请求取消机制优化

**改动文件**

| 文件 | 改动摘要 |
|------|----------|
| `app/services/request_cancellation_service.py` | 双通道：`threading.Event` + `asyncio.Event`；新增 `cancel_on_disconnect` 异步上下文管理器；TTL stale 清理 |
| `app/api/v1/chat_api.py` | 包装 `body_iterator`，每次 `yield` 前检查 `request.is_disconnected()` |

**取消传播链路**

```
客户端断开
    │
    ▼
_disconnect_aware_generator (chat_api)
    │ 停止 yield → GeneratorExit
    ▼
_consume_event_queue_async (graph_runner)
    │ cancel_request(run_id)
    ▼
graph_worker 检测 is_cancelled() → break
    │
    ▼
后台线程退出，资源释放
```

**效果**：客户端断开后资源释放从最长 `idle_timeout_sec`（45s）缩短至 ~0.5s。

---

## 总体收益

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 首包延迟（新会话） | 300ms~2s | <10ms（命中池）|
| SSE 推送方式 | 整体完成后推送 | 逐块实时推送 |
| 并发 SSE 请求 | 线程池阻塞 | asyncio 原生并发 |
| 工具调用线程 | `ThreadPoolExecutor` 独占 | `asyncio.to_thread` 弹性 |
| 客户端断开感知 | 最长 45s | ~0.5s |
| Agent 拓扑代码 | 各自重复 60+ 行 | 基类统一 10 行调用 |
