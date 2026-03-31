"""
【模块说明】
定义系统内部流转的强类型请求对象，以及 Supervisor 并发执行态的轻量声明。

核心职责：
- 定义 AgentRequest：单智能体请求载荷封装
- 定义 BatchAgentRequest：批量处理请求载荷
- 保证从 API 层传递到 Agent 层的参数不会出现类型错误
- 支持类型检查和序列化/反序列化

设计理念：
- 强类型约束：使用 Pydantic 模型进行类型验证
- 类型安全：确保数据结构的一致性和可靠性
- 灵活扩展：支持可选字段和自定义配置
- 数据流转：统一 API 层到 Agent 层的数据格式

使用场景：
1. API 层接收用户请求后，构造 AgentRequest 传递给 Agent 层
2. Supervisor 调度各个子 Agent 时，构造对应的 AgentRequest
3. 批量处理场景下，使用 BatchAgentRequest 封装多个 AgentRequest
4. Agent 执行时，从 AgentRequest 中提取参数和上下文

相关模块：
- agent.base.BaseAgent：使用 AgentRequest 进行初始化
- api/chat.py：构造 AgentRequest 并调用 Agent
- agent.graphs.supervisor：调度 Agent 时使用 AgentRequest
"""

from typing import List, Optional, Dict, Any

from langchain_core.language_models import BaseChatModel
from pydantic import Field

from models.schemas.base import ArbitraryTypesBaseSchema


class AgentRequest(ArbitraryTypesBaseSchema):
    """
    单智能体请求载荷封装

    设计目的：
    - 封装 Agent 执行所需的所有参数
    - 在 API 层和 Agent 层之间传递数据
    - 支持类型检查和验证
    - 保持数据结构的一致性

    字段说明：
    - user_input: 用户当前轮次的输入文本
      - 来源：前端 API 请求、用户输入
      - 用途：传递给 LLM 进行推理
      - 示例："查询北京的天气"

    - state: 外部传入的初始状态（可选）
      - 来源：历史会话状态、全局上下文
      - 用途：传递历史消息、会话摘要等上下文信息
      - 结构：{"messages": [...], "context_summary": "..."}
      - 示例：{"messages": [HumanMessage("查询天气")], "context_summary": "用户在北京"}

    - session_id: 会话 ID
      - 来源：前端 API 请求
      - 用途：绑定 Checkpointer 的线程 ID，用于状态持久化
      - 格式：字符串
      - 示例："session_123456"

    - subgraph_id: 子图命名空间标识（可选）
      - 来源：默认使用 Agent 类名，也可指定
      - 用途：标识不同的子图，用于状态隔离
      - 格式：字符串
      - 示例："search_agent"、"yunyou_agent"

    - model: 已初始化好的 LLM 实例
      - 来源：全局模型配置
      - 用途：Agent 调用的语言模型
      - 类型：BaseChatModel
      - 示例：GLM-4、GPT-4 等

    - llm_config: 其他模型参数配置（可选）
      - 来源：全局模型配置或用户指定
      - 用途：传递模型相关的高级配置
      - 结构：字典
      - 示例：{"temperature": 0.7, "max_tokens": 2000}

    使用示例：
    >>> request = AgentRequest(
    ...     user_input="查询北京的天气",
    ...     session_id="session_123",
    ...     model=llm,
    ...     state={"messages": [HumanMessage("用户在北京")]}
    ... )
    >>> agent = SearchAgent(request)
    >>> for event in agent.run(request):
    ...     print(event)

    序列化示例：
    >>> request_dict = request.model_dump()
    >>> print(request_dict)
    {"user_input": "查询北京的天气", "session_id": "session_123", ...}

    注意事项：
    - session_id 必须唯一，用于状态持久化和会话管理
    - model 必须已初始化，不能传递模型名称
    - state 是可选的，但在多轮对话中通常需要传递历史消息
    - subgraph_id 不指定时，会使用 Agent 类名作为默认值
    """
    user_input: str  # 用户当前轮次的输入文本
    state: Optional[Dict[str, Any]] = None  # 外部传入的初始状态（按需使用）
    session_id: str  # 用于绑定 Checkpointer 的线程 ID
    subgraph_id: str  # 子图命名空间标识
    model: Any  # 已初始化模型对象，支持 BaseChatModel 或 with_fallbacks 包装后的 Runnable
    llm_config: Optional[Dict[str, Any]] = None  # 其他模型参数配置


class SupervisorExecutionState(ArbitraryTypesBaseSchema):
    """
    Supervisor 并发编排阶段的最小执行态。

    说明：
    - `plan` 保存 Planner 拆出的原子任务列表；
    - `current_task` 表示单个 Worker 当前正在处理的任务。
    """

    plan: List[str] = Field(default_factory=list)
    current_task: str = ""


class BatchAgentRequest(ArbitraryTypesBaseSchema):
    """
    批量处理请求载荷（用于并发场景）

    设计目的：
    - 封装多个 AgentRequest，支持批量处理
    - 控制并发线程数，避免资源耗尽
    - 提升批量处理的效率和性能
    - 统一批量处理的参数配置

    使用场景：
    1. 批量查询：同时查询多个不同的问题
    2. 并发执行：同时执行多个不同的 Agent
    3. 批量测试：批量测试 Agent 的功能和性能
    4. 数据处理：批量处理多个数据项

    字段说明：
    - inputs: AgentRequest 列表
      - 类型：List[AgentRequest]
      - 用途：包含多个 Agent 请求
      - 示例：[request1, request2, request3]

    - max_threads: 最大并发线程数
      - 类型：int
      - 默认值：2
      - 用途：控制并发度，避免资源耗尽
      - 示例：4（最多 4 个线程并发执行）

    - model: 共享的 LLM 实例
      - 类型：BaseChatModel
      - 用途：所有请求共享同一个模型实例
      - 示例：GLM-4

    设计考虑：
    - 共享模型：所有请求共享同一个模型实例，节省资源
    - 并发控制：通过 max_threads 控制并发度，避免资源耗尽
    - 顺序保证：批量处理的顺序不一定保证，需要根据业务需求处理

    使用示例：
    >>> requests = [
    ...     AgentRequest(user_input="查询天气", session_id="s1", model=llm),
    ...     AgentRequest(user_input="搜索新闻", session_id="s2", model=llm),
    ...     AgentRequest(user_input="执行代码", session_id="s3", model=llm)
    ... ]
    >>> batch_request = BatchAgentRequest(
    ...     inputs=requests,
    ...     max_threads=3,
    ...     model=llm
    ... )
    >>> # 批量执行逻辑（在具体实现中）
    >>> # ...

    注意事项：
    - max_threads 应根据系统资源合理设置，避免资源耗尽
    - 共享模型实例时，需要注意线程安全和性能问题
    - 批量处理的顺序不一定保证，需要根据业务需求处理
    - 错误处理需要考虑单个请求失败的情况

    性能优化：
    - 合理设置 max_threads，平衡并发度和资源消耗
    - 使用连接池复用模型资源
    - 批量处理时可以考虑异步执行提升性能
    """
    inputs: List[AgentRequest]
    max_threads: int = 2
    model: Any
