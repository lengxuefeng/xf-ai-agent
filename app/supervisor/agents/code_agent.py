"""代码执行 Agent。

本模块在"生成代码"和"执行代码"之间做语义拆分：
1. 用户只要求写代码时，直接返回代码正文；
2. 用户明确要求运行/测试时，才进入审批与执行链路。

核心职责：
- 根据用户需求生成代码（支持多种语言）
- 识别用户是否要求执行代码
- 执行 Python 代码（受控环境）
- 代码执行前的安全审批机制
- 格式化代码和执行结果

业务场景示例：
1. 用户问"写一个快速排序算法" → 生成代码，不执行，返回代码示例
2. 用户问"运行这段 Python 代码" → 生成代码，进入审批链路，执行并返回结果
3. 用户问"用 Java 写一个 Hello World" → 生成 Java 代码，不支持执行，返回代码示例
4. 用户问"帮我测试这段代码" → 识别为执行意图，进入审批链路

设计要点：
- 语义拆分：区分"写代码"和"执行代码"两种意图
- 审批机制：所有代码执行前必须用户确认
- 语言支持：优先支持 Python 执行，其他语言仅生成代码
- 代码清洗：去除 Markdown 围栏，提取纯代码
- 结果格式化：友好的代码展示和结果反馈

与其他 Agent 的区别：
- weather_agent：使用天气 API
- search_agent：使用搜索引擎
- sql_agent：使用数据库查询
- code_agent：生成和执行代码

执行能力：
- Python：支持生成 + 执行（受控环境）
- 其他语言（Java、JavaScript、TypeScript 等）：仅支持生成

输出结果：
- 代码生成：Markdown 格式的代码块
- 代码执行：执行结果（输出、错误等）
- 审批流程：执行前确认
"""

import re
from typing import Any, TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig

from supervisor.base import BaseAgent
from supervisor.graph_state import AgentRequest
from config.constants.approval_constants import (
    ApprovalDecision,
    CODE_APPROVAL_ACTION_NAME,
    CODE_APPROVAL_MESSAGE,
    DEFAULT_ALLOWED_DECISIONS,
)
from config.constants.code_agent_keywords import CODE_EXECUTE_HINTS, CODE_GENERATE_ONLY_HINTS
from common.utils.code_tools import execute_python_code
from common.utils.custom_logger import get_logger
from prompts.agent_prompts.code_prompt import CodePrompt

log = get_logger(__name__)


def _message_text(message: BaseMessage | None) -> str:
    """
    将消息对象统一转为文本，便于执行意图判断

    设计目的：
    - 统一处理不同类型的消息内容（str、list、dict）
    - 提取纯文本，便于后续处理
    - 避免复杂结构导致的解析问题

    处理逻辑：
    1. 如果消息为 None，返回空字符串
    2. 如果 content 是字符串，直接返回
    3. 如果 content 是列表，提取所有文本部分
    4. 如果 content 是其他类型，转为字符串

    参数说明：
    - message: 消息对象

    返回值：
    - str: 提取的文本内容

    示例：
    >>> msg = AIMessage(content="这是一条消息")
    >>> _message_text(msg)
    "这是一条消息"
    >>> msg = AIMessage(content=[{"type": "text", "text": "这是一条消息"}])
    >>> _message_text(msg)
    "这是一条消息"
    """
    if message is None:
        return ""

    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts).strip()

    return str(content or "").strip()


def _normalize_model_content(content: Any) -> str:
    """
    把模型 content 统一规整为纯文本，避免 list/dict 直接串到前端

    设计目的：
    - LLM 返回的 content 可能是复杂结构（list、dict）
    - 需要提取其中的文本内容
    - 避免复杂结构传给前端导致显示问题

    处理逻辑：
    1. 如果是字符串，直接返回
    2. 如果是列表，提取所有文本项
    3. 如果是字典，提取 text 或 content 字段
    4. 其他情况转为字符串

    参数说明：
    - content: 模型返回的 content

    返回值：
    - str: 规整后的文本

    示例：
    >>> _normalize_model_content(["这是一段文本", {"text": "这是另一段文本"}])
    "这是一段文本\\n这是另一段文本"
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip() if parts else str(content)

    if isinstance(content, dict):
        text = content.get("text") or content.get("content") or ""
        if isinstance(text, str):
            return text.strip()

    return str(content or "").strip()


def _strip_markdown_fences(code: str) -> str:
    """
    剥离模型偶尔返回的 Markdown 围栏，只保留代码正文

    设计目的：
    - LLM 可能返回 ```python``` 这样的围栏格式
    - 执行代码时需要剥离围栏，只保留纯代码
    - 避免围栏影响代码执行

    围栏格式：
    - ```python
    - ```javascript
    - ``` 等等

    参数说明：
    - code: 可能包含 Markdown 围栏的代码

    返回值：
    - str: 剥离围栏后的纯代码

    示例：
    >>> _strip_markdown_fences("```python\\nprint('hello')\\n```")
    "print('hello')"
    >>> _strip_markdown_fences("print('hello')")
    "print('hello')"
    """
    raw = str(code or "").strip()
    if not raw:
        return ""

    # 匹配 Markdown 围栏格式：```language\n...```
    fenced_match = re.fullmatch(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", raw, flags=re.S)
    if fenced_match:
        return fenced_match.group(1).strip()

    # 如果不匹配完整围栏，去除所有 ``` 符号
    return raw.replace("```", "").strip()


def _detect_requested_language(text: str) -> str:
    """
    根据用户输入推断目标语言，默认返回 python

    设计目的：
    - 检测用户要求的编程语言
    - 用于生成代码时的语言标记
    - 判断是否支持自动执行（目前只支持 Python）

    检测逻辑：
    - 遍历语言别名列表，匹配文本中的关键词
    - "go" 使用单词边界匹配，避免误判（如 "good" 不匹配）
    - 其他语言使用子串匹配

    支持的语言：
    - Java: java, jdk, javac
    - Python: python, py, python3
    - JavaScript: javascript, js, node
    - TypeScript: typescript, ts
    - Go: golang, go (单词边界)
    - Rust: rust
    - C++: c++, cpp
    - C#: c#, csharp, .net
    - PHP: php
    - Ruby: ruby

    参数说明：
    - text: 用户输入文本

    返回值：
    - str: 检测到的语言名称，默认 "python"

    示例：
    >>> _detect_requested_language("用 Python 写一个排序算法")
    "python"
    >>> _detect_requested_language("写一个 Java 程序")
    "java"
    >>> _detect_requested_language("用 Go 实现这个功能")
    "go"
    """
    normalized = (text or "").strip().lower()
    if not normalized:
        return "python"

    # 语言别名列表：(语言名, 关键词元组)
    language_aliases = (
        ("java", ("java", "jdk", "javac")),
        ("python", ("python", "py", "python3")),
        ("javascript", ("javascript", "js", "node")),
        ("typescript", ("typescript", "ts")),
        ("go", ("golang", "go语言", " go ")),  # " go " 用于单词边界匹配
        ("rust", ("rust",)),
        ("cpp", ("c++", "cpp")),
        ("csharp", ("c#", "csharp", ".net")),
        ("php", ("php",)),
        ("ruby", ("ruby",)),
    )

    for language, aliases in language_aliases:
        for alias in aliases:
            # 特殊处理 "go"：使用单词边界匹配，避免误判 "good" 等
            if alias == " go ":
                if re.search(r"(?<![a-z])go(?![a-z])", normalized):
                    return language
                continue
            # 其他语言使用子串匹配
            if alias in normalized:
                return language

    return "python"


def _language_display_name(language: str) -> str:
    """
    获取语言的显示名称

    设计目的：
    - 将语言代码转为可读的显示名称
    - 用于代码示例的友好展示
    - 未知语言返回原名称或"代码"

    参数说明：
    - language: 语言代码（如 "python"、"java"）

    返回值：
    - str: 语言的显示名称

    示例：
    >>> _language_display_name("python")
    "Python"
    >>> _language_display_name("java")
    "Java"
    >>> _language_display_name("unknown")
    "代码"
    """
    mapping = {
        "java": "Java",
        "python": "Python",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "go": "Go",
        "rust": "Rust",
        "cpp": "C++",
        "csharp": "C#",
        "php": "PHP",
        "ruby": "Ruby",
    }
    return mapping.get(language, language or "代码")


def _format_generated_code_reply(
    code: str,
    *,
    language: str,
    execution_requested: bool,
    execution_supported: bool,
) -> str:
    """
    生成对用户更友好的代码回复

    设计目的：
    - 将生成的代码格式化为友好的 Markdown 格式
    - 根据执行意图和支持情况添加不同的前言
    - 统一代码展示风格

    前言逻辑：
    1. 用户要求执行且支持执行：标准前言
    2. 用户要求执行但不支持执行（非 Python）：提示仅支持 Python
    3. 用户未要求执行：标准前言

    参数说明：
    - code: 生成的代码
    - language: 代码语言
    - execution_requested: 用户是否要求执行
    - execution_supported: 是否支持执行（目前只有 Python 支持）

    返回值：
    - str: 格式化后的代码回复

    示例：
    >>> _format_generated_code_reply(
    ...     "print('hello')",
    ...     language="python",
    ...     execution_requested=True,
    ...     execution_supported=True
    ... )
    "按你的要求，我先给你整理了一个 Python 示例：\\n\\n```python\\nprint('hello')\\n```"
    """
    cleaned_code = _strip_markdown_fences(code)
    display_name = _language_display_name(language)
    preface = f"按你的要求，我先给你整理了一个 {display_name} 示例："

    # 用户要求执行但不支持执行（非 Python）
    if execution_requested and not execution_supported:
        preface = (
            f"按你的要求，我先给你整理了一个 {display_name} 示例。"
            f"当前内置自动执行链路主要支持 Python，所以这段 {display_name} 代码我先不直接运行："
        )
    elif not execution_requested:
        preface = f"按你的要求，我先给你整理了一个 {display_name} 示例："

    return f"{preface}\n\n```{language}\n{cleaned_code}\n```"


def _latest_human_text(messages: List[BaseMessage]) -> str:
    """
    提取最近一条用户消息文本

    设计目的：
    - 从消息列表中提取最新的用户输入
    - 用于判断执行意图和语言要求
    - 支持多轮对话场景

    实现逻辑：
    - 从消息列表末尾反向遍历
    - 找到第一个类型为 "human" 的消息
    - 提取其文本内容

    参数说明：
    - messages: 消息列表

    返回值：
    - str: 最新用户消息文本，如果没有返回空字符串

    示例：
    >>> messages = [
    ...     AIMessage(content="好的"),
    ...     HumanMessage(content="用 Python 写一个排序算法")
    ... ]
    >>> _latest_human_text(messages)
    "用 Python 写一个排序算法"
    """
    for message in reversed(messages or []):
        if getattr(message, "type", "") == "human":
            return _message_text(message)
    return ""


def _should_execute_request(latest_human_text: str) -> bool:
    """
    判断当前请求是否明确要求执行代码

    设计目的：
    - 区分"写代码"和"执行代码"两种意图
    - 只有明确要求执行时才进入审批和执行链路
    - 提升用户体验，避免不必要的执行流程

    检测逻辑：
    1. 如果文本包含执行关键词（"运行"、"执行"、"测试"等），返回 True
    2. 如果文本包含仅生成关键词（"写"、"生成"等），返回 False
    3. 其他情况返回 False（保守策略）

    执行关键词：运行、执行、测试、跑一下、运行代码、测试代码等
    生成关键词：写、生成、举例、示例、示例代码等

    参数说明：
    - latest_human_text: 最新用户消息文本

    返回值：
    - bool: True 表示要求执行，False 表示仅生成代码

    示例：
    >>> _should_execute_request("运行这段代码")
    True
    >>> _should_execute_request("写一个排序算法")
    False
    >>> _should_execute_request("帮我测试这个函数")
    True
    """
    text = str(latest_human_text or "").strip().lower()
    if not text:
        return False

    # 检测执行关键词
    if any(hint in text for hint in CODE_EXECUTE_HINTS):
        return True

    # 检测仅生成关键词
    if any(hint in text for hint in CODE_GENERATE_ONLY_HINTS):
        return False

    return False


class CodeAgentState(TypedDict):
    """代码编写子图状态

    状态字段说明：
    - messages: 消息列表，包括用户消息、AI 消息
    - generated_code: 当前轮生成的代码正文
    - should_execute: 是否应该进入执行链路
    - execution_requested: 用户是否明确要求执行
    - execution_supported: 当前语言是否支持自动执行
    - requested_language: 用户要求的代码语言
    - code_to_execute: 待执行的代码
    - execution_result: 执行结果

    使用场景：
    - 在子图执行过程中维护状态
    - 跨节点传递代码和执行信息
    - 追踪执行意图和执行结果
    """
    messages: Annotated[List[BaseMessage], add_messages]
    generated_code: Optional[str]  # 当前轮生成的代码正文
    should_execute: Optional[bool]  # 是否应该进入执行链路
    execution_requested: Optional[bool]  # 用户是否明确要求执行
    execution_supported: Optional[bool]  # 当前语言是否支持自动执行
    requested_language: Optional[str]  # 用户要求的代码语言
    code_to_execute: Optional[str]  # 待执行的代码
    execution_result: Optional[str]  # 执行结果


class CodeAgent(BaseAgent):
    """代码执行 Agent：生成并执行 Python 代码

    主要功能：
    - 根据用户需求生成代码（支持多种语言）
    - 识别用户是否要求执行代码
    - 执行 Python 代码（受控环境）
    - 代码执行前的安全审批
    - 格式化代码和执行结果

    工作流程：
    1. 接收用户请求
    2. 判断执行意图和语言要求
    3. 生成代码（LLM）
    4. 如果要求执行且支持执行，进入审批链路
    5. 审批通过后执行代码
    6. 格式化结果并返回

    典型使用场景：
    - "写一个快速排序算法" → 生成代码
    - "运行这段 Python 代码" → 生成代码 + 审批 + 执行
    - "用 Java 写一个 Hello World" → 生成代码（不支持执行）
    - "帮我测试这个函数" → 生成代码 + 审批 + 执行

    执行能力：
    - Python：支持生成 + 执行
    - 其他语言：仅支持生成

    审批机制：
    - 所有 Python 代码执行前必须审批
    - 使用 LangGraph interrupt 实现挂起
    - 支持批准、拒绝等决策
    """

    def __init__(self, req: AgentRequest):
        """
        初始化代码执行 Agent

        参数说明：
        - req: Agent 请求对象，包含模型配置、会话信息等

        初始化步骤：
        1. 调用父类 BaseAgent 初始化
        2. 验证模型配置，确保代码模型可用
        3. 创建 LLM 实例
        4. 构建生成代码的提示词模板
        5. 构建代码子图

        异常处理：
        - 如果模型未配置，抛出 ValueError 提示检查配置

        示例：
        >>> req = AgentRequest(model=llm, session_id="xxx")
        >>> code_agent = CodeAgent(req)
        >>> result = code_agent.invoke({"messages": [HumanMessage("写一个排序算法")]})
        """
        super().__init__(req)
        if not req.model:
            raise ValueError("Code Agent 模型初始化失败，请检查配置。")
        self.llm = req.model
        self.subgraph_id = "code_agent"

        # 提示词：生成代码的系统提示词
        # SYSTEM 包含代码生成规则、最佳实践等
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CodePrompt.SYSTEM),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Runnable:
        """
        构建代码生成/执行双模式图

        子图结构：
        1. generate_code：生成代码（LLM）
        2. execute_code：执行代码（含审批中断）
        3. analyze_result：分析结果并格式化

        流程：
        - START → generate_code
        - generate_code → (条件判断) → execute_code 或 analyze_result
        - execute_code → analyze_result
        - analyze_result → END

        条件判断逻辑：
        - 如果 should_execute=True → 进入 execute_code
        - 如果 should_execute=False → 直接进入 analyze_result

        返回值：
        - StateGraph: 编译后的可执行子图，使用全局 checkpointer
        """
        workflow = StateGraph(CodeAgentState)

        # 添加节点
        workflow.add_node("generate_code", self._generate_code_node, retry_policy=self.RETRY_POLICY)
        workflow.add_node("execute_code", self._execute_code_node, retry_policy=self.RETRY_POLICY)
        workflow.add_node("analyze_result", self._analyze_result_node, retry_policy=self.RETRY_POLICY)

        # 定义边
        workflow.add_edge(START, "generate_code")
        workflow.add_conditional_edges("generate_code", self._route_after_generation)
        workflow.add_edge("execute_code", "analyze_result")
        workflow.add_edge("analyze_result", END)

        # 编译图 (使用全局 checkpointer，支持中断恢复)
        return workflow.compile(checkpointer=self.checkpointer)

    async def _generate_code_node(self, state: CodeAgentState, config: RunnableConfig):
        """
        生成代码节点

        功能说明：
        - 使用 LLM 生成代码
        - 检测用户是否要求执行代码
        - 检测代码语言
        - 判断是否支持自动执行

        处理流程：
        1. 调用 LLM 生成代码
        2. 提取最新用户消息文本
        3. 检测用户要求的代码语言
        4. 清洗生成的代码（去除 Markdown 围栏）
        5. 判断用户是否要求执行
        6. 判断是否支持执行（只有 Python 支持）
        7. 更新状态

        参数说明：
        - state: 子图状态
        - config: 运行配置（可能包含 workspace_root）

        返回值：
        - dict: 更新后的状态，包含生成的代码和执行相关信息
        """
        chain = self.prompt | self.llm
        response = await chain.ainvoke({"messages": state["messages"]}, config=config)

        # 提取最新用户消息，用于检测语言和执行意图
        latest_human_text = _latest_human_text(state["messages"])

        # 检测用户要求的代码语言
        requested_language = _detect_requested_language(latest_human_text)

        # 清洗生成的代码（去除 Markdown 围栏）
        code = _strip_markdown_fences(_normalize_model_content(getattr(response, "content", response)))

        # 判断是否要求执行
        execution_requested = _should_execute_request(latest_human_text)

        # 判断是否支持执行（目前只支持 Python）
        execution_supported = requested_language in {"python"}

        # 决定是否进入执行链路
        should_execute = execution_requested and execution_supported

        return {
            "generated_code": code,
            "code_to_execute": code,
            "should_execute": should_execute,
            "execution_requested": execution_requested,
            "execution_supported": execution_supported,
            "requested_language": requested_language,
        }

    def _route_after_generation(self, state: CodeAgentState) -> str:
        """
        根据用户意图决定是否进入执行环节

        功能说明：
        - 条件路由函数
        - 根据 should_execute 决定下一个节点

        路由规则：
        - should_execute=True → execute_code
        - should_execute=False → analyze_result

        参数说明：
        - state: 子图状态

        返回值：
        - str: 下一个节点名称

        示例：
        >>> state = {"should_execute": True}
        >>> CodeAgent()._route_after_generation(state)
        "execute_code"
        """
        return "execute_code" if state.get("should_execute") else "analyze_result"

    def _execute_code_node(self, state: CodeAgentState, config: RunnableConfig):
        """
        执行代码节点（含审批中断）

        功能说明：
        - 使用 LangGraph interrupt 实现审批中断
        - 挂起子图执行，等待用户审批决策
        - 根据审批决策执行或拒绝代码

        审批流程：
        1. 构建审批请求数据（包含代码和工作目录）
        2. 调用 interrupt 挂起执行
        3. 等待用户审批（在 Resume 时恢复）
        4. 根据审批决策执行或拒绝

        决策类型：
        - APPROVE: 批准执行代码
        - REJECT: 拒绝执行代码
        - MODIFY: 修改代码后执行（预留）

        参数说明：
        - state: 子图状态
        - config: 运行配置（包含 workspace_root）

        返回值：
        - dict: 更新后的状态，包含执行结果

        异常处理：
        - 代码执行错误时，返回错误消息

        工作目录说明：
        - 从 config 或 state 获取 workspace_root
        - 代码在工作目录内执行（受控环境）
        """
        code = str(state.get("code_to_execute") or "").strip()
        if not code:
            return {"execution_result": "未生成可执行代码。"}

        # 获取工作目录（受控执行环境）
        workspace_root = (
            config.get("configurable", {}).get("workspace_root")
            or self.req.state.get("workspace_root")
            or None
        )

        # --- Native Interrupt Logic ---
        # 构建审批请求数据
        decision = interrupt({
            "action_requests": [{
                "type": "code_approval",
                "name": CODE_APPROVAL_ACTION_NAME,
                "args": {"code": code, "workspace_root": workspace_root},
                "description": f"即将执行 Python 代码:\n{code[:100]}...",  # 截断显示
            }],
            "message": CODE_APPROVAL_MESSAGE,
            "allowed_decisions": list(DEFAULT_ALLOWED_DECISIONS),
        })

        # 处理审批决策
        action = decision.get("action") if isinstance(decision, dict) else decision
        if action == ApprovalDecision.REJECT.value:
            return {"execution_result": "用户拒绝执行代码。"}

        # 执行代码
        try:
            result = execute_python_code(
                code,
                cwd=workspace_root,
                workspace_root=workspace_root,
            )
            return {"execution_result": str(result)}
        except Exception as e:
            return {"execution_result": f"代码执行错误: {str(e)}"}

    def _analyze_result_node(self, state: CodeAgentState, config: RunnableConfig):
        """
        分析结果节点

        功能说明：
        - 如果未执行代码，格式化代码示例并返回
        - 如果已执行代码，包装执行结果并返回
        - 生成最终的 AI 消息

        处理逻辑：
        1. 检查 should_execute
        2. 如果 False（未执行）：格式化代码示例，添加友好前言
        3. 如果 True（已执行）：简单包装执行结果
        4. 返回 AIMessage

        参数说明：
        - state: 子图状态

        返回值：
        - dict: 更新后的状态，包含格式化后的消息

        示例：
        # 未执行代码
        >>> state = {
        ...     "should_execute": False,
        ...     "generated_code": "print('hello')",
        ...     "requested_language": "python",
        ...     "execution_requested": False,
        ...     "execution_supported": True
        ... }
        >>> code_agent._analyze_result_node(state)
        {"messages": [AIMessage(content="按你的要求，我先给你整理了一个 Python 示例：\n\n```python\nprint('hello')\n```")]}
        """
        if not state.get("should_execute"):
            # 未执行代码：格式化代码示例
            code = str(state.get("generated_code") or state.get("code_to_execute") or "").strip()
            requested_language = str(state.get("requested_language") or "python").strip() or "python"
            return {
                "messages": [
                    AIMessage(
                        content=_format_generated_code_reply(
                            code,
                            language=requested_language,
                            execution_requested=bool(state.get("execution_requested")),
                            execution_supported=bool(state.get("execution_supported")),
                        )
                    )
                ]
            }

        # 已执行代码：简单包装结果
        result = str(state.get("execution_result") or "代码执行已结束，但未返回结果。")
        return {"messages": [AIMessage(content=CodePrompt.EXECUTION_PREFIX.format(result))]}
