"""代码执行 Agent。

本模块在“生成代码”和“执行代码”之间做语义拆分：
1. 用户只要求写代码时，直接返回代码正文；
2. 用户明确要求运行/测试时，才进入审批与执行链路。
"""
import re
from typing import Any, TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

from agent.base import BaseAgent
from agent.graph_state import AgentRequest
from constants.approval_constants import (
    ApprovalDecision,
    CODE_APPROVAL_ACTION_NAME,
    CODE_APPROVAL_MESSAGE,
    DEFAULT_ALLOWED_DECISIONS,
)
from constants.code_agent_keywords import CODE_EXECUTE_HINTS, CODE_GENERATE_ONLY_HINTS
from utils.code_tools import execute_python_code
from utils.custom_logger import get_logger
from agent.prompts.code_prompt import CodePrompt

log = get_logger(__name__)


def _message_text(message: BaseMessage | None) -> str:
    """将消息对象统一转为文本，便于执行意图判断。"""
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
    """把模型 content 统一规整为纯文本，避免 list/dict 直接串到前端。"""
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
    """剥离模型偶尔返回的 Markdown 围栏，只保留代码正文。"""
    raw = str(code or "").strip()
    if not raw:
        return ""

    fenced_match = re.fullmatch(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", raw, flags=re.S)
    if fenced_match:
        return fenced_match.group(1).strip()

    return raw.replace("```", "").strip()


def _detect_requested_language(text: str) -> str:
    """根据用户输入推断目标语言，默认返回 python/text 风格围栏。"""
    normalized = (text or "").strip().lower()
    if not normalized:
        return "python"

    language_aliases = (
        ("java", ("java", "jdk", "javac")),
        ("python", ("python", "py", "python3")),
        ("javascript", ("javascript", "js", "node")),
        ("typescript", ("typescript", "ts")),
        ("go", ("golang", "go语言", " go ")),
        ("rust", ("rust",)),
        ("cpp", ("c++", "cpp")),
        ("csharp", ("c#", "csharp", ".net")),
        ("php", ("php",)),
        ("ruby", ("ruby",)),
    )
    for language, aliases in language_aliases:
        for alias in aliases:
            if alias == " go ":
                if re.search(r"(?<![a-z])go(?![a-z])", normalized):
                    return language
                continue
            if alias in normalized:
                return language
    return "python"


def _language_display_name(language: str) -> str:
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
    """生成对用户更友好的代码回复。"""
    cleaned_code = _strip_markdown_fences(code)
    display_name = _language_display_name(language)
    preface = f"按你的要求，我先给你整理了一个 {display_name} 示例："
    if execution_requested and not execution_supported:
        preface = (
            f"按你的要求，我先给你整理了一个 {display_name} 示例。"
            f"当前内置自动执行链路主要支持 Python，所以这段 {display_name} 代码我先不直接运行："
        )
    elif not execution_requested:
        preface = f"按你的要求，我先给你整理了一个 {display_name} 示例："
    return f"{preface}\n\n```{language}\n{cleaned_code}\n```"


def _latest_human_text(messages: List[BaseMessage]) -> str:
    """提取最近一条用户消息文本。"""
    for message in reversed(messages or []):
        if getattr(message, "type", "") == "human":
            return _message_text(message)
    return ""


def _should_execute_request(latest_human_text: str) -> bool:
    """判断当前请求是否明确要求执行代码。"""
    text = str(latest_human_text or "").strip().lower()
    if not text:
        return False

    if any(hint in text for hint in CODE_EXECUTE_HINTS):
        return True
    if any(hint in text for hint in CODE_GENERATE_ONLY_HINTS):
        return False
    return False


class CodeAgentState(TypedDict):
    """代码编写子图状态"""
    messages: Annotated[List[BaseMessage], add_messages]
    generated_code: Optional[str]  # 当前轮生成的代码正文
    should_execute: Optional[bool]  # 是否应该进入执行链路
    execution_requested: Optional[bool]  # 用户是否明确要求执行
    execution_supported: Optional[bool]  # 当前语言是否支持自动执行
    requested_language: Optional[str]  # 用户要求的代码语言
    code_to_execute: Optional[str]  # 待执行的代码
    execution_result: Optional[str]  # 执行结果


class CodeAgent(BaseAgent):
    """代码执行Agent：生成并执行Python代码"""

    def __init__(self, req: AgentRequest):
        super().__init__(req)
        if not req.model:
            raise ValueError("Code Agent 模型初始化失败，请检查配置。")
        self.llm = req.model
        self.subgraph_id = "code_agent"

        # 提示词
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CodePrompt.SYSTEM),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建代码生成/执行双模式图。"""
        workflow = StateGraph(CodeAgentState)

        # 添加节点
        workflow.add_node("generate_code", self._generate_code_node)
        workflow.add_node("execute_code", self._execute_code_node)
        workflow.add_node("analyze_result", self._analyze_result_node)

        # 定义边
        workflow.add_edge(START, "generate_code")
        workflow.add_conditional_edges("generate_code", self._route_after_generation)
        workflow.add_edge("execute_code", "analyze_result")
        workflow.add_edge("analyze_result", END)

        # 编译图 (使用全局 checkpointer)
        return workflow.compile(checkpointer=self.checkpointer)

    def _generate_code_node(self, state: CodeAgentState, config: RunnableConfig):
        """生成代码节点"""
        chain = self.prompt | self.llm
        response = chain.invoke({"messages": state["messages"]}, config=config)
        latest_human_text = _latest_human_text(state["messages"])
        requested_language = _detect_requested_language(latest_human_text)
        code = _strip_markdown_fences(_normalize_model_content(getattr(response, "content", response)))
        execution_requested = _should_execute_request(latest_human_text)
        execution_supported = requested_language in {"python"}
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
        """根据用户意图决定是否进入执行环节。"""
        return "execute_code" if state.get("should_execute") else "analyze_result"

    def _execute_code_node(self, state: CodeAgentState, config: RunnableConfig):
        """执行代码节点"""
        code = state["code_to_execute"]
        workspace_root = (
            config.get("configurable", {}).get("workspace_root")
            or self.req.state.get("workspace_root")
            or None
        )

        # --- Native Interrupt Logic ---
        decision = interrupt({
            "action_requests": [{
                "type": "code_approval",
                "name": CODE_APPROVAL_ACTION_NAME,
                "args": {"code": code, "workspace_root": workspace_root},
                "description": f"即将执行 Python 代码:\n{code[:100]}..." # 截断显示
            }],
            "message": CODE_APPROVAL_MESSAGE,
            "allowed_decisions": list(DEFAULT_ALLOWED_DECISIONS),
        })

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

    def _analyze_result_node(self, state: CodeAgentState):
        """分析结果节点"""
        if not state.get("should_execute"):
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
        result = state["execution_result"]
        # 简单包装结果
        return {"messages": [AIMessage(content=CodePrompt.EXECUTION_PREFIX.format(result))]}
