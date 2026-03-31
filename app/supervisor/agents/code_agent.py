import re

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from common.utils.code_tools import execute_python_code
from config.constants.approval_constants import (
    ApprovalDecision,
    CODE_APPROVAL_ACTION_NAME,
    CODE_APPROVAL_MESSAGE,
    DEFAULT_ALLOWED_DECISIONS,
)
from prompts.agent_prompts.code_prompt import CodePrompt
from supervisor.base import BaseAgent
from supervisor.graph_state import AgentRequest, AgentState


def _strip_markdown_fences(code: str) -> str:
    raw = str(code or "").strip()
    if not raw:
        return ""
    fenced_match = re.fullmatch(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", raw, flags=re.S)
    if fenced_match:
        return fenced_match.group(1).strip()
    return raw.replace("```", "").strip()


@tool(CODE_APPROVAL_ACTION_NAME)
def execute_python_tool(code: str, workspace_root: str = "") -> str:
    """仅当用户明确要求运行、执行或测试 Python 代码时调用。参数 code 必须是完整可运行的 Python 代码。"""
    normalized_code = _strip_markdown_fences(code)
    if not normalized_code:
        return "未提供可执行的 Python 代码。"

    normalized_workspace = str(workspace_root or "").strip() or None
    decision = interrupt(
        {
            "action_requests": [
                {
                    "type": "code_approval",
                    "name": CODE_APPROVAL_ACTION_NAME,
                    "args": {"code": normalized_code, "workspace_root": normalized_workspace},
                    "description": f"即将执行 Python 代码:\n{normalized_code[:200]}",
                }
            ],
            "message": CODE_APPROVAL_MESSAGE,
            "allowed_decisions": list(DEFAULT_ALLOWED_DECISIONS),
        }
    )
    action = decision.get("action") if isinstance(decision, dict) else decision
    if action == ApprovalDecision.REJECT.value:
        return "用户拒绝执行代码。"

    try:
        result = execute_python_code(
            normalized_code,
            cwd=normalized_workspace,
            workspace_root=normalized_workspace,
        )
    except Exception as exc:
        return f"代码执行错误: {exc}"
    return CodePrompt.EXECUTION_PREFIX.format(str(result))


class CodeAgent(BaseAgent):
    def __init__(self, req: AgentRequest):
        super().__init__(req)
        if not req.model:
            raise ValueError("Code Agent 模型初始化失败，请检查配置。")
        self.llm = req.model
        self.subgraph_id = "code_agent"
        self.system_prompt = (
            f"{CodePrompt.SYSTEM}"
            "当用户明确要求运行、执行、测试或验证 Python 代码时，必须调用执行工具。"
            "如果用户只是要代码示例，则直接回答，不要调用工具。"
        )
        self.tools = [execute_python_tool]
        self.graph = self._build_graph()

    async def _model_node(self, state: AgentState, config: RunnableConfig):
        messages = list(state.get("messages", []) or [])
        llm_with_tools = self.llm.bind_tools(self.tools)
        response = await llm_with_tools.ainvoke(messages, config=config)
        return {"messages": [response]}

    def _build_graph(self) -> Runnable:
        workflow = StateGraph(AgentState)
        workflow.add_node("model_node", self._model_node, retry_policy=self.RETRY_POLICY)
        workflow.add_node("tools", ToolNode(self.tools), retry_policy=self.RETRY_POLICY)
        workflow.add_edge(START, "model_node")
        workflow.add_conditional_edges("model_node", tools_condition)
        workflow.add_edge("tools", "model_node")
        return workflow.compile(checkpointer=self.checkpointer)
