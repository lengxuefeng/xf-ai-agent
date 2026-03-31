"""
医疗问答 Agent：处理医疗健康相关问题的智能体

核心职责：
- 回答用户关于医疗、健康、疾病等方面的问题
- 提供专业的医疗健康建议
- 在回答后附加免责声明，强调非专业医疗诊断
- 确保回答的专业性和安全性

业务场景示例：
1. 用户问"高血压的症状是什么？" → 生成专业回答 + 免责声明
2. 用户问"如何预防糖尿病？" → 提供预防建议 + 免责声明
3. 用户问"感冒应该吃什么药？" → 给出建议 + 免责声明
4. 用户问"什么是健康的生活方式？" → 提供健康建议 + 免责声明

设计要点：
- 基于 LangGraph 构建简单的单节点子图
- 使用专业的医疗提示词系统
- 所有回答后强制附加免责声明
- 强调 AI 回答仅供参考，不能替代专业医疗诊断
- 保持专业、客观、安全的回答风格

与其他 Agent 的区别：
- weather_agent：使用天气 API
- search_agent：使用搜索引擎
- sql_agent：使用数据库查询
- code_agent：生成和执行代码
- medical_agent：回答医疗健康问题（纯 LLM，无工具调用）

安全考虑：
- 所有回答必须附加免责声明
- 明确表示不能替代专业医疗诊断
- 遇到紧急情况建议及时就医
- 避免给出具体的处方用药建议

输出结果：
- 专业的医疗健康建议
- 强制附加的免责声明
- 友好的提示语

免责声明模板：
"注：以上内容仅供参考，不能替代专业医疗诊断。如有健康问题，请及时就医。"
"""

from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END, add_messages

from supervisor.base import BaseAgent
from supervisor.graph_state import AgentRequest
from common.utils.custom_logger import get_logger
from prompts.agent_prompts.medical_prompt import MedicalPrompt

log = get_logger(__name__)


class MedicalAgentState(TypedDict):
    """
    医疗问答子图状态

    状态字段说明：
    - messages: 消息列表，包括用户消息、AI 消息

    使用场景：
    - 在子图执行过程中维护状态
    - 跨节点传递消息
    - 支持多轮对话
    """
    messages: Annotated[List[BaseMessage], add_messages]


class MedicalAgent(BaseAgent):
    """
    医疗问答智能体 (LangGraph 1.0 Refactored)

    主要功能：
    - 回答医疗健康相关问题
    - 提供专业的健康建议
    - 所有回答后附加免责声明
    - 确保回答的专业性和安全性

    工作流程：
    1. 接收用户问题
    2. 使用 LLM 生成专业回答
    3. 在回答后附加免责声明
    4. 返回最终回复

    典型使用场景：
    - "高血压的症状是什么？"
    - "如何预防糖尿病？"
    - "感冒应该吃什么药？"
    - "什么是健康的生活方式？"

    安全机制：
    - 所有回答必须附加免责声明
    - 强调不能替代专业医疗诊断
    - 遇到紧急情况建议及时就医

    与其他 Agent 的区别：
    - 纯 LLM 对话，无工具调用
    - 使用医疗专用提示词
    - 强制附加免责声明
    """

    def __init__(self, req: AgentRequest):
        """
        初始化医疗问答 Agent

        参数说明：
        - req: Agent 请求对象，包含模型配置、会话信息等

        初始化步骤：
        1. 调用父类 BaseAgent 初始化
        2. 验证模型配置，确保医疗模型可用
        3. 创建 LLM 实例
        4. 构建医疗问答的提示词模板
        5. 构建医疗子图

        异常处理：
        - 如果模型未配置，抛出 ValueError 提示检查配置

        示例：
        >>> req = AgentRequest(model=llm, session_id="xxx")
        >>> medical_agent = MedicalAgent(req)
        >>> result = medical_agent.invoke({"messages": [HumanMessage("高血压的症状是什么？")]})
        """
        super().__init__(req)
        if not req.model:
            raise ValueError("医疗模型初始化失败，请检查配置。")
        self.llm = req.model
        self.subgraph_id = "medical_agent"

        # 提示词：医疗问答的系统提示词
        # SYSTEM 包含医疗问答的专业要求、安全规范等
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", MedicalPrompt.SYSTEM),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.graph = self._build_graph()

    async def _model_node(self, state: MedicalAgentState):
        """
        模型节点：生成医疗问答回复

        功能说明：
        - 使用 LLM 生成专业的医疗健康回答
        - 在回答后强制附加免责声明
        - 确保回答的专业性和安全性

        处理流程：
        1. 调用 LLM 生成回答
        2. 在回答内容后附加免责声明
        3. 返回 AI 消息

        免责声明：
        - MedicalPrompt.DISCLAIMER 包含标准的免责声明文本
        - 所有回答都必须附加
        - 强调不能替代专业医疗诊断

        参数说明：
        - state: 子图状态

        返回值：
        - dict: 更新后的状态，包含 AI 消息

        示例：
        >>> state = {"messages": [HumanMessage("高血压的症状是什么？")]}
        >>> medical_agent._model_node(state)
        {"messages": [AIMessage(content="高血压的主要症状包括头晕、头痛、心悸等...\\n\\n注：以上内容仅供参考，不能替代专业医疗诊断。如有健康问题，请及时就医。")]}
        """
        chain = self.prompt | self.llm
        response = await chain.ainvoke(state)

        # 在回答后附加免责声明
        response.content = f"{self._message_text(response)}{MedicalPrompt.DISCLAIMER}"

        return {"messages": [response]}

    def _build_graph(self) -> Runnable:
        """
        构建医疗问答子图

        子图结构：
        - 单节点设计，直接调用模型生成回答

        流程：
        - START → agent → END

        设计说明：
        - 医疗问答不需要工具调用，因此采用简单的单节点设计
        - 所有逻辑在 _model_node 中完成
        - 使用全局 checkpointer 支持中断恢复

        返回值：
        - Runnable: 编译后的可执行子图，使用全局 checkpointer
        """
        workflow = StateGraph(MedicalAgentState)
        workflow.add_node("agent", self._model_node, retry_policy=self.RETRY_POLICY)
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)

        return workflow.compile(checkpointer=self.checkpointer)
