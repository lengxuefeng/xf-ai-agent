import asyncio

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
# from langchain.agents import create_tool_calling_agent, AgentExecutor


# 工具定义
@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    # import eval
    try:
        result = "eval(expression)"
        return f"计算结果: {result}"
    except:
        return "计算错误"


# 配置支持推理的模型
# llm = ChatOpenAI(
#     model="GLM-4.6",
#     api_key="your-api-key",
#     base_url="https://open.bigmodel.cn/api/coding/paas/v4",
#     temperature=0.7,
#     extra_body={
#         "reasoning_enabled": True,
#         "max_reasoning_tokens": 1024,
#     },
# )

llm = ChatOpenAI(
# model = ChatZhipuAI(
    model="GLM-4.6",
    temperature=0.5,
    api_key="372825e177b84308934ad6564cab60f7.1yrv1Sv8mL7YhHZA",
    # base_url="https://open.bigmodel.cn/api/paas/v4",
    base_url="https://open.bigmodel.cn/api/coding/paas/v4",
    extra_body={
        "reasoning_enabled": True,
        "max_reasoning_tokens": 1024,
    },
)
# 创建 Agent
tools = [calculate]
# agent = create_tool_calling_agent(llm, tools)
agent_executor = create_agent(
    model=llm,
    tools=tools,
    # verbose=True,
)


async def run():
    """运行并展示推理过程"""

    reasoning_content = ""
    final_answer = ""

    async for event in agent_executor.astream_events(
            {"input": "计算 123 * 456 + 789"},
            version="v1"
    ):
        event_type = event["event"]

        # 获取推理内容
        if "on_chat_model_end" in event_type:
            output = event.get("data", {}).get("output", {})
            if hasattr(output, 'response_metadata'):
                metadata = output.response_metadata

                if 'reasoning_content' in metadata:
                    reasoning_content = metadata['reasoning_content']
                    print(f"\n💭 思考过程:\n{reasoning_content}\n")

                if 'token_usage' in metadata:
                    usage = metadata['token_usage']
                    reasoning_tokens = usage.get('reasoning_tokens', 0)
                    if reasoning_tokens > 0:
                        print(f"💭 推理使用了 {reasoning_tokens} tokens\n")

        # 流式输出
        elif "on_chat_model_stream" in event_type:
            chunk = event["data"]["chunk"]
            final_answer += chunk.content
            print(chunk.content, end="")


asyncio.run(run())