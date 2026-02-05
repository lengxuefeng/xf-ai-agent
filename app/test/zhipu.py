# from langchain_community.llms import ZhipuAI
# from langchain_core.messages import HumanMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from pydantic import SecretStr
# # 1. 初始化智谱AI LLM（指定glm-4v-flash模型）
# # llm = ZhipuAI(
# #     model="glm-4v-flash",  # 多模态模型名称
# #     api_key="ms-6c650c00-122f-48ba-b393-fd2ce93c4526",  # 替换为你的API Key
# #     temperature=0.1,  # 温度越低，结果越稳定（OCR场景推荐0.1）
# # )
#
# llm =  ChatOpenAI(
#         model="ZhipuAI/glm-4v-flash",
#         api_key=SecretStr("ms-6c650c00-122f-48ba-b393-fd2ce93c4526"),
#         base_url="https://api-inference.modelscope.cn/v1",
#     )
#
# # 2. 定义Prompt（明确OCR任务要求）
# prompt = ChatPromptTemplate.from_messages([
#     HumanMessage(
#         content=[
#             {"type": "text", "text": "请识别图片中的所有文字，包括格式（表格、换行），输出纯文本结果，不要额外解释："},
#             # 支持图片URL或Base64编码（二选一）
#             {"type": "image_url", "image_url": {"url": "https://www.chiniurou.com/uploads/allimg/250221/1-250221105140404.JPG"}},  # 图片URL
#             # {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."}},  # Base64编码（本地图片转Base64后传入）
#         ]
#     )
# ])
#
# # 3. 构建LangChain工作流（Prompt + LLM + 输出解析）
# chain = prompt | llm | StrOutputParser()
#
# # 4. 执行调用（OCR识别）
# result = chain.invoke({})
# print("OCR识别结果：")
# print(result)

from openai import OpenAI
import base64
import requests


def ocr_with_openai_compatible(image_url):
    # 使用 ModelScope 的 OpenAI 兼容接口
    client = OpenAI(
        api_key="ms-6c650c00-122f-48ba-b393-fd2ce93c4526",
        base_url="https://api-inference.modelscope.cn/v1/",
    )

    # 下载图片并转换为 base64
    response = requests.get(image_url)
    image_base64 = base64.b64encode(response.content).decode('utf-8')

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请识别图片中的所有文字，包括格式（表格、换行），输出纯文本结果，不要额外解释"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="ZhipuAI/glm-4v-flash",  # 使用支持的模型
            messages=messages,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"错误: {e}"


# 执行 OCR
image_url = "https://www.chiniurou.com/uploads/allimg/250221/1-250221105140404.JPG"
result = ocr_with_openai_compatible(image_url)
print("OCR识别结果：")
print(result)