# -*- coding: utf-8 -*-
"""
聊天服务层常量。

定义了ChatService运行时需要的所有配置和错误提示。
这些常量确保了聊天服务的稳定性和可维护性。

设计要点：
1. 错误提示要明确，让用户知道发生了什么
2. 默认配置要合理，保证基本体验
3. SSE头部配置要正确，确保流式传输不被缓冲
4. 支持灵活的模型配置，方便接入不同的LLM服务
"""

from typing import Dict, Final

from config.constants.sse_constants import SseContentType


# 超时错误提示
# 当模型服务响应时间超过配置阈值时的错误消息
# 场景：模型负载过高、网络延迟导致请求超时
CHAT_SERVICE_ERROR_TIMEOUT: Final[str] = "【系统提示】网络请求超时，请稍后重试。"

# 连接错误提示
# 当无法连接到模型服务时的错误消息
# 场景：网络断开、模型服务宕机、DNS解析失败
CHAT_SERVICE_ERROR_CONNECTION: Final[str] = "【系统提示】模型服务连接断开，请检查网络或重试。"

# 运行时错误提示模板
# 底层服务执行异常时的错误消息模板，需要填入具体的错误信息
# 场景：数据库操作失败、代码执行异常、API调用报错
CHAT_SERVICE_ERROR_RUNTIME_TEMPLATE: Final[str] = "【系统提示】底层服务执行异常: {error}"

# 通用错误提示模板
# 其他未明确分类的错误使用的模板
# 场景：代码bug、配置错误等
CHAT_SERVICE_ERROR_FALLBACK_TEMPLATE: Final[str] = "发生异常: {error}"

# 流程中断提示模板
# 当生成过程被中断时的提示，需要填入中断原因
# 场景：用户点击停止按钮、客户端断开连接、系统超时
CHAT_SERVICE_INTERRUPTED_TEMPLATE: Final[str] = "【系统提示: 生成被中断 ({error})】"

# 流程中断追加提示
# 当生成过程中断后，在已有内容后追加的提示
# 场景：已经生成了部分内容，然后被中断，需要告知用户内容不完整
CHAT_SERVICE_INTERRUPTED_APPEND_TEMPLATE: Final[str] = "\n\n系统提示：输出异常中断（{error}）"

# SSE流式传输的HTTP头部配置
# 这些头部确保了SSE的正确传输，避免被浏览器或代理服务器缓冲
STREAM_HEADERS: Final[Dict[str, str]] = {
    "Cache-Control": "no-cache",         # 禁用缓存，确保实时推送
    "Connection": "keep-alive",          # 保持长连接
    "X-Accel-Buffering": "no",          # Nginx不缓冲，直接推送
}

# 默认模型配置
# 用户没有自定义模型配置时使用的默认值
# 这些参数是根据智谱GLM-4模型优化过的，其他模型可能需要调整
CHAT_DEFAULT_MODEL_CONFIG: Final[Dict[str, object]] = {
    # 核心模型配置
    "model": "glm-4.7",                  # 模型名称，GLM-4.7是智谱的最新模型
    "model_service": "zhipu",           # 模型服务提供商
    "service_type": "zhipu",            # 服务类型，用于路由器选择

    # 高级功能开关
    "deep_thinking_mode": "auto",       # 深度思考模式，auto表示根据任务复杂度自动决定
    "rag_enabled": False,                # RAG（检索增强生成）开关，默认关闭以节省token

    # RAG相关配置
    "similarity_threshold": 0.7,        # 相似度阈值，用于检索相关文档
    "embedding_model": "bge-m3:latest", # 嵌入模型，用于文档向量化
    "embedding_model_key": "",         # 嵌入模型的API密钥（如果需要）

    # 模型推理参数
    "temperature": 0.2,                 # 温度参数，越低输出越稳定，越高越随机
    "top_p": 1.0,                      # 核采样参数，控制输出的多样性
    "max_tokens": 2000,                 # 最大输出token数，控制回复长度

    # API访问配置
    "model_key": "",                    # 模型API密钥
    "model_url": "",                    # 模型服务地址
}

# AI内容类型集合
# 用于在SSE事件流中识别哪些事件包含AI生成的文本内容
# 场景：前端需要提取STREAM和MESSAGE类型的内容，拼接成完整回复
CHAT_AI_CONTENT_TYPES = (SseContentType.STREAM, SseContentType.MESSAGE)
