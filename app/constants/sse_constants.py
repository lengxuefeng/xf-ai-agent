# -*- coding: utf-8 -*-
"""SSE 事件类型与统一提示文案。"""
from enum import Enum


class SseEventType(str, Enum):
    """SSE 事件类型枚举"""

    # 响应开始事件
    RESPONSE_START = "response_start"

    # 响应结束事件
    RESPONSE_END = "response_end"

    # 思考过程事件
    THINKING = "thinking"

    # 流式内容事件
    STREAM = "stream"

    # 错误事件
    ERROR = "error"

    # 中断审批事件
    INTERRUPT = "interrupt"

    # 日志事件
    LOG = "log"

    # 结构化工作流事件
    WORKFLOW_EVENT = "workflow_event"


class SsePayloadField(str, Enum):
    """SSE 负载字段名枚举"""

    # 事件类型字段
    TYPE = "type"

    # 事件内容字段
    CONTENT = "content"

    # 结构化负载字段
    PAYLOAD = "payload"


class SseMessage:
    """GraphRunner 常用提示文案常量"""

    # 检测到审批恢复操作
    RESUME_DETECTED = "检测到审批结果，正在恢复执行..."

    # 恢复参数无效
    INVALID_RESUME_PARAM = "参数无效或无待恢复任务"

    # 处理中心跳提示
    PROCESSING_HEARTBEAT = "系统正在处理中，请稍候..."

    # 超时错误提示
    ERROR_TIMEOUT = "处理超时，请稍后重试。"

    # 连接异常错误提示
    ERROR_CONNECTION = "模型服务连接异常，请稍后重试。"

    # 运行期错误提示
    ERROR_RUNTIME = "底层服务执行异常，请稍后重试。"

    # 空闲超时错误提示
    ERROR_IDLE_TIMEOUT = "长时间未收到模型响应，请重试。"

    # 任务中断错误提示
    ERROR_TASK_INTERRUPTED = "任务已中断，请重试。"

    # 需要人工审批提示
    NEED_MANUAL_APPROVAL = "检测到需要人工审核，请点击“批准/拒绝”后继续。"

    # 恢复后仍需审批提示
    RESUME_NEED_MANUAL_APPROVAL = "恢复执行后仍需人工审核，请继续审批。"

    # 未找到中断任务错误
    ERROR_NO_INTERRUPTED_TASK = "未找到处于中断状态的子任务"

    # 恢复无结果错误
    ERROR_RESUME_EMPTY_RESULT = "恢复执行未生成结果，请重试。"


class SseContentType:
    """用于正文提取的消息类型常量"""

    # 流式内容类型
    STREAM = SseEventType.STREAM.value

    # 消息内容类型
    MESSAGE = "message"
