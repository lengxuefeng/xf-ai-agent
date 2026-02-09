from typing import Dict, Any, Optional
from utils.custom_logger import get_logger, LogTarget

log = get_logger(__name__)


class InterruptService:
    """人工审核服务"""

    def __init__(self):
        # 存储待审核的请求 {session_id: {message_id: request_data}}
        self.pending_approvals: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def register_pending_approval(
        self,
        session_id: str,
        message_id: str,
        action_name: str,
        action_args: Dict[str, Any],
        description: str = ""
    ):
        """
        注册待审核请求

        Args:
            session_id: 会话ID
            message_id: 消息ID
            action_name: 工具名称
            action_args: 工具参数
            description: 描述
        """
        if session_id not in self.pending_approvals:
            self.pending_approvals[session_id] = {}

        self.pending_approvals[session_id][message_id] = {
            "action_name": action_name,
            "action_args": action_args,
            "description": description,
            "status": "pending"
        }

        log.warning(f"注册待审核请求: {session_id}:{message_id}", target=LogTarget.ALL)

    def handle_approval(
        self,
        session_id: str,
        message_id: str,
        decision: str,
        user_id: Optional[int] = None,
        action_name: Optional[str] = None,
        action_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理审核结果

        Args:
            session_id: 会话ID
            message_id: 消息ID
            decision: 决策 ('approve' 或 'reject')
            user_id: 用户ID
            action_name: 工具名称
            action_args: 工具参数

        Returns:
            处理结果
        """
        # 检查是否有待审核的请求
        if session_id not in self.pending_approvals or message_id not in self.pending_approvals[session_id]:
            log.warning(f"未找到待审核请求: {session_id}:{message_id}", target=LogTarget.ALL)
            # 如果提供了action_name和action_args，临时创建一个请求
            if action_name and action_args:
                self.register_pending_approval(
                    session_id=session_id,
                    message_id=message_id,
                    action_name=action_name,
                    action_args=action_args,
                    description=""
                )
            else:
                raise ValueError("未找到待审核请求")

        # 更新审核状态
        self.pending_approvals[session_id][message_id]["status"] = decision
        self.pending_approvals[session_id][message_id]["user_id"] = user_id

        # 记录审核结果
        if decision == "approve":
            log.warning(
                f"✅ 用户 {user_id} 批准了工具调用: {action_name}",
                target=LogTarget.ALL
            )
        else:
            log.warning(
                f"❌ 用户 {user_id} 拒绝了工具调用: {action_name}",
                target=LogTarget.ALL
            )

        # TODO: 实际实现中，这里应该恢复图执行
        # 当前简化版本只记录日志
        if decision == "approve":
            # 批准后，应该继续执行被中断的图
            # 这需要保存和恢复图状态，暂时未实现
            log.warning("审核已批准，工具执行功能开发中...", target=LogTarget.ALL)
        else:
            log.warning("审核已拒绝，操作已取消", target=LogTarget.ALL)

        return {
            "session_id": session_id,
            "message_id": message_id,
            "decision": decision,
            "status": "processed"
        }

    def get_pending_approval(
        self,
        session_id: str,
        message_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取待审核请求

        Args:
            session_id: 会话ID
            message_id: 消息ID

        Returns:
            待审核请求数据
        """
        if session_id in self.pending_approvals and message_id in self.pending_approvals[session_id]:
            return self.pending_approvals[session_id][message_id]
        return None


# 创建全局实例
interrupt_service = InterruptService()
