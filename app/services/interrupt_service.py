from datetime import datetime
from typing import Dict, Any, Optional, List

from constants.approval_constants import (
    APPROVAL_HANDLE_STATUS_PROCESSED,
    ApprovalDecision,
    ApprovalStatus,
    DEFAULT_ALLOWED_DECISIONS,
)
from db import get_db_context
from models.interrupt_approval import InterruptApprovalModel
from utils.custom_logger import get_logger, LogTarget

log = get_logger(__name__)


class InterruptService:
    """中断审批服务：处理Agent执行过程中的人工审批请求"""

    def __init__(self):
        """初始化服务，准备内存兜底容器"""
        # 兜底内存态：仅在数据库不可用时启用
        self.pending_approvals: Dict[str, Dict[str, Dict[str, Any]]] = {}

    @staticmethod
    def _normalize_decision(decision: str) -> str:
        """标准化审批动作为approve或reject"""
        d = (decision or "").strip().lower()
        if d not in DEFAULT_ALLOWED_DECISIONS:
            raise ValueError("decision 必须是 approve 或 reject")
        return d

    @staticmethod
    def _row_to_dict(row: InterruptApprovalModel) -> Dict[str, Any]:
        """将ORM行对象转换为字典"""
        return {
            "id": row.id,
            "session_id": row.session_id,
            "message_id": row.message_id,
            "action_name": row.action_name,
            "action_args": row.action_args or {},
            "description": row.description or "",
            "status": row.status,
            "user_id": row.user_id,
            "decision_time": row.decision_time.isoformat() if row.decision_time else None,
            "agent_name": row.agent_name,
            "subgraph_thread_id": row.subgraph_thread_id,
            "checkpoint_id": row.checkpoint_id,
            "checkpoint_ns": row.checkpoint_ns,
            "is_consumed": bool(row.is_consumed),
            "create_time": row.create_time.isoformat() if row.create_time else None,
            "update_time": row.update_time.isoformat() if row.update_time else None,
        }

    @staticmethod
    def _query_by_message(db, session_id: str, message_id: str) -> Optional[InterruptApprovalModel]:
        """按会话和消息ID查询审批记录"""
        return (
            db.query(InterruptApprovalModel)
            .filter(
                InterruptApprovalModel.session_id == session_id,
                InterruptApprovalModel.message_id == message_id,
            )
            .order_by(InterruptApprovalModel.id.desc())
            .first()
        )

    @staticmethod
    def _query_session_pending(db, session_id: str) -> List[InterruptApprovalModel]:
        """查询会话所有待审批且未消费的记录"""
        return (
            db.query(InterruptApprovalModel)
            .filter(
                InterruptApprovalModel.session_id == session_id,
                InterruptApprovalModel.status == ApprovalStatus.PENDING.value,
                InterruptApprovalModel.is_consumed.is_(False),
            )
            .order_by(InterruptApprovalModel.create_time.desc(), InterruptApprovalModel.id.desc())
            .all()
        )

    def _upsert_pending_row(
        self,
        db,
        session_id: str,
        message_id: str,
        action_name: str,
        action_args: Dict[str, Any],
        description: str = "",
        agent_name: Optional[str] = None,
        subgraph_thread_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        checkpoint_ns: Optional[str] = None,
    ) -> InterruptApprovalModel:
        """写入或更新待审批记录"""
        row = self._query_by_message(db, session_id, message_id)
        if row:
            row.action_name = action_name
            row.action_args = action_args or {}
            row.description = description or ""
            row.status = ApprovalStatus.PENDING.value
            row.user_id = None
            row.decision_time = None
            row.is_consumed = False
            row.agent_name = agent_name or row.agent_name
            row.subgraph_thread_id = subgraph_thread_id or row.subgraph_thread_id
            row.checkpoint_id = checkpoint_id or row.checkpoint_id
            row.checkpoint_ns = checkpoint_ns or row.checkpoint_ns
            return row

        row = InterruptApprovalModel(
            session_id=session_id,
            message_id=message_id,
            action_name=action_name,
            action_args=action_args or {},
            description=description or "",
            status=ApprovalStatus.PENDING.value,
            user_id=None,
            decision_time=None,
            is_consumed=False,
            agent_name=agent_name,
            subgraph_thread_id=subgraph_thread_id,
            checkpoint_id=checkpoint_id,
            checkpoint_ns=checkpoint_ns,
        )
        db.add(row)
        db.flush()
        return row

    def _fallback_register(
        self,
        session_id: str,
        message_id: str,
        action_name: str,
        action_args: Dict[str, Any],
        description: str = "",
        agent_name: Optional[str] = None,
        subgraph_thread_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        checkpoint_ns: Optional[str] = None,
    ):
        """数据库不可用时，将审批请求存入内存"""
        if session_id not in self.pending_approvals:
            self.pending_approvals[session_id] = {}
        self.pending_approvals[session_id][message_id] = {
            "action_name": action_name,
            "action_args": action_args or {},
            "description": description or "",
            "status": ApprovalStatus.PENDING.value,
            "user_id": None,
            "decision_time": None,
            "agent_name": agent_name,
            "subgraph_thread_id": subgraph_thread_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_ns": checkpoint_ns,
            "is_consumed": False,
        }
        log.warning(f"[fallback] 注册待审核请求: {session_id}:{message_id}", target=LogTarget.ALL)

    def register_pending_approval(
        self,
        session_id: str,
        message_id: str,
        action_name: str,
        action_args: Dict[str, Any],
        description: str = "",
        agent_name: Optional[str] = None,
        subgraph_thread_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        checkpoint_ns: Optional[str] = None,
    ):
        """注册待审核请求，优先写入数据库，失败则回退内存"""
        try:
            with get_db_context() as db:
                self._upsert_pending_row(
                    db=db,
                    session_id=session_id,
                    message_id=message_id,
                    action_name=action_name,
                    action_args=action_args or {},
                    description=description or "",
                    agent_name=agent_name,
                    subgraph_thread_id=subgraph_thread_id,
                    checkpoint_id=checkpoint_id,
                    checkpoint_ns=checkpoint_ns,
                )
            log.warning(f"注册待审核请求(PG): {session_id}:{message_id}", target=LogTarget.ALL)
        except Exception as e:
            log.error(f"注册待审核请求写入 PG 失败，降级内存: {e}", target=LogTarget.ALL)
            self._fallback_register(
                session_id=session_id,
                message_id=message_id,
                action_name=action_name,
                action_args=action_args or {},
                description=description or "",
                agent_name=agent_name,
                subgraph_thread_id=subgraph_thread_id,
                checkpoint_id=checkpoint_id,
                checkpoint_ns=checkpoint_ns,
            )

    def handle_approval(
        self,
        session_id: str,
        message_id: str,
        decision: str,
        user_id: Optional[int] = None,
        action_name: Optional[str] = None,
        action_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """处理审批结果，优先写入数据库，失败则回退内存"""
        normalized_decision = self._normalize_decision(decision)

        try:
            with get_db_context() as db:
                row = self._query_by_message(db, session_id, message_id)
                if not row:
                    log.warning(f"未找到待审核请求(PG): {session_id}:{message_id}", target=LogTarget.ALL)
                    pending_rows = self._query_session_pending(db, session_id)
                    if len(pending_rows) == 1:
                        row = pending_rows[0]
                        message_id = row.message_id
                        log.warning(
                            f"使用兜底待审核ID(PG): {session_id}:{message_id}",
                            target=LogTarget.ALL,
                        )

                if not row and action_name and action_args is not None:
                    row = self._upsert_pending_row(
                        db=db,
                        session_id=session_id,
                        message_id=message_id,
                        action_name=action_name,
                        action_args=action_args,
                        description="",
                    )

                if not row:
                    raise ValueError("未找到待审核请求")

                row.status = normalized_decision
                row.user_id = user_id
                row.decision_time = datetime.utcnow()
                row.is_consumed = False
                log_action_name = action_name or row.action_name

            if normalized_decision == ApprovalDecision.APPROVE.value:
                log.warning(f"✅ 用户 {user_id} 批准了工具调用: {log_action_name}", target=LogTarget.ALL)
                log.warning("审核已批准，等待前端触发恢复执行。", target=LogTarget.ALL)
            else:
                log.warning(f"❌ 用户 {user_id} 拒绝了工具调用: {log_action_name}", target=LogTarget.ALL)
                log.warning("审核已拒绝，操作已取消", target=LogTarget.ALL)

            return {
                "session_id": session_id,
                "message_id": message_id,
                "decision": normalized_decision,
                "status": APPROVAL_HANDLE_STATUS_PROCESSED,
            }
        except ValueError:
            raise
        except Exception as e:
            log.error(f"处理审批写入 PG 失败，降级内存: {e}", target=LogTarget.ALL)
            if session_id not in self.pending_approvals or message_id not in self.pending_approvals[session_id]:
                session_pending = self.pending_approvals.get(session_id, {})
                if len(session_pending) == 1:
                    message_id = next(iter(session_pending.keys()))
                elif action_name and action_args is not None:
                    self._fallback_register(
                        session_id=session_id,
                        message_id=message_id,
                        action_name=action_name,
                        action_args=action_args,
                        description="",
                    )
                else:
                    raise ValueError("未找到待审核请求")

            self.pending_approvals[session_id][message_id]["status"] = normalized_decision
            self.pending_approvals[session_id][message_id]["user_id"] = user_id
            self.pending_approvals[session_id][message_id]["decision_time"] = datetime.utcnow().isoformat()
            self.pending_approvals[session_id][message_id]["is_consumed"] = False
            return {
                "session_id": session_id,
                "message_id": message_id,
                "decision": normalized_decision,
                "status": APPROVAL_HANDLE_STATUS_PROCESSED,
            }

    def get_pending_approval(
        self,
        session_id: str,
        message_id: str
    ) -> Optional[Dict[str, Any]]:
        """获取单条审批请求"""
        try:
            with get_db_context() as db:
                row = self._query_by_message(db, session_id, message_id)
                if not row:
                    return None
                return self._row_to_dict(row)
        except Exception as e:
            log.error(f"读取审批请求(PG)失败，降级内存: {e}", target=LogTarget.ALL)
            return self.pending_approvals.get(session_id, {}).get(message_id)

    def fetch_latest_resolved_approval(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话最新已审批未消费的记录，用于恢复执行"""
        try:
            with get_db_context() as db:
                row = (
                    db.query(InterruptApprovalModel)
                    .filter(
                        InterruptApprovalModel.session_id == session_id,
                        InterruptApprovalModel.status.in_([
                            ApprovalStatus.APPROVE.value,
                            ApprovalStatus.REJECT.value,
                        ]),
                        InterruptApprovalModel.is_consumed.is_(False),
                    )
                    .order_by(
                        InterruptApprovalModel.decision_time.desc().nullslast(),
                        InterruptApprovalModel.update_time.desc(),
                        InterruptApprovalModel.id.desc(),
                    )
                    .first()
                )
                if not row:
                    return None
                return self._row_to_dict(row)
        except Exception as e:
            log.error(f"读取恢复审批记录(PG)失败，降级内存: {e}", target=LogTarget.ALL)
            approvals = self.pending_approvals.get(session_id, {})
            for message_id, data in reversed(list(approvals.items())):
                if data.get("status") in {
                    ApprovalStatus.APPROVE.value,
                    ApprovalStatus.REJECT.value,
                } and not data.get("is_consumed", False):
                    d = dict(data)
                    d["session_id"] = session_id
                    d["message_id"] = message_id
                    return d
            return None

    def fetch_resolved_approval_by_message(self, session_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        """按审批消息 ID 精确读取已审批且未消费的审批记录。"""
        if not message_id:
            return None
        try:
            with get_db_context() as db:
                row = self._query_by_message(db, session_id, message_id)
                if not row:
                    return None
                row_data = self._row_to_dict(row)
                if row_data["status"] not in {ApprovalStatus.APPROVE.value, ApprovalStatus.REJECT.value}:
                    return None
                if row_data["is_consumed"]:
                    return None
                return row_data
        except Exception as e:
            log.error(f"按消息读取恢复审批记录(PG)失败，降级内存: {e}", target=LogTarget.ALL)
            data = self.pending_approvals.get(session_id, {}).get(message_id)
            if not data:
                return None
            if data.get("status") not in {ApprovalStatus.APPROVE.value, ApprovalStatus.REJECT.value}:
                return None
            if data.get("is_consumed", False):
                return None
            result = dict(data)
            result["session_id"] = session_id
            result["message_id"] = message_id
            return result

    def mark_approval_consumed(self, session_id: str, message_id: str):
        """标记审批为已消费，防止重复恢复"""
        if not message_id:
            return
        try:
            with get_db_context() as db:
                row = self._query_by_message(db, session_id, message_id)
                if not row:
                    return
                row.is_consumed = True
        except Exception as e:
            log.error(f"标记审批消费(PG)失败，降级内存: {e}", target=LogTarget.ALL)
            data = self.pending_approvals.get(session_id, {}).get(message_id)
            if data:
                data["is_consumed"] = True


# 创建全局实例
interrupt_service = InterruptService()
