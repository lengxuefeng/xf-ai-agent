import tempfile
import time
import unittest
from pathlib import Path

from agent.agents.code_agent import _should_execute_request
from api.v1.health_api import _check_runtime_harness
from runtime.core.run_context import build_run_context
from runtime.core.run_state_store import run_state_store
from runtime.core.session_manager import runtime_session_manager
from runtime.exec.command_session_service import command_session_service
from runtime.workspace.manager import workspace_manager
from runtime.workspace.path_guard import workspace_path_guard
from services.chat_service import ChatService
from utils.code_tools import execute_python_code


class RuntimeHarnessLayersTest(unittest.TestCase):
    """运行时 Harness 核心层回归测试。"""

    def test_workspace_and_extra_data_include_runtime_snapshot_and_artifacts(self):
        """消息扩展数据应带上运行态快照和产物清单。"""
        session_id = "sess-runtime-extra"
        run_context = build_run_context(
            session_id=session_id,
            user_input="给我总结一下 Harness",
            run_id="sess-runtime-extra:run",
        )
        runtime_session_manager.register_run(run_context)
        workspace_manager.prepare_run_workspace(run_context)
        workspace_manager.write_text_artifact(
            run_context,
            name="bootstrap_note.txt",
            content="runtime ready",
            category="artifact",
        )
        runtime_session_manager.attach_meta(run_context, workspace=workspace_manager.prepare_run_workspace(run_context))
        runtime_session_manager.mark_completed(
            run_context,
            phase="response_completed",
            summary="运行已完成",
            title="运行完成",
        )

        service = ChatService()
        extra = service._build_extra_data(
            ["掌柜正在调度"],
            [{"phase": "workspace_prepared", "title": "卷宗工位已就绪"}],
            session_id=session_id,
            final_response="最终回报",
        )

        self.assertIsInstance(extra, dict)
        self.assertIn("runtime_snapshot", extra)
        self.assertIn("runtime_artifacts", extra)
        self.assertTrue(any(item["name"] == "final_response.md" for item in extra["runtime_artifacts"]))
        self.assertEqual(extra["runtime_snapshot"]["status"], "completed")

        run_state_store.remove(run_context.run_id)

    def test_execute_python_code_uses_runtime_exec_runner(self):
        output = execute_python_code("print('hello harness')")
        self.assertIn("hello harness", output)

    def test_workspace_guard_rejects_outside_allowed_root(self):
        """页面终端工作目录不应越过允许的根目录。"""
        with self.assertRaises(ValueError):
            workspace_path_guard.resolve_workspace_root("/tmp")

    def test_workspace_manager_can_list_create_read_and_save_files_inside_bound_root(self):
        """工作台应支持在已绑定目录内浏览、创建、读取和保存文件。"""
        allowed_parent = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=str(allowed_parent)) as temp_dir:
            session_id = "sess-workbench-tree"
            workspace_manager.bind_external_workspace(session_id, temp_dir)

            created_dir = workspace_manager.create_workspace_directory(session_id, "src")
            self.assertEqual(created_dir["path"], "src")

            created_file = workspace_manager.create_workspace_file(
                session_id,
                "src/main.py",
                "print('hello workbench')\n",
            )
            self.assertEqual(created_file["path"], "src/main.py")

            root_listing = workspace_manager.list_workspace_directory(session_id)
            self.assertTrue(any(item["path"] == "src" and item["type"] == "directory" for item in root_listing["items"]))

            src_listing = workspace_manager.list_workspace_directory(session_id, "src")
            self.assertTrue(any(item["path"] == "src/main.py" and item["type"] == "file" for item in src_listing["items"]))

            loaded = workspace_manager.read_workspace_file(session_id, "src/main.py")
            self.assertIn("hello workbench", loaded["content"])

            saved = workspace_manager.save_workspace_file(session_id, "src/main.py", "print('updated')\n")
            self.assertIn("updated", saved["content"])

    def test_workspace_manager_rejects_path_escape_when_saving_files(self):
        """工作台保存文件时不允许越出已绑定目录。"""
        allowed_parent = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=str(allowed_parent)) as temp_dir:
            session_id = "sess-workbench-guard"
            workspace_manager.bind_external_workspace(session_id, temp_dir)
            with self.assertRaises(ValueError):
                workspace_manager.save_workspace_file(session_id, "../escape.txt", "blocked")

    def test_workspace_manager_can_rename_and_delete_entries_inside_bound_root(self):
        """工作台应支持在绑定目录内重命名和删除文件、目录。"""
        allowed_parent = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=str(allowed_parent)) as temp_dir:
            session_id = "sess-workbench-mutate"
            workspace_manager.bind_external_workspace(session_id, temp_dir)
            workspace_manager.create_workspace_directory(session_id, "src")
            workspace_manager.create_workspace_file(session_id, "src/main.py", "print('hello')\n")

            renamed = workspace_manager.rename_workspace_entry(session_id, "src/main.py", "src/app.py")
            self.assertEqual(renamed["path"], "src/app.py")
            self.assertTrue((Path(temp_dir) / "src" / "app.py").exists())

            deleted_file = workspace_manager.delete_workspace_entry(session_id, "src/app.py")
            self.assertEqual(deleted_file["path"], "src/app.py")
            self.assertFalse((Path(temp_dir) / "src" / "app.py").exists())

            workspace_manager.create_workspace_directory(session_id, "src/nested")
            deleted_dir = workspace_manager.delete_workspace_entry(session_id, "src/nested")
            self.assertEqual(deleted_dir["type"], "directory")
            self.assertFalse((Path(temp_dir) / "src" / "nested").exists())

    def test_command_session_service_runs_inside_bound_workspace(self):
        """页面终端应能在合法目录内执行受控命令。"""
        allowed_parent = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=str(allowed_parent)) as temp_dir:
            snapshot = command_session_service.start_command(
                session_id="sess-terminal",
                workspace_root=temp_dir,
                command_text="python3 -c \"print('terminal harness ok')\"",
            )
            command_id = str(snapshot["command_id"])

            latest = snapshot
            for _ in range(30):
                latest = command_session_service.get_session(command_id)
                if latest and latest.get("status") != "running":
                    break
                time.sleep(0.2)

            self.assertIsNotNone(latest)
            self.assertEqual(latest["status"], "completed")
            self.assertIn("terminal harness ok", latest["output"])
            self.assertEqual(latest["cwd"], temp_dir)

    def test_code_agent_should_only_execute_when_user_explicitly_requests_run(self):
        """代码 Agent 应区分“写代码”和“执行代码”两类请求。"""
        self.assertFalse(_should_execute_request("帮我写一个 Python hello world"))
        self.assertFalse(_should_execute_request("给我一个 Java 冒泡排序示例"))
        self.assertTrue(_should_execute_request("帮我执行这段 Python 代码"))
        self.assertTrue(_should_execute_request("run this script for me"))

    def test_health_runtime_harness_reports_ok(self):
        """健康检查应暴露 Harness 各子系统状态。"""
        result = _check_runtime_harness()
        self.assertEqual(result["status"], "ok")
        self.assertIn("tool_registry", result)
        self.assertIn("workspace_root", result)
        self.assertIn("terminal_runtime", result)


if __name__ == "__main__":
    unittest.main()
