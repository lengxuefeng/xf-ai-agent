import glob
import logging
import os
import threading
import uuid
from typing import TypedDict, List, Callable, Optional, Union

import yaml
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain.messages import SystemMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from utils.custom_logger import get_logger, LogTarget

# 配置日志记录器
log = get_logger(__name__)

# 定义技能结构
class Skill(TypedDict):
    name: str
    description: str
    content: str


class SkillLoader:
    """负责从文件系统中加载和解析技能文件"""

    @staticmethod
    def parse_skill_file(file_path: str) -> Optional[Skill]:
        """解析带有 YAML Front Matter 的 Markdown 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.startswith('---'):
                log.warning(f"SKILL_WARNING: {file_path} 不以YAML前端内容（---）开头。跳过.")
                return None

            parts = content.split('---', 2)
            if len(parts) < 3:
                log.warning(f"SKILL_WARNING: {file_path} 格式错误。需要YAML前端内容（---）包裹。")
                return None

            yaml_content = parts[1]
            markdown_content = parts[2].strip()

            metadata = yaml.safe_load(yaml_content)

            if not metadata or 'name' not in metadata or 'description' not in metadata:
                log.warning(f"SKILL_WARNING: {file_path} YAML前端内容（---）中缺少'name'或'description'字段。")
                return None

            return {
                "name": metadata['name'].strip(),
                "description": metadata['description'].strip(),
                "content": markdown_content
            }
        except Exception as e:
            log.error(f"SKILL_ERROR: 加载技能 {file_path} 时出错: {e}")
            return None

    @staticmethod
    def load_from_dir(directory: str) -> List[Skill]:
        """加载目录下所有 SKILL_*.md 文件"""
        skills = []
        # 确保路径存在
        if not os.path.exists(directory):
            log.error(f"SKILL_ERROR: 目录不存在: {directory}")
            return []

        search_pattern = os.path.join(directory, "SKILL_*.md")
        files = glob.glob(search_pattern)

        if not files:
            log.warning(f"SKILL_INFO: 目录 {directory} 中未找到匹配模式 SKILL_*.md 的技能文件")
            return []

        for file_path in files:
            skill = SkillLoader.parse_skill_file(file_path)
            if skill:
                skills.append(skill)
                log.info(f"SKILL_SUCCESS: 已加载技能 '{skill['name']}'", target=LogTarget.ALL)

        return skills


class ConfigurableSkillMiddleware(AgentMiddleware):
    """
    可配置的技能中间件。
    支持直接传入文件夹路径，自动处理加载逻辑。
    """

    _dir_skill_cache: dict[str, List[Skill]] = {}
    _cache_lock = threading.RLock()

    def __init__(self, skill_source: Union[str, List[Skill]]):
        """
        初始化中间件。

        Args:
            skill_source:
                - str: 技能文件夹路径，会自动加载。
                - List[Skill]: 已经加载好的技能列表。
        """
        self.skills: List[Skill] = []

        # 核心逻辑：根据传入参数类型决定如何加载
        if isinstance(skill_source, str):
            with self._cache_lock:
                cached_skills = self._dir_skill_cache.get(skill_source)
            if cached_skills is not None:
                self.skills = [dict(item) for item in cached_skills]
                log.info(f"MIDDLEWARE: 命中技能缓存: {skill_source}", target=LogTarget.ALL)
            else:
                log.info(f"MIDDLEWARE: 正在从目录初始化技能: {skill_source}", target=LogTarget.ALL)
                loaded_skills = SkillLoader.load_from_dir(skill_source)
                with self._cache_lock:
                    # 缓存副本，避免外部误修改共享对象
                    self._dir_skill_cache[skill_source] = [dict(item) for item in loaded_skills]
                self.skills = loaded_skills
            log.info(f"✅ 云柚代理工具加载成功: {len(self.skills)} 个工具", target=LogTarget.ALL)
        elif isinstance(skill_source, list):
            self.skills = skill_source
        else:
            raise ValueError("skill_source 必须是文件夹路径(str)或技能列表(List[Skill])")

        self.skills_prompt = self._build_skills_prompt()

        # 动态创建绑定了当前 skills 的 tool
        self.tools = [self._create_load_skill_tool()]

    def _build_skills_prompt(self) -> str:
        """生成系统提示词中的技能列表部分"""
        if not self.skills:
            log.warning("MIDDLEWARE: 未配置任何技能，Agent 将无法加载具体逻辑。")
            return ""

        skills_list = []
        for skill in self.skills:
            skills_list.append(
                f"- **{skill['name']}**: {skill['description']}"
            )
        return "\n".join(skills_list)

    def _load_skill(self, skill_name: str) -> str:
        """将技能的完整内容加载到代理（Agent）的上下文中。"""
        for skill in self.skills:
            if skill["name"] == skill_name:
                log.info(f"TOOL_USE: Agent 正在加载技能 '{skill_name}'")
                return f"已加载技能: {skill_name}\n\n{skill['content']}"

        available = ", ".join(skill["name"] for skill in self.skills)
        return f"未找到技能 '{skill_name}'。当前可用技能: {available}"

    def _create_load_skill_tool(self):
        """创建技能加载 Tool。"""
        return tool(
            "load_skill",
            description=(
                "将技能的完整内容加载到代理（Agent）的上下文中。"
                "当你需要处理特定类型请求的详细信息时使用此工具。"
            ),
        )(self._load_skill)

    def get_tools(self) -> List[Callable]:
        """提供给外部 Agent 使用的工具列表。"""
        return self.tools

    def get_prompt(self) -> str:
        """生成可直接用于 system prompt 的技能说明片段。"""
        if not self.skills:
            return "当前未配置扩展技能。"

        return (
            "你可以按需调用 `load_skill` 工具加载具体业务规则。\\n"
            "可用技能如下:\\n"
            f"{self.skills_prompt}"
        )

    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """注入技能描述到 System Prompt"""
        if not self.skills:
            return handler(request)

        skills_addendum = (
            f"\n\n## 可用技能 (Available Skills)\n\n{self.skills_prompt}\n\n"
            "当你需要处理特定类型请求的详细信息时，请务必调用 load_skill 工具获取详细逻辑。"
        )

        # 兼容处理 content_blocks 和 string
        if hasattr(request.system_message, "content_blocks"):
            new_content = list(request.system_message.content_blocks) + [
                {"type": "text", "text": skills_addendum}
            ]
            new_system_message = SystemMessage(content=new_content)
        else:
            original_content = request.system_message.content
            if isinstance(original_content, str):
                new_content = original_content + skills_addendum
            else:
                new_content = str(original_content or "") + skills_addendum
            new_system_message = SystemMessage(content=new_content)

        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)


if __name__ == '__main__':
    # 1. 开启日志显示 (如果不加这行，log.info 不会打印)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    demo_api_key = os.getenv("SKILL_DEMO_API_KEY", "").strip()
    if not demo_api_key:
        raise RuntimeError("请先设置环境变量 SKILL_DEMO_API_KEY 再运行本地示例。")

    model = ChatOpenAI(
        model="GLM-4.6",  # 请确认你的模型名称正确
        temperature=0.1,  # 稍微调低温度，让工具调用更稳定
        api_key=demo_api_key,
        base_url=os.getenv("SKILL_DEMO_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4"),
    )

    # 技能文件路径
    SKILL_DIR = "/Users/lengxuefeng/Documents/xuefeng/ai/gemini-cli/xf-ai-agent/skills/yunyou"

    # 2. 创建支持技能的代理
    # 注意：这里直接传入了 SKILL_DIR 字符串
    agent = create_agent(
        model,
        system_prompt=(
            "你是一个 SQL 查询助手，帮助用户编写针对业务数据库的查询语句。"
        ),
        # 直接传路径，Middleware 会自己去加载
        middleware=[ConfigurableSkillMiddleware(SKILL_DIR)],
        checkpointer=InMemorySaver(),
    )

    # 配置此对话线程
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\n------------------ 开始对话 ------------------")

    # 请求编写 SQL 查询
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "写一个 SQL 查询，找出今天holter2小时的记录条数"
                    ),
                }
            ]
        },
        config
    )

    print("\n------------------ 对话结束 ------------------")
    # 打印对话内容
    for message in result["messages"]:
        if hasattr(message, 'pretty_print'):
            message.pretty_print()
        else:
            print(f"[{message.type}]: {message.content}")
