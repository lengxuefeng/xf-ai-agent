import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FileUtils:

    @staticmethod
    def get_project_root() -> Path:
        """
        从脚本位置向上查找，直到找到项目根目录（此处以包含'.git'文件夹为标志）。
        您可以根据自己项目的特点更改标志，例如 'pyproject.toml' 文件。
        """
        # 获取当前文件（file_utils.py）的绝对路径
        current_file_path = Path(__file__).resolve()
        # 从文件所在的目录开始向上查找
        parent_dir = current_file_path.parent

        # 循环向上查找，直到找到包含 .git 目录的父级目录
        while parent_dir != parent_dir.parent:  # 循环直到根目录 '/'
            if (parent_dir / '.git').exists() or (parent_dir / 'pyproject.toml').exists():
                return parent_dir
            parent_dir = parent_dir.parent

        # 如果找不到标志，则抛出异常，或返回一个默认值
        raise FileNotFoundError("无法自动定位项目根目录。请确保项目中包含 .git 目录或 pyproject.toml 文件。")

    @staticmethod
    def read_project_file(relative_file_path: str) -> str | None:
        """
        安全地读取项目内指定相对路径的文件内容，并保持其原始格式。

        Args:
            relative_file_path: 从项目根目录算起的相对文件路径。
                               例如: 'app/agent/template/yunyou_agent_cue_word.txt'

        Returns:
            如果文件成功读取，返回文件的完整内容 (字符串)。
            如果文件不存在或发生错误，返回 None。
        """
        try:
            # 1. 自动获取项目根目录
            project_root = FileUtils.get_project_root()

            # 2. 将项目根目录和相对路径拼接成一个完整的绝对路径
            full_path = project_root / relative_file_path

            logger.info(f"正在尝试读取文件: {full_path}")

            # 3. 检查文件是否存在
            if not full_path.is_file():
                logger.error(f"错误：文件不存在于路径 '{full_path}'")
                return None

            # 4. 读取文件内容
            content = full_path.read_text(encoding='utf-8')
            return content

        except FileNotFoundError as e:
            print(e)
            return None
        except Exception as e:
            print(f"读取文件时发生未知错误: {e}")
            return None


# --- 如何在其他文件中调用这个函数 ---
if __name__ == '__main__':
    # 这是一个使用示例，您可以将这个逻辑放到您需要读取文件的任何地方

    # 您只需要提供从项目根目录 'xf-ai-agent/' 开始的路径即可
    path_from_root = 'app/agent/template/yunyou_agent_cue_word.txt'

    file_content = FileUtils.read_project_file(path_from_root)

    if file_content is not None:
        print("\n--- 文件内容 ---")
        print(file_content)
        print("--- 内容结束 ---\n")
    else:
        print("\n文件读取失败。")
