import logging
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Any, Callable

logger = logging.getLogger(__name__)


class ConcurrentExecutor:
    """封装线程并发执行逻辑，供多个 LangGraph 子图复用"""

    def __init__(self, max_threads: int = 4):
        """
        初始化并发执行器

        Args:
            max_threads: 最大线程数
        """
        self.max_threads = max_threads
        logger.info(f"ConcurrentExecutor 初始化，max_threads={max_threads}")

    def run_concurrent(self, items: List[Any], func: Callable[[Any], Any]) -> List[Any]:
        """
        并发执行任务

        Args:
            items: 输入项列表（如 LangGraph 状态列表）
            func: 要执行的函数（如 CodeAgent.run）

        Returns:
            List[Any]: 每个输入的执行结果
        """
        try:
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                results = list(executor.map(func, items))
            logger.info(f"并发执行完成，处理 {len(items)} 个任务")
            return results
        except Exception as e:
            logger.error(f"并发执行失败: {str(e)}")
            raise


concurrent_executor = ConcurrentExecutor()
