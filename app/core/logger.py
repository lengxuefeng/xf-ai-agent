import logging
import sys


def setup_logger():
    """
    设置应用程序的根日志记录器。
    应在应用程序启动时调用一次。
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
    )
