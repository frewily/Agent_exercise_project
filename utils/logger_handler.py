import logging
import os
from utils.path_tool import get_abs_path
from datetime import datetime

# 日志文件路径
LOG_ROOT = get_abs_path('logs')

# 确保日志的目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志的格式配置
DEFAULT_LOG_FORMAT = logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] %(name)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_logger(
        name: str = "agent",
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        log_file = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加handlers
    if logger.hasHandlers():
        return logger

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)

    # 文件handler
    if log_file is None:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger

# 快捷获取日志器
logger = get_logger()

if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.debug("调试日志")
    logger.warning("警告日志")