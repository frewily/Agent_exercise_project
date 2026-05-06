import os
import random

from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

rag = RagSummarizeService()

user_ids = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]

external_data = {}

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)

@tool(description="获取指定城市天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    return f"城市{city}的天气是晴天，气温26摄氏度，南风3级，最近一周，多云"

@tool(description="获取用户所在城市名称，以纯字符串的形式返回·")
def get_user_city() -> str:
    return random.choice(["北京", "上海", "广州", "深圳", "杭州"])

@tool(description="获取用户ID，以纯数字的形式返回")
def get_user_id() -> int:
    return random.choice(user_ids)

@tool(description="获取指定月份的日历，以纯字符串的形式返回")
def get_calendar(month: str) -> str:
    return random.choice(month_arr)

@tool(description="获取系统当前月份，格式为YYYY-MM，无入参")
def get_current_month() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m")

def generate_external_data():
    global external_data
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"{external_data_path} not found")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:  # 跳过第一行
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")  #替换双引号为空字符串
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time]= {
                    "feature": feature,
                    "efficiency": efficiency,
                    "consumables": consumables,
                    "comparison": comparison,
                }

@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串的形式返回，如果未检索到则返回空字符串")
def fetch_external_data(user_id: int, month: str) -> str:
    generate_external_data()

    try:
        return external_data[str(user_id)][month]
    except KeyError:
        logger.info(f"未找到用户{user_id}在{month}的记录")
        return ""

@tool(description="无入参，无返回值，调用后触发中间件为报告自动生成的场景注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"

if __name__ == '__main__':
    print(fetch_external_data("1", "2025-01"))