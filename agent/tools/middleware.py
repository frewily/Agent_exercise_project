from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from typing import Callable
from utils.prompt_loader import load_system_prompts,load_report_prompts
from langgraph.runtime import Runtime
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from utils.logger_handler import logger

@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,  # 请求的数据封装
        handler: Callable[[ToolCallRequest], ToolMessage | Command]  # 工具执行函数
) -> ToolMessage | Command:         # 工具执行的监控
    """
    作用：包装所有工具调用，记录执行日志
    功能：
        记录正在执行的工具名称和参数
        捕获工具执行结果或异常
        特别地，当调用 fill_content_for_report 工具时，会在运行时上下文中标记 report = True，用于后续动态切换提示词
    """
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]工具参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        # 如果为生成报告，则将标记设置为True，下面动态提示词模版进行动态提示词切换
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return  result
    except Exception as e:
        logger.error(f"[tool monitor]工具{request.tool_call['name']}调用失败")
        logger.error(e)
        raise e

@before_model
def log_before_model(
        state: AgentState,  # 整个agent智能体中的状态记录
        runtime: Runtime,  #记录整个执行过程的上下文信息
):         #在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息")

    logger.debug(f"[log_before-model]{type(state['messages'][-1]).__name__} "
                 f"| {state['messages'][-1].content.strip()}")  # 输出消息类型和具体内容
                 
    return None

@dynamic_prompt  # 每一次在生成提示词前，调用此函数
def report_prompt_switch(request: ModelRequest):         #动态切换提示词
    """
    作用：根据上下文动态选择提示词模板
    功能：
        检查运行时上下文中的 report 标记
        如果是报告生成模式，加载报告专用提示词 (load_report_prompts())
        否则加载系统默认提示词 (load_system_prompts())
    """
    is_report = request.runtime.context.get("report", False)  # 是否生成报告(默认false)
    if is_report:
        return load_report_prompts()
    return load_system_prompts()