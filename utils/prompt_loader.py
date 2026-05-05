from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_conf["system_prompt_path"])
    except KeyError as e:
        logger.error(f"{e}")
        raise e

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except FileNotFoundError as e:
        logger.error(f"{e}")
        raise e

def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_prompt_path"])
    except KeyError as e:
        logger.error(f"{e}")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except FileNotFoundError as e:
        logger.error(f"{e}")
        raise e

def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"{e}")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except FileNotFoundError as e:
        logger.error(f"{e}")
        raise e