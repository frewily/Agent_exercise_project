from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize,get_weather,get_user_city,get_user_id,
                                     get_calendar,fetch_external_data,fill_context_for_report,get_current_month)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            tools=[
                rag_summarize,
                get_weather,
                get_user_city,
                get_user_id,
                get_calendar,
                fetch_external_data,
                fill_context_for_report,
                get_current_month
            ],
            system_prompt=load_system_prompts(),
            middleware=[
                monitor_tool,
                log_before_model,
                report_prompt_switch
            ],
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages":[
                {
                    "role": "user",
                    "content": query
                }
            ]
        }

        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"

if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("扫地机器人在我所在的地区气温下如何保养"):
        print(chunk, end="", flush=True)