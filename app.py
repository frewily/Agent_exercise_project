import time

import streamlit as st
from agent.react_agent import ReactAgent

# 标题
st.title("智能扫地机器人客服")
st.divider()

if "agent" not in st.session_state:
    st.session_state.agent = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# 用户输入提示词
prompt = st.chat_input("请输入您的问题")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append(
        {
            "role": "user",
            "content": prompt
        }
    )

    response_messages = []
    with st.spinner("思考中"):
        res_stream = st.session_state["agent"].execute_stream(prompt)

        def capture(generator, cache_list):

            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        if response_messages:
            st.session_state["message"].append(
                {
                    "role": "assistant",
                    "content": response_messages[-1]
                }
            )