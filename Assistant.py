import os
import json

import streamlit as st
import openai

# configuring openai - api key
working_dir = os.path.dirname(os.path.abspath(__file__))
#config_data = json.load(open(f"{working_dir}/config.json"))
openai.api_key = st.secrets["OPENAI_API_KEY"]

# configuring streamlit page settings
st.set_page_config(
    page_title="Best Assistant",
    page_icon="💬",
    layout="centered"
)

# List of models
models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]

# Create a select box for the models
st.session_state["openai_model"] = st.sidebar.selectbox("Select OpenAI model", models, index=0)
model=st.session_state["openai_model"]



# initialize chat session in streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# streamlit page title
st.title("🤖 Best Assistant")

# display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# input field for user's message
user_prompt = st.chat_input("Ask GPT-4o...")

if user_prompt:
    # add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # send user's message to GPT-4o and get a response
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            *st.session_state.chat_history
        ]
    )

    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # display GPT-4o's response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)