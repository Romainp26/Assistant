import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
import pandas as pd
import PyPDF2
from PIL import Image
import easyocr
import numpy as np
import matplotlib.pyplot as plt


# configuring openai - api key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# List of models
models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]

# Create a select box for the models
st.session_state["openai_model"] = st.sidebar.selectbox("Select OpenAI model", models, index=0)
model=st.session_state["openai_model"]

# Initialize LangChain OpenAI Client
openai_chat = ChatOpenAI(openai_api_key=openai_api_key, model_name=model)


# Streamlit Title
st.title("🤖 Best Assistant")

# System Prompt
system_prompt = """
You are a helpful assistant.
"""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)

# Function to process CSV
def process_csv(file):
    try:
        data = pd.read_csv(file)
        summary = data.describe(include="all").to_string()
        return f"CSV Summary:\n\n{summary}", data
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None, None

# Function to process PDF
def process_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return f"PDF Content:\n\n{text[:2000]}"  # Limiter le contenu à 2000 caractères
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

# Function to process Image using EasyOCR
def process_image(file):
    try:
        # Open the image from the uploaded file
        image = Image.open(file)
        
        # Convert image to numpy array
        image_np = np.array(image)

        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])  # Add 'fr' for French support if needed

        # Perform OCR on the numpy array
        results = reader.readtext(image_np)

        # Extract text from results
        extracted_text = "\n".join([result[1] for result in results])
        return f"Extracted Text from Image:\n\n{extracted_text[:2000]}"  # Limit to 2000 characters
    except Exception as e:
        return f"Error processing image: {e}"



# File Uploader
uploaded_file = st.file_uploader("Upload a file (CSV, PDF, or Image)", type=["csv", "pdf", "png", "jpg", "jpeg"])
file_context = ""

if uploaded_file:
    # Determine file type and process
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "csv":
        file_context, data = process_csv(uploaded_file)
        st.dataframe(data)
    elif file_type == "pdf":
        file_context = process_pdf(uploaded_file)
        data = None
    elif file_type in ["png", "jpg", "jpeg"]:
        file_context = process_image(uploaded_file)
        data = None
    else:
        st.error("Unsupported file type!")
else:
    data = None


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input and Chatbot Response
if prompt := st.chat_input("Ask a question about the uploaded file:"):
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar='👨🏻‍💻'):
        st.markdown(prompt)

    # Define prompt with file context if available
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt + "\n" + file_context),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    # Create LangChain conversation
    conversation = LLMChain(
        llm=openai_chat,
        prompt=chat_prompt,
        verbose=False,
        memory=st.session_state.memory,
    )



    # Generate response
    response = conversation.predict(human_input=prompt)

    # Append assistant message to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant",avatar='🤖'):
        st.markdown(response)