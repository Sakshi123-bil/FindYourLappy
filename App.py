from dotenv import load_dotenv
import streamlit as st
import os
from langchain.llms.bedrock import Bedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import speech_recognition as sr

# Environment Setup
load_dotenv()  # Load environment variables

# Add your OpenAI API key here
openai_api_key = os.getenv("OPEN_AI_KEY")

# Streamlit Setup
custom_css = """
    <style>
        .title {
            font-size: 30px !important;
        }
        .chat-history {
            overflow-y: scroll;
            height: 200px;
            border: 1px solid #ddd;
            padding: 5px;
        }
        .clear-button {
            margin-top: 10px;
            margin-left: 10px;
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown("#Find Your Dream Laptop")

# Database Setup
singlestore_uri = 'mysql+mysqlconnector://admin:Sakshi123@svc-49a33054-45f9-4bdd-a096-9692d8679ff0-dml.aws-oregon-4.svc.singlestore.com:3306/TestData'
db = SQLDatabase.from_uri(singlestore_uri)

# Prompt Templates
schema_template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(schema_template)

response_template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:

{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_template(response_template)

# Query Execution
def get_schema(_):
    return db.get_table_info()

def run_query(query):
    return db.run(query)

# Pipeline Setup
llm = ChatOpenAI(api_key=openai_api_key)  # Use your OpenAI API key here

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
        schema=get_schema,
        response=lambda vars: run_query(vars["query"]),
    )
    | prompt_response
    | llm
    | StrOutputParser()
)

# Streamlit UI
class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

session_state = SessionState(chat_history=[])

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Say something...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.write("Could not understand audio")
        return ""
    except sr.RequestError as e:
        st.write(f"Error: {e}")
        return ""

st.sidebar.markdown("## Input Options")
input_option = st.sidebar.radio("Select Input Option:", ["Text Input", "Voice Input"])

if input_option == "Text Input":
    user_question = st.text_input("Enter your question:")
    if st.button("Search"):
        response = full_chain.invoke({"question": user_question})
        session_state.chat_history.append({"question": user_question, "response": response})
        st.write("Response:", response)

elif input_option == "Voice Input":
    if st.button("Start Recording"):
        user_question = recognize_speech()
        st.text_input("Voice Input:", user_question)
        response = full_chain.invoke({"question": user_question})
        session_state.chat_history.append({"question": user_question, "response": response})
        st.write("Response:", response)

# Display chat history and clear chat history button
st.sidebar.markdown("## Chat History")
chat_history_container = st.sidebar.empty()
chat_history_container.markdown("No chat history yet.")
clear_button_clicked = st.sidebar.button("Clear Chat History")

if clear_button_clicked:
    session_state.chat_history = []

if session_state.chat_history:
    chat_history_container.markdown("## Chat History")
    for chat in session_state.chat_history:
        chat_history_container.write(f"Question: {chat['question']}")
        chat_history_container.write(f"Response: {chat['response']}")