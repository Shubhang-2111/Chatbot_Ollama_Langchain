import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.llms import Ollama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

# Prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assitant. Please respond to the question asked"),
        ("user","Question:{question}"),
    ]
)


# streamlit 
st.title("Chatbot using Gemma Model")
st.write("This chatbot is built using the Gemma model from LangChain")


if "conversation" not in st.session_state:
    st.session_state.conversation = []


input_text = st.text_input("What question u have in mind ?")

## Ollama model
llm = Ollama(model="gemma2:2b")
output_parser = StrOutputParser()

chain = prompt|llm|output_parser

# The | operator represents a flow where the output of 
# one component becomes the input for the next.

if input_text:
    # Add user input to conversation history
    st.session_state.conversation.append({"role": "user", "content": input_text})

    # Get response from the model
    response = chain.invoke({"question": input_text})
    
    # Add assistant's response to the conversation history
    st.session_state.conversation.append({"role": "assistant", "content": response})

# Display conversation history
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Assistant:** {message['content']}")