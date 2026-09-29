import streamlit as st
from langsmith import create_agent
from dotenv import load_dotenv
from langsmith import tool

load_dotenv()

st.title("Welcome to the First UI")
st.write("This is a simple Streamlit application.")

agent = create_agent(model="openai:gpt-5.5", )

question = st.text_input(
    "Ask me anything:",
    key="question_input",
    placeholder="Type your question here...",
)

if st.button("Submit"):
    if question.strip():
        st.write(f"You asked: {question}")
        st.info(f"A response to '{question}' will be generated here.")
        st.balloons()
    else:
        st.warning("Please enter a question before submitting.")

st.success("Thanks for trying this Streamlit app! 🎉")