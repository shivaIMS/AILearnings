from langchain.agents import create_agent
# Import create_agent from LangChain.
from langchain.tools import tool
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
st.title("Welcome to the Test Case Generator")
requirements = st.text_input(
    "Enter the requirement : ",
    placeholder="Type your requirement here...",
    key="requirement_input",
    help="Please provide a clear and concise requirement for the agent to generate test cases.",
    icon="📝",
    label_visibility="visible"
)


model = st.selectbox("Select the model", ["Functional", "Positive", "Negative", "All"])

# voice_input = st.audio_input("Record your voice")

test_cases_count = st.number_input(
    "Enter the number of test cases for the agent:",
    min_value=1,
    step=1,
    key="test_case_count_input",
)

agent = create_agent(
    model="openai:gpt-5.5",
    system_prompt="""
        You are a Senior QA Lead Engineer provides guidance, motivation, and personal development advice.You can also offer support as a family member.
        Please respond to the user's questions in a friendly and supportive manner,
        using the appropriate tone based on the context of the question.
        1. Read the requirement {requirements} and provide the appropriate response based on the selected model {model} and test_cases_count {test_cases_count}.
        2. If the user has provided a voice input, please transcribe it and include it in your response.
        3. Only provide information that is relevant to the user's question and avoid making up facts.
        4. Give the CSV in downloadbale link format with the test cases generated based on the requirement and model selected.
        5. Provide the response in a clear and concise manner, using bullet points or numbered lists where appropriate.
        6.  test_cases_count: {test_cases_count} is the number of test cases to generate based on the requirement and model selected.
        7.  If the user has not provided a voice input, please respond with a message indicating that the voice input is optional and that they can still receive guidance based on their text input.
    """,
)

if st.button("Submit") and requirements:
    with st.spinner("Thinking..."):
        response = agent.invoke(
            {
                "role": "user",
                "messages": f"Please provide guidance based on the requirement: {requirements}, Model: {model}, Test Cases Count: {test_cases_count}",
            }
        )
    st.write(response["messages"][-1].content)
