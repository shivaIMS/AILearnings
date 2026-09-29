from langsmith import create_agent
from dotenv import load_dotenv
from langsmith import tool
from streamlit import text_input

load_dotenv()

requirement = text_input(
    "Enter the requirement for the agent:",
    key="requirement_input",
    placeholder="Type your requirement here...",
)

Test_case_type = text_input(
    "Enter the test case type for the agent: types should be Functional,Positive, Negative, Boundary All"
)

test_case_count = int(
    text_input(
        "Enter the number of test cases for the agent:",
        key="test_case_count_input",
        placeholder="Type the number of test cases here... ",
    )
)

agent = create_agent(
    model="openai:gpt-5.5",
    requirement=requirement,
    test_case_type=Test_case_type,
    test_case_count=test_case_count,
)

agent.invoke(
    {
        "role": "user",
        "messages": f"Please generate {test_case_count} {Test_case_type} test cases for the requirement: {requirement}",
    }
)
