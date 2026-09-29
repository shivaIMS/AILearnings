from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

requirement = input("Enter the requirement: ")

test_case_type = input("Enter the test case type: ")

test_case_count = int(input("Enter the number of test cases: "))

agent = create_agent(
    model="openai:gpt-5.5",
    system_prompt="""
You are a Senior QA Engineer. Generate the requested number of clear, executable
test cases based only on the user's requirement. Focus on the requested test type.
Return the test cases as a CSV-formatted response with columns for ID, Title,
Test Type, Preconditions, Steps, Test Data, Expected Result, and Priority.
Do not claim to provide a downloadable file; return the CSV text directly.
""",
)

user_prompt = (
    f"Generate exactly {test_case_count} {test_case_type} test cases "
    f"for this requirement: {requirement}"
)
response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})

print(response["messages"][-1].content)
