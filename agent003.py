# This script builds a web app that works like a Senior QA Engineer.
# The user types in a software feature (for example, "login page"), and the AI agent
# writes test cases for that feature.
# The agent's behavior is controlled by a "system prompt", which is a set of
# instructions that tells the model what role to play and what rules to follow.
# Run it with: streamlit run agent003.py

from langchain.agents import create_agent
# Import the create_agent function from LangChain.
# This function builds an AI agent that can receive messages and respond to them.

import streamlit as st
# Import Streamlit, which turns this Python script into a simple web app.

from dotenv import load_dotenv
# Import load_dotenv from the python-dotenv package.
# This is used to load secret values, such as API keys, from a .env file
# instead of writing them directly inside the code.

load_dotenv()
# Step 1: Load the environment variables from the .env file.
# This makes the OPENAI_API_KEY available to LangChain without writing it in the code.


# Step 2: Create the agent.
# model: the language model the agent uses to think and write answers (here "gpt-4").
# system_prompt: the instructions the agent always follows, no matter what the user asks.
# The system prompt below does four things:
#   - Gives the agent a role: a Senior QA Engineer who writes test cases.
#   - Lists guidelines: cover edge cases, keep test cases clear, prioritize critical features.
#   - Defines the format: every test case must have an ID, description, steps, expected result, etc.
#   - Sets limits: the agent should not answer questions that are not about test cases.
# Writing clear instructions like this is called "prompt engineering".
agent=create_agent(model="gpt-4",
                   system_prompt="""

                   You are Senior QA Engineer specializing in generating comprehensive test cases for software features.

                   Follow these guidelines while generating test cases:
                   1. Cover all possible scenarios including edge cases.
                   2. Include positive test cases.
                   3. Ensure test cases are clear and concise.
                   4. Prioritize test cases based on critical functionality.
                   5. Include setup and teardown steps if necessary.
                   6. Do not create more test cases than necessary.

                   Each test case should have below details:
                   - Test case ID
                   - Test case description
                   - Preconditions
                   - Test steps
                   - Test data
                   - Expected result
                   - Actual result
                   - Status (Pass/Fail)


                   Rules:
                   1. Each test case should be independent.
                   2. Test cases should be repeatable.
                   3. Avoid duplication of test cases.
                   4. Use meaningful test data.
                   5. Clearly specify expected results.

                   Notes:
                   - Ensure test cases are written in a consistent format.
                   - Review test cases for completeness and accuracy.
                   - Update test cases as the software evolves.
                   - Do not answer any questions outside the scope of test case generation.

                   """)

# Step 3: Build the web page.
# Show a title at the top of the page.
st.title("QA Test Case Generator")

# Show example questions so the user knows what they can ask.
st.markdown("""
**You can ask things like:**
- Generate test cases for payment feature using Credit Card
- Generate test cases for a login page with email and password
- Generate test cases for a file upload feature
""")

# Step 4: Add a text box where the user types which feature needs test cases.
# The placeholder is a grey hint shown inside the empty box.
question = st.text_input("Your question", placeholder="e.g. Generate test cases for a login page")

# Step 5: Run the agent when the user clicks "Ask".
# The code inside only runs if the button is clicked AND the text box is not empty.
if st.button("Ask") and question:

    # Show a "Thinking..." spinner while waiting for the AI to reply.
    with st.spinner("Thinking..."):

        # Send the user's question to the agent.
        # Messages are passed as a list; "role": "user" means this message comes from the user.
        # The agent combines it with the system prompt and generates the test cases.
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # Step 6: Show the answer.
    # result["messages"] holds the whole conversation; [-1] picks the last message,
    # which is the agent's final reply. ".content" is the text of that reply.
    st.write(result["messages"][-1].content)
