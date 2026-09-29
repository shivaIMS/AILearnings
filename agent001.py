# This script creates a simple AI agent using LangChain and asks it a question.
# An agent is a program that understands a request, decides what to do,
# and uses a language model to generate a response.

from langchain.agents import create_agent
# Import the create_agent function from LangChain.
# This function is used to build an AI agent that can receive messages and respond to them.

from dotenv import load_dotenv
# Import load_dotenv from the python-dotenv package.
# This is used to load secret values, such as API keys, from a .env file
# instead of writing them directly inside the code.

import streamlit as st
# Import Streamlit, which turns this Python script into a simple web app.

load_dotenv()
# Load the environment variables defined in the .env file into this program.
# For example, this makes an API key like OPENAI_API_KEY available to LangChain
# without typing it directly in the script.

agent = create_agent(model="openai:gpt-4") #openai:gpt-5.5 gpt-4

# Create the agent and specify which language model it should use.
# Here, the agent is set up to use OpenAI's "gpt-4" model to understand
# requests and generate answers.

# Build a simple web page with Streamlit.
# Run it with: streamlit run agent001.py

st.title("Ask Me Anything")

st.markdown("""
**You can ask things like:**
- What is Playwright?
- Explain REST vs GraphQL
- How do I write a good bug report?
""")

# Show example questions so the user knows what they can ask.

question = st.text_input("Your question", placeholder="e.g. What is Playwright?")

# Text box where the user types a question; the placeholder is a hint.

if st.button("Ask") and question:

    with st.spinner("Thinking..."):

        result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # Send the question to the agent (same as before, but using the user's text)
    # and show only the final answer on the page.
    st.write(result["messages"][-1].content)
