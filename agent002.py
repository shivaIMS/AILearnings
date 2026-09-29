from langchain.agents import create_agent
# Import create_agent from LangChain.
# This function builds an AI agent that can read a user's message, think about
# it, and decide how to respond - including deciding whether to use one of the
# tools we give it below.

from langchain.tools import tool
import streamlit as st
# Import the "tool" decorator from LangChain.
# A tool is a small Python function that the agent is allowed to call when it
# needs to. The @tool decorator turns a normal function into something the
# agent can discover and use on its own.

from dotenv import load_dotenv
# Import load_dotenv from the python-dotenv package.
# This lets us load secret values, such as API keys, from a .env file instead
# of writing them directly inside the code.

load_dotenv()
# Load the environment variables defined in the .env file into this program.
# For example, this makes an API key like OPENAI_API_KEY available to
# LangChain without typing it directly in the script.


@tool
def spiritual_advice(message):
    """
        use this tool for providing spiritual advice and guidance, meditation, and mindfulness practices.
    """
    # This function is a "tool" the agent can choose to call.
    # The text inside the triple quotes (the docstring) is not just a comment -
    # the agent actually reads it to decide when this tool should be used.
    # Here, it tells the agent: use this tool when the user's message is about
    # spirituality, meditation, or mindfulness.
    return f"Answer this like a spiritual teacher: {message}"
    # This builds and returns a new instruction string that tells the language
    # model to answer in the style of a spiritual teacher, using the user's
    # original message as the topic.

@tool
def friend_suggestion(name):
    """
        use this tool for emotional support,encouragement, casual conversation, and friendly advice.
    """
    # Another tool the agent can pick. Its docstring tells the agent to use
    # this one when the user seems to want emotional support or a casual,
    # friendly chat rather than formal advice.
    return f"Answer this like a friend: {name}"
    # Returns an instruction telling the language model to reply the way a
    # friend would, using whatever text was passed in as "name".

@tool
def life_coach(message):
    """
        use this tool for providing guidance, motivation, and personal development advice, producvity , goals, career planning.


    """
    # This tool's docstring tells the agent to use it for motivation,
    # productivity, goal-setting, and career-related questions.
    return f"Answer this like a life coach: {message}"
    # Returns an instruction telling the language model to reply in the style
    # of a life coach, based on the user's message.

@tool
def family_member_advice(message):
    """
        use this tool for providing advice and support as a family member.

    """
    # This tool's docstring tells the agent to use it when the user wants
    # advice that sounds like it is coming from a family member.
    return f"Answer this like a family member: {message}"
    # Returns an instruction telling the language model to reply in the style
    # of a family member, based on the user's message.


agent = create_agent(
                    model="openai:gpt-5.5",

                    tools=[spiritual_advice, friend_suggestion, life_coach, family_member_advice]


                     )
# Create the agent and configure it with two things:
# 1. model - which language model the agent should use to think and respond
#    (here, OpenAI's "gpt-5.5" model).
# 2. tools - the list of tool functions defined above that the agent is
#    allowed to choose from. When the agent receives a message, it looks at
#    each tool's docstring and decides which tool (if any) best matches what
#    the user is asking for.


st.title("Life Advice Assistant")

st.markdown("""
**You can ask things like:**
- I feel stressed at work, any advice? (friend)
- How can I start meditating? (spiritual teacher)
- How do I plan my career for the next 5 years? (life coach)
- How do I handle a disagreement with my parents? (family member)
""")

question = st.text_input("Your question", placeholder="e.g. How can I be more productive?")

if st.button("Ask") and question:

    with st.spinner("Thinking..."):

        result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    st.write(result["messages"][-1].content)
