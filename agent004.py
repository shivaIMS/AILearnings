# This script builds a web app that finds the latest news on any topic and summarizes it.
# It introduces "tools": normal Python functions that the AI agent is allowed to call.
# The agent reads the user's question, decides which tool to use, calls it,
# looks at the result, and then decides what to do next.
# Two tools are used here:
#   1. get_news       - searches the internet for recent news using Tavily (a search API for AI apps).
#   2. summarize_news - asks a language model to turn those articles into short bullet points.
# Run it with: streamlit run agent004.py

from langchain.agents import create_agent
# Import the create_agent function from LangChain.
# This function builds an AI agent that can receive messages, use tools, and respond.

from dotenv import load_dotenv
# Import load_dotenv to read secret values (API keys) from a .env file.

from tavily import TavilyClient
# Import the Tavily client. Tavily is a web search service built for AI agents.
# We use it to search for the latest news articles.

from langchain.tools import tool
# Import the @tool decorator. Putting @tool above a function turns it into a tool
# that the agent can call on its own.

from langchain.chat_models import init_chat_model
# Import init_chat_model, a simple way to create a language model object
# that we can send a prompt to and get a reply from.

import streamlit as st
# Import Streamlit, which turns this Python script into a simple web app.

import os
# Import os to read environment variables such as TAVILY_API_KEY.

load_dotenv()
# Step 1: Load the API keys from the .env file.
# This file should contain OPENAI_API_KEY and TAVILY_API_KEY.

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
# Step 2: Create the Tavily client using the Tavily API key.
# os.getenv reads the key that load_dotenv loaded from the .env file.


# Step 3: Create the first tool: get_news.
# The @tool decorator makes this function available to the agent.
# The text inside """ """ (the docstring) is very important: the agent reads it
# to understand what the tool does and when it should use it.
# The type hint "topic: str" tells the agent it must pass a text value.
@tool
def get_news(topic: str):

    """Use this tool to get the latest news articles related to the specified topic."""

    # Search Tavily for news only (topic="news"), using the user's topic as the search query.
    # max_results=5 limits the search to the top 5 articles.
    response=tavily_client.search(topic="news",query=topic,max_results=5)

    # Tavily returns a dictionary. The "results" key holds the list of articles
    # (each article has a title, URL, and content). We return only that list.
    return response["results"]



# Step 4: Create the second tool: summarize_news.
# This tool takes the list of articles from get_news and returns a short summary as text.
@tool
def summarize_news(articles: list) -> str:

    """Use this tool to summarize a list of news articles into a single text."""

    # Build the prompt (instructions) for the language model.
    # The f-string puts the articles directly into the prompt text.
    # "Only use information from the articles" stops the model from making up facts.
    prompt = f"""
    Summarize the news articles below as short bullet points.
    For each story, give the key facts in 1-2 sentences and include its source URL.
    Only use information from the articles.

    Articles:
    {articles}
    """

    # Create a language model to write the summary.
    # "openai:gpt-5.5" means: use the OpenAI provider and the gpt-5.5 model.
    model = init_chat_model("openai:gpt-5.5")

    # Send the prompt to the model and return only the text of its reply.
    return model.invoke(prompt).content


# Step 5: Create the agent and give it the tools.
# model: the language model that does the thinking and decides which tool to call.
# tools: the list of tools the agent is allowed to use.
# system_prompt: instructions that tell the agent the order to use the tools in
# and remind it not to make up news.
agent = create_agent(
                    model="openai:gpt-5.5",

                    tools=[get_news, summarize_news],

                    system_prompt="""
                    You are a news assistant.

                    1. Use get_news to fetch the latest articles for the user's topic.
                    2. Pass those articles to summarize_news.
                    3. Reply with a short bulleted summary of the key stories, each with its source URL.
                    Only report what the articles say; do not make up news.
                    """
                    )


# Step 6: Build the web page.
# Show a title at the top of the page.
st.title("Latest News Summarizer")

# Show example questions so the user knows what they can ask.
st.markdown("""
**You can ask things like:**
- What is the latest news on AI agents?
- Latest news on the stock market
- What is happening in cricket today?
""")

# Step 7: Add a text box where the user types a news topic.
question = st.text_input("Your question", placeholder="e.g. Latest news on AI agents")

# Step 8: Run the agent when the user clicks "Ask" and the text box is not empty.
if st.button("Ask") and question:

    # Show a "Thinking..." spinner while the agent works.
    with st.spinner("Thinking..."):

        # Send the question to the agent. Behind the scenes the agent:
        #   a) calls get_news to search for articles,
        #   b) calls summarize_news with those articles,
        #   c) writes the final answer for the user.
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # Step 9: Show the answer.
    # result["messages"] holds every step (user question, tool calls, tool results, final reply).
    # [-1] picks the last message, which is the agent's final answer.
    st.write(result["messages"][-1].content)
