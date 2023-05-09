from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# from ..config import Config
# import os

# os.environ['OPENAI_API_KEY'] = Config.open_ai_key
# os.environ['SERPER_API_KEY'] = Config.serper_api_key

llm = OpenAI(temperature=0)
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search"
    )
]

self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
self_ask_with_search.run("what is the tallest building in the world second largest economy?")