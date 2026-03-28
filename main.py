from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that searches weather over the internet.
    Args:
        query: The query to search for.
    Returns:
        The search result.
    """
    print(f"Searching for: {query}")
    return tavily.search(query)


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Running the agent...")
    result = agent.invoke(
        {"messages": HumanMessage(
            content="Search 3 job postings for an ai engineer in Georgia, California and Washington on Linkedin with a list of details."
        )}
    )
    pprint(result)


if __name__ == "__main__":
    main()
