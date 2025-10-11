from langchain_community.tools import DuckDuckGoSearchRun

def web_search_tool():
    """Return a DuckDuckGo search tool usable by agents (no API key)."""
    return DuckDuckGoSearchRun()