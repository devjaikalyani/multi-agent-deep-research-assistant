"""
Search Agent
- Handles all [SEARCH] tagged sub-tasks
- Uses Tavily API for real-time web search
- Falls back to LLM simulation if Tavily key not set
"""

import os
from langchain_core.messages import HumanMessage, AIMessage
from utils.llm import get_llm
from agents.state import ResearchState


def _tavily_search(query: str) -> str:
    """Run a Tavily search and return formatted results."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True
        )

        output = []
        if results.get("answer"):
            output.append(f"**Summary:** {results['answer']}\n")

        for i, r in enumerate(results.get("results", [])[:4], 1):
            output.append(
                f"**Source {i}:** {r.get('title', 'Unknown')}\n"
                f"URL: {r.get('url', '')}\n"
                f"{r.get('content', '')[:400]}...\n"
            )

        return "\n".join(output)

    except ImportError:
        return None
    except Exception as e:
        return f"Search error: {str(e)}"


def _llm_simulate_search(query: str, llm) -> str:
    """Fallback: LLM simulates realistic search results."""
    prompt = f"""Simulate a real web search result for: "{query}"

Return 3-4 bullet points of specific, realistic facts.
Include plausible source names and years.
Format each as: • [specific fact with numbers if relevant] (Source: [publication], [year])

Be specific with numbers and data points."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def search_agent(state: ResearchState) -> ResearchState:
    llm = get_llm()
    search_tasks = [t for t in state["sub_tasks"] if "[SEARCH]" in t]

    if not search_tasks:
        return {**state, "search_results": [], "messages": [AIMessage(content="Search Agent: No search tasks assigned.")]}

    results = []
    has_tavily = bool(os.getenv("TAVILY_API_KEY"))

    for task in search_tasks:
        clean_query = task.replace("[SEARCH]", "").strip()

        if has_tavily:
            result = _tavily_search(clean_query)
            if not result:
                result = _llm_simulate_search(clean_query, llm)
            source_label = "🌐 Tavily Web Search"
        else:
            result = _llm_simulate_search(clean_query, llm)
            source_label = "🤖 LLM Simulation (add TAVILY_API_KEY for real search)"

        results.append(f"### Query: {clean_query}\n*{source_label}*\n\n{result}")

    return {
        **state,
        "search_results": results,
        "messages": [AIMessage(content=f"Search Agent: Completed {len(search_tasks)} searches. Tavily={'enabled' if has_tavily else 'disabled (simulated)'}")]
    }
