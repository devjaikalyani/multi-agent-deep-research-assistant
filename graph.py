"""
LangGraph Pipeline
Assembles all 5 agents into a directed graph with a conditional revision loop.

Flow:
  orchestrator → search → data → writer → critic
                                    ↑         |
                                    └─revise──┘ (if critique says REVISE)
                                              |
                                             END (if APPROVE)
"""

from langgraph.graph import StateGraph, END
from agents.state import ResearchState
from agents.orchestrator import orchestrator_agent
from agents.search_agent import search_agent
from agents.data_agent import data_agent
from agents.writer_agent import writer_agent
from agents.critic_agent import critic_agent, should_revise


def build_graph():
    """Build and compile the full research agent graph."""
    workflow = StateGraph(ResearchState)

    # Register all agent nodes
    workflow.add_node("orchestrator", orchestrator_agent)
    workflow.add_node("search",       search_agent)
    workflow.add_node("data",         data_agent)
    workflow.add_node("writer",       writer_agent)
    workflow.add_node("critic",       critic_agent)

    # Linear flow
    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "search")
    workflow.add_edge("search",       "data")
    workflow.add_edge("data",         "writer")
    workflow.add_edge("writer",       "critic")

    # Conditional edge: loop back to writer or end
    workflow.add_conditional_edges(
        "critic",
        should_revise,
        {
            "revise": "writer",  # Loop back for revision
            "end":    END        # Finish
        }
    )

    return workflow.compile()


def run_research(query: str) -> dict:
    """
    Main entry point. Run the full research pipeline.
    Returns the final state dict with all results.
    """
    graph = build_graph()

    initial_state = ResearchState(
        query=query,
        sub_tasks=[],
        search_results=[],
        data_results=[],
        chart_data=[],
        draft_report="",
        critique="",
        final_report="",
        revision_count=0,
        messages=[]
    )

    return graph.invoke(initial_state)


if __name__ == "__main__":
    result = run_research("Analyze the EV market in India for 2025")
    print(result["final_report"])
