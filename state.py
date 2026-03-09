"""
Shared state definition for the Multi-Agent Research Assistant.
All agents read from and write to this state object.
"""

from typing import TypedDict, List, Annotated
import operator


class ResearchState(TypedDict):
    query: str                            # Original user query
    sub_tasks: List[str]                  # Tasks decomposed by Orchestrator
    search_results: List[str]             # Web search results
    data_results: List[str]               # Structured data analysis results
    chart_data: List[dict]                # Plotly chart specs for UI
    draft_report: str                     # First draft from Writer Agent
    critique: str                         # Critic Agent feedback
    final_report: str                     # Final approved report
    revision_count: int                   # How many times Writer has revised
    messages: Annotated[List, operator.add]  # Full agent message log
