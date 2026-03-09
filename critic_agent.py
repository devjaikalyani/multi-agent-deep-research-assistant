"""
Critic Agent
- Scores the draft report on 4 dimensions
- Returns APPROVE or REVISE with specific feedback
- LangGraph router uses this to either finalize or loop back to Writer
"""

from langchain_core.messages import HumanMessage, AIMessage
from utils.llm import get_llm
from agents.state import ResearchState

MAX_REVISIONS = 2  # Max times Writer will revise before forced approval


def critic_agent(state: ResearchState) -> ResearchState:
    llm = get_llm()

    revision_count = state.get("revision_count", 0)

    # Force approve after max revisions to avoid infinite loops
    if revision_count >= MAX_REVISIONS:
        return {
            **state,
            "critique": "APPROVE (max revisions reached)",
            "final_report": state["draft_report"],
            "messages": [AIMessage(content=f"Critic Agent: Max revisions reached. Force approving.")]
        }

    prompt = f"""You are a strict research quality critic.

Original Query: {state['query']}

Report to Review:
{state['draft_report']}

Score the report on each dimension (1-5):
1. Query Coverage: Does it fully answer the original query?
2. Data Support: Are claims backed by specific data/numbers?
3. Structure: Is it well-organized with clear sections?
4. Depth: Does it go beyond surface-level observations?

Then give a verdict:

If ALL scores are 4 or 5 → respond with:
APPROVE
[Brief reason why it's good]

If ANY score is 3 or below → respond with:
REVISE
- [Specific issue 1 with how to fix it]
- [Specific issue 2 with how to fix it]
- [Specific issue 3 with how to fix it]

Start your response with exactly APPROVE or REVISE on the first line."""

    response = llm.invoke([HumanMessage(content=prompt)])
    critique = response.content.strip()

    approved = critique.upper().startswith("APPROVE")

    if approved:
        final_report = state["draft_report"]
        verdict = "APPROVED ✅"
    else:
        final_report = ""  # Will be set after Writer revises
        verdict = "REVISE requested"

    return {
        **state,
        "critique": critique,
        "final_report": final_report,
        "revision_count": revision_count + (0 if approved else 1),
        "messages": [AIMessage(content=f"Critic Agent: {verdict}. Revision #{revision_count}.")]
    }


def should_revise(state: ResearchState) -> str:
    """
    LangGraph conditional edge function.
    Returns 'revise' to loop back to Writer, or 'end' to finish.
    """
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    if critique.upper().startswith("APPROVE") or revision_count >= MAX_REVISIONS:
        return "end"
    return "revise"
