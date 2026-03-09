"""
Orchestrator Agent
- Receives the user's research query
- Decomposes it into tagged sub-tasks for each specialist agent
- Tags: [SEARCH], [DATA], [WRITE]
"""

import json
from langchain_core.messages import HumanMessage, AIMessage
from utils.llm import get_llm
from agents.state import ResearchState


def orchestrator_agent(state: ResearchState) -> ResearchState:
    llm = get_llm()

    prompt = f"""You are an expert research orchestrator managing a team of AI agents.

User Query: {state['query']}

Decompose this into 4-6 specific sub-tasks. Tag each with the responsible agent:
- [SEARCH] → needs real-time web search (trends, news, recent data)
- [DATA]   → needs structured data analysis (numbers, comparisons, statistics)
- [WRITE]  → synthesis or writing task (summaries, conclusions, implications)

Rules:
- Be specific. Bad: "[SEARCH] find EV info". Good: "[SEARCH] Find India EV market size and growth rate 2023-2025"
- Mix at least one of each tag type
- Return ONLY a valid JSON array of strings, no markdown fences

Example output:
["[SEARCH] Find India EV passenger car sales volumes Q1-Q4 2024",
 "[SEARCH] Identify top 5 EV manufacturers in India by market share 2024",
 "[DATA] Compare EV adoption rates across India, China, USA, Europe",
 "[DATA] Analyze price range distribution of EVs available in India",
 "[WRITE] Summarize key growth drivers and policy tailwinds for India EV market",
 "[WRITE] Outline main challenges: infrastructure gaps, battery costs, range anxiety"]
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        sub_tasks = json.loads(raw.strip())
    except Exception:
        # Fallback: parse line by line
        lines = [l.strip() for l in response.content.split("\n") if l.strip() and ("[SEARCH]" in l or "[DATA]" in l or "[WRITE]" in l)]
        sub_tasks = lines if lines else [
            "[SEARCH] Research the main aspects of: " + state["query"],
            "[DATA] Analyze key statistics related to: " + state["query"],
            "[WRITE] Summarize findings and implications for: " + state["query"],
        ]

    return {
        **state,
        "sub_tasks": sub_tasks,
        "revision_count": 0,
        "messages": [AIMessage(content=f"Orchestrator: Decomposed query into {len(sub_tasks)} sub-tasks: {', '.join(sub_tasks[:2])}...")]
    }
