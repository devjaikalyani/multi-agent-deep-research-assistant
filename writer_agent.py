"""
Writer Agent
- Synthesizes search results + data analysis into a detailed professional report
- On revision cycles, incorporates Critic Agent feedback
"""

from langchain_core.messages import HumanMessage, AIMessage
from utils.llm import get_llm
from agents.state import ResearchState


def writer_agent(state: ResearchState) -> ResearchState:
    llm = get_llm(temperature=0.4)

    search_context = "\n\n".join(state.get("search_results", [])) or "No search results available."

    # data_results can be a list of dicts (new) or strings (legacy)
    raw_data = state.get("data_results", [])
    data_parts = []
    for dr in raw_data:
        if isinstance(dr, dict):
            task    = dr.get("task", "")
            table   = dr.get("table_md", "")
            insight = dr.get("insight", "")
            data_parts.append(f"### {task}\n{table}\n{insight}".strip())
        else:
            data_parts.append(str(dr))
    data_context = "\n\n".join(data_parts) or "No data analysis available."

    critique = state.get("critique", "")
    revision = state.get("revision_count", 0)

    revision_note = ""
    if critique and revision > 0:
        revision_note = f"""
--- REVISION INSTRUCTIONS ---
This is revision #{revision}. The Critic Agent flagged these issues:

{critique}

Address ALL points above in this revision. Be specific and thorough.
--- END REVISION INSTRUCTIONS ---
"""

    prompt = f"""You are a McKinsey-level senior research analyst. Write an exceptionally detailed, professional research report.

Original Research Query: {state['query']}

{revision_note}

## Web Search Findings:
{search_context}

## Data Analysis:
{data_context}

---

Write a comprehensive, deeply detailed research report in Markdown. This report should be substantial — minimum 1200 words of actual content.

Use this exact structure:

# [Descriptive Report Title]

**Prepared by:** Multi-Agent AI Research System
**Research Date:** 2025
**Coverage:** [brief scope description]

---

## Executive Summary

Write 4-5 sentences. Cover: (1) what the topic is and why it matters now, (2) the single most important quantitative finding, (3) key driver or trend, (4) primary risk or challenge, (5) strategic implication.

---

## Key Findings

Provide 7-9 bullet points. Each bullet must:
- Start with a bold label (e.g., **Market Size:**, **Growth Rate:**, **Top Player:**)
- Include at least one specific number, percentage, or data point
- Be 2-3 sentences long — not just a one-liner

---

## Market Overview & Current Landscape

Write 3-4 substantial paragraphs covering:
- Current state of the market/topic with specific numbers
- Historical context and how we got here
- Key segments, geographies, or sub-sectors
- Recent major developments in the last 12-18 months

---

## Detailed Analysis

Write 4-5 paragraphs. Each paragraph should focus on a distinct angle:
- Quantitative deep-dive referencing the data tables provided
- Competitive landscape: name specific companies/players with their metrics
- Technology or innovation trends driving change
- Consumer/user behavior and demand signals
- Policy, regulatory, or macro-economic context

Reference specific data from the analysis above (tables, percentages, growth rates).

---

## Growth Drivers & Opportunities

Write 3-4 paragraphs covering:
- Primary growth catalysts with supporting evidence
- Emerging opportunities (new segments, geographies, use cases)
- Investment and funding trends if applicable
- 2-3 year forward outlook with projected numbers where possible

---

## Challenges & Risk Factors

Provide a structured list of 4-5 challenges. For each:
**[Challenge Name]:** Write 2-3 sentences explaining the challenge, its magnitude, and current mitigation efforts.

---

## Strategic Implications & Recommendations

Write 3-4 paragraphs covering:
- What this means for different stakeholders (investors, operators, policymakers, consumers)
- 3-5 concrete, actionable recommendations
- Near-term priorities vs. long-term strategic moves

---

## Conclusion

Write 2-3 paragraphs: summarize the thesis, restate the most critical finding, and end with a forward-looking statement about where this topic will be in 2-3 years.

---

*Research conducted by Multi-Agent AI Research System | Data Sources: Live Web Search (Tavily) + Structured Database Analysis | Report generated 2025*

---

CRITICAL RULES:
- NEVER use dollar signs ($) before numbers — write "USD 7.3 billion" or "7.3 billion USD" instead. This prevents formatting issues.
- Use INR, USD, EUR etc. as text prefixes, never $ symbol
- Be specific — every paragraph must have at least one concrete number or data point
- Reference the data tables and search findings throughout
- Write in active, authoritative voice
- Minimum report length: 1200 words
- Use markdown tables to summarize comparative data where helpful
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        **state,
        "draft_report": response.content,
        "messages": [AIMessage(content=f"Writer Agent: Draft {'revision #' + str(revision) if revision > 0 else ''} completed ({len(response.content)} chars).")]
    }