# 🤖 Multi-Agent Deep Research Assistant

> 5 specialized LangGraph agents that autonomously research, analyze, and report on any topic.

## Architecture

```
User Query → Orchestrator → Search Agent → Data Agent → Writer Agent → Critic Agent → Final Report
                                                               ↑               |
                                                               └── revision loop┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph (StateGraph) |
| LLM | GPT-4o or Claude Sonnet |
| Web Search | Tavily API |
| Database | SQLite + Pandas |
| Vector Memory | ChromaDB |
| Charts | Plotly |
| UI | Streamlit |

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/research-agent
cd research-agent
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
python3.11 -m venv .venv311 # change venv to venv311 which have python 3.11
source .venv311/bin/activate # uses python 3.11 compatible with all the used packages
streamlit run app.py
```

## Project Structure

```
research-agent/
├── app.py                 # Streamlit UI
├── graph.py               # LangGraph pipeline
├── requirements.txt
├── .env.example
├── agents/
│   ├── state.py           # Shared ResearchState
│   ├── orchestrator.py    # Task decomposition
│   ├── search_agent.py    # Tavily web search
│   ├── data_agent.py      # SQL + Pandas + charts
│   ├── writer_agent.py    # Report synthesis
│   └── critic_agent.py    # Quality review + revision loop
└── utils/
    ├── llm.py             # OpenAI / Claude factory
    ├── db.py              # SQLite + CSV loader
    └── memory.py          # ChromaDB vector memory
```

## Adding Your Own Data

```python
from utils.db import load_csv_to_db
load_csv_to_db("my_data.csv", "my_table")
```

## Resume Description

> **Multi-Agent Deep Research Assistant** | LangGraph · GPT-4o · Tavily · SQLite · ChromaDB · Streamlit
>
> Built a multi-agent system with 5 specialized agents using LangGraph. Agents autonomously decompose queries, retrieve real-time web data, run SQL/Pandas analysis, synthesize reports, and self-improve via a Critic→Writer revision loop. Implemented ChromaDB vector memory for multi-turn session context.

## Deploy

Push to GitHub → connect to [share.streamlit.io](https://share.streamlit.io) → add secrets → live in 2 minutes.
