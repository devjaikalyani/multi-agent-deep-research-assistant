"""
Data Agent - Fixed version with robust JSON parsing
"""

import re
import json
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from utils.llm import get_llm
from utils.db import get_db_connection, get_db_schema
from agents.state import ResearchState


def _generate_sql(task: str, schema: str, llm) -> str:
    response = llm.invoke([HumanMessage(content=f"""You are an expert SQL analyst.

Database Schema:
{schema}

Task: {task}

Write a single SQL SELECT query to answer this task.
Return ONLY the raw SQL query. No markdown, no explanation.
If schema has no relevant tables, return exactly: NO_MATCH""")])
    return response.content.strip()


def _run_sql(query: str):
    try:
        conn = get_db_connection()
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df, ""
    except Exception as e:
        return None, str(e)


def _df_to_chart_spec(df: pd.DataFrame, task: str):
    if df is None or df.empty or len(df.columns) < 2:
        return None
    try:
        x_col = df.columns[0]
        num_cols = df.select_dtypes(include="number").columns
        y_col = num_cols[0] if len(num_cols) > 0 else df.columns[1]

        x_values = []
        for v in df[x_col].tolist():
            if isinstance(v, (pd.Timestamp, pd.Period)):
                x_values.append(str(v))
            elif pd.isna(v):
                x_values.append(None)
            else:
                x_values.append(v)

        y_values = []
        for v in df[y_col].tolist():
            if isinstance(v, (pd.Timestamp, pd.Period)):
                y_values.append(str(v))
            elif pd.isna(v):
                y_values.append(None)
            else:
                y_values.append(float(v) if isinstance(v, (int, float)) else v)

        return {
            "type": "bar",
            "x": [str(v) if v is not None else "" for v in x_values],
            "y": y_values,
            "x_label": str(x_col),
            "y_label": str(y_col),
            "title": str(task)[:70]
        }
    except Exception as e:
        print(f"Error creating chart spec: {e}")
        return None


def _safe_parse_json(raw: str):
    if not raw:
        return None
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def _llm_simulate_data(task: str, llm):
    prompt = f"""You are a data analyst. Generate realistic data for:

Task: {task}

Return ONLY a valid JSON object, no extra text:
{{
  "table_md": "| Col1 | Col2 | Col3 |\\n| --- | --- | --- |\\n| row1a | row1b | row1c |\\n| row2a | row2b | row2c |",
  "insight": "2-3 sentences with specific numbers and key finding.",
  "chart": {{
    "type": "bar",
    "x": ["A", "B", "C", "D"],
    "y": [12.5, 18.2, 22.1, 25.5],
    "x_label": "Category",
    "y_label": "Value",
    "title": "Chart Title Here"
  }}
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = _safe_parse_json(response.content)

    if parsed and isinstance(parsed, dict) and "table_md" in parsed:
        chart = parsed.get("chart")
        if chart and isinstance(chart, dict) and all(k in chart for k in ["x", "y", "title"]):
            chart["x"] = [str(x) for x in chart.get("x", [])]
            chart["y"] = [float(y) if isinstance(y, (int, float)) else 0 for y in chart.get("y", [])]
            chart["x_label"] = str(chart.get("x_label", ""))
            chart["y_label"] = str(chart.get("y_label", ""))
            chart["title"] = str(chart.get("title", ""))
        else:
            chart = None
        return parsed.get("table_md", ""), parsed.get("insight", ""), chart
    else:
        return f"Analysis completed for: {task}", "", None


def data_agent(state: ResearchState) -> ResearchState:
    llm = get_llm()
    data_tasks = [t for t in state["sub_tasks"] if "[DATA]" in t]

    if not data_tasks:
        return {
            **state,
            "data_results": [],
            "chart_data": [],
            "messages": [AIMessage(content="Data Agent: No data tasks assigned.")]
        }

    results = []
    charts = list(state.get("chart_data", []))
    schema = get_db_schema()

    for task in data_tasks:
        clean_task = task.replace("[DATA]", "").strip()

        sql = _generate_sql(clean_task, schema, llm)
        if sql and "NO_MATCH" not in sql.upper() and sql.upper().lstrip().startswith("SELECT"):
            df, error = _run_sql(sql)
            if df is not None and not df.empty:
                chart = _df_to_chart_spec(df, clean_task)
                if chart:
                    charts.append(chart)
                results.append({
                    "task": clean_task,
                    "source": "sqlite",
                    "sql": sql,
                    "table_md": df.to_markdown(index=False),
                    "insight": "",
                    "chart": chart
                })
                continue

        table_md, insight, chart = _llm_simulate_data(clean_task, llm)
        if chart:
            charts.append(chart)
        results.append({
            "task": clean_task,
            "source": "llm",
            "sql": None,
            "table_md": table_md,
            "insight": insight,
            "chart": chart
        })

    serializable_charts = []
    for chart in charts:
        if chart and isinstance(chart, dict):
            try:
                json.dumps(chart)
                serializable_charts.append(chart)
            except (TypeError, ValueError):
                safe_chart = {
                    "type": str(chart.get("type", "bar")),
                    "x": [str(x) for x in chart.get("x", [])],
                    "y": [float(y) if isinstance(y, (int, float)) else 0 for y in chart.get("y", [])],
                    "x_label": str(chart.get("x_label", "")),
                    "y_label": str(chart.get("y_label", "")),
                    "title": str(chart.get("title", ""))
                }
                serializable_charts.append(safe_chart)

    return {
        **state,
        "data_results": results,
        "chart_data": serializable_charts,
        "messages": [AIMessage(content=f"Data Agent: Completed {len(data_tasks)} analyses with {len(serializable_charts)} charts.")]
    }