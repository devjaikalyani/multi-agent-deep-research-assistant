"""
Multi-Agent Deep Research Assistant — Professional UI v3
Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import re, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import build_graph
from agents.state import ResearchState
from utils.memory import (save_research, retrieve_similar,
                           get_all_sessions, get_session_report, memory_available)

# ─── Session State ─────────────────────────────────────────────────────────────
for _k, _v in [("loaded_report", None), ("loaded_query", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global background */
  [data-testid="stAppViewContainer"] { background:#0f1117; }
  [data-testid="stSidebar"]          { background:#161b27; border-right:1px solid #1e2436; }
  .block-container                   { padding-top:2rem; padding-bottom:4rem; max-width:1100px; }

  /* ── FONT SIZES (global boost) ── */
  html, body                         { font-size:17px !important; }
  p, li, label                       { font-size:16px !important; line-height:1.9 !important; }
  td, th                             { font-size:15px !important; }

  /* ── Hero ── */
  .hero-title {
    font-size:3.2rem; font-weight:800; line-height:1.15; margin-bottom:0.4rem;
    background:linear-gradient(135deg,#a78bfa,#60a5fa,#34d399);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  }
  .hero-sub { color:#64748b; font-size:1.1rem; }

  /* ── Pipeline steps ── */
  .pipe-step {
    text-align:center; padding:11px 8px; border-radius:10px;
    font-size:16px; font-weight:600; border:1.5px solid transparent; transition:all 0.3s;
  }
  .pipe-idle    { background:#1a2035; color:#3d4f6e; border-color:#1e2a42; }
  .pipe-done    { background:#0d2818; color:#4ade80; border-color:#166534; }
  .pipe-running { background:#1c1400; color:#fbbf24; border-color:#854d0e;
                  animation:pulse 1.4s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.55} }

  /* ── Metric cards ── */
  .metric-card {
    background:#1a2035; border:1px solid #2e3650; border-radius:12px;
    padding:16px 10px; text-align:center;
  }
  .metric-val { font-size:1.5rem; font-weight:800; color:#a78bfa; }
  .metric-lbl { font-size:13px; color:#64748b; margin-top:3px; }

  /* ── Report headings & text ── */
  [data-testid="stMarkdownContainer"] h1 {
    color:#a78bfa !important; font-size:2.1rem !important; font-weight:800;
    border-bottom:2px solid #2e3650; padding-bottom:14px; margin-bottom:20px;
  }
  [data-testid="stMarkdownContainer"] h2 {
    color:#60a5fa !important; font-size:1.5rem !important; font-weight:700;
    border-left:4px solid #60a5fa; padding-left:14px; margin-top:2.2rem;
  }
  [data-testid="stMarkdownContainer"] h3 {
    color:#34d399 !important; font-size:1.2rem !important; font-weight:600; margin-top:1.4rem;
  }
  [data-testid="stMarkdownContainer"] p  {
    color:#cbd5e1 !important; font-size:16px !important; line-height:1.95 !important; margin-bottom:14px;
  }
  [data-testid="stMarkdownContainer"] li {
    color:#94a3b8 !important; font-size:16px !important; line-height:1.9 !important; margin-bottom:8px;
  }
  [data-testid="stMarkdownContainer"] strong { color:#e2e8f0 !important; }
  [data-testid="stMarkdownContainer"] blockquote {
    border-left:4px solid #7c3aed; background:#1a1535;
    padding:12px 18px; border-radius:0 8px 8px 0; margin:18px 0;
  }
  [data-testid="stMarkdownContainer"] table { width:100%; border-collapse:collapse; }
  [data-testid="stMarkdownContainer"] th {
    background:#1e2a42; color:#a78bfa !important;
    padding:11px 14px; font-size:14px !important;
  }
  [data-testid="stMarkdownContainer"] td {
    border-bottom:1px solid #1e2436; padding:11px 14px;
    color:#cbd5e1 !important; font-size:14px !important;
  }

  /* ── Data analysis cards ── */
  .analysis-card {
    background:#161b27; border:1px solid #2e3650; border-radius:12px;
    padding:18px 22px; margin-bottom:18px;
  }
  .analysis-title { font-size:15px; font-weight:700; color:#e2e8f0; margin-bottom:6px; }
  .badge-sql { background:#0d2818; color:#4ade80; padding:2px 9px; border-radius:10px; font-size:11px; font-weight:700; }
  .badge-llm { background:#1a1a35; color:#818cf8; padding:2px 9px; border-radius:10px; font-size:11px; font-weight:700; }
  .insight-box {
    background:#1e2a42; border-left:3px solid #60a5fa; border-radius:0 8px 8px 0;
    padding:10px 14px; margin-top:12px; color:#93c5fd; font-size:14px; line-height:1.7;
  }

  /* ── Source cards ── */
  .source-card {
    background:#161b27; border:1px solid #2e3650; border-radius:10px;
    padding:16px 20px; margin-bottom:12px;
  }
  .source-title { font-weight:700; color:#e2e8f0; font-size:15px; }
  .source-body  { color:#64748b; font-size:14px; margin-top:8px; line-height:1.75; }

  /* ── Status badges ── */
  .badge-approved {
    background:#0d2818; color:#4ade80; border:1px solid #166534;
    border-radius:20px; padding:5px 16px; font-size:14px; font-weight:600; display:inline-block;
  }
  .badge-revised {
    background:#1c1400; color:#fbbf24; border:1px solid #854d0e;
    border-radius:20px; padding:5px 16px; font-size:14px; font-weight:600; display:inline-block;
  }

  /* ── Past session banner ── */
  .session-banner {
    background:#1a1535; border:1px solid #7c3aed; border-radius:12px;
    padding:16px 22px; margin-bottom:22px;
  }
  .session-label {
    font-size:11px; color:#7c3aed; font-weight:700;
    text-transform:uppercase; letter-spacing:1.2px;
  }
  .session-query { font-size:17px; color:#e2e8f0; font-weight:600; margin-top:6px; }

  /* ── Sidebar ── */
  .sb-section { color:#64748b; font-size:18px; text-transform:uppercase; letter-spacing:1px; font-weight:700; margin:18px 0 8px; }
  .agent-row  { display:flex; align-items:flex-start; gap:10px; padding:9px 0; border-bottom:1px solid #1e2436; }
  .agent-icon { font-size:18px; flex-shrink:0; }
  .agent-name { color:#e2e8f0; font-size:18px; font-weight:600; }
  .agent-desc { color:#475569; font-size:16px; margin-top:1px; }

  /* ── Tabs ── */
  button[data-baseweb="tab"]                       { color:#475569 !important; font-size:14px !important; }
  button[data-baseweb="tab"][aria-selected="true"] { color:#a78bfa !important; border-bottom-color:#a78bfa !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width:6px; height:6px; }
  ::-webkit-scrollbar-track { background:#0f1117; }
  ::-webkit-scrollbar-thumb { background:#2e3650; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─── Agent Definitions ─────────────────────────────────────────────────────────
AGENTS = [
    ("🧠", "orchestrator", "Orchestrator", "Breaks query into sub-tasks"),
    ("🔍", "search",       "Search",       "Real-time web search"),
    ("📊", "data",         "Data",         "SQL + Pandas analysis"),
    ("✍️",  "writer",       "Writer",       "Synthesizes the report"),
    ("✅", "critic",       "Critic",       "Reviews & approves"),
]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 16px">
      <div style="font-size:25px;font-weight:800;color:#a78bfa;">🔬 Research Agent</div>
      <div style="font-size:18px;color:#475569;margin-top:4px;">5 AI agents · LangGraph pipeline</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Agent Pipeline</div>', unsafe_allow_html=True)
    for icon, _, name, desc in AGENTS:
        st.markdown(f"""
        <div class="agent-row">
          <div class="agent-icon">{icon}</div>
          <div><div class="agent-name">{name}</div><div class="agent-desc">{desc}</div></div>
        </div>""", unsafe_allow_html=True)

    # ── Past Sessions ──────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Past Sessions</div>', unsafe_allow_html=True)
    _mem_ok   = memory_available()
    _sessions = get_all_sessions() if _mem_ok else []

    if _mem_ok and _sessions:
        for _i, _s in enumerate(_sessions[:7]):
            _ts    = _s.get("timestamp", "")[:10]
            _q     = _s.get("query", "")
            _wc    = _s.get("report_length", 0)
            _wc_s  = f"{_wc // 5:,} words" if _wc else ""
            _short = (_q[:48] + "…") if len(_q) > 48 else _q
            _key   = f"sb_sess_{_i}_{_ts}_{_s.get('doc_id','')}"

            # Render as a styled button — one per session
            if st.button(
                f"{_short}",
                key=_key,
                use_container_width=True,
                help=f"{_ts}  ·  {_wc_s}"
            ):
                _report = get_session_report(_q)
                if _report:
                    st.session_state.loaded_report = _report
                    st.session_state.loaded_query  = _q
                    st.rerun()
                else:
                    st.session_state.loaded_report = "⚠️ This session was saved before full-report storage. Please re-run the query to save it."
                    st.session_state.loaded_query  = _q
                    st.rerun()
    elif _mem_ok:
        st.caption("No sessions yet — run a query!")
    else:
        st.markdown('<div style="font-size:13px;color:#ef4444;">⚠ chromadb not installed</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Stack</div>', unsafe_allow_html=True)
    for _tech in ["LangGraph", "GPT-4o / Claude", "Tavily Search",
                  "SQLite + Pandas", "ChromaDB", "Streamlit + Plotly"]:
        st.markdown(f'<div style="font-size:18px;color:#475569;padding:2px 0;">· {_tech}</div>',
                    unsafe_allow_html=True)


# ─── Main: Hero ───────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Multi-Agent Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Ask any question — 5 specialized AI agents research, analyze, and report.</div>',
            unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ─── Search bar ───────────────────────────────────────────────────────────────
c1, c2 = st.columns([5, 1])
with c1:
    query = st.text_input("q", placeholder="🔍  e.g. Analyze the EV market in India for 2026",
                          label_visibility="collapsed", key="main_query")
with c2:
    run_btn = st.button("Run Research →", use_container_width=True, type="primary")

# Example chips
chip_cols = st.columns(5)
EXAMPLES = ["EV market India 2026","AI in healthcare trends",
            "AWS vs GCP vs Azure","Generative AI startups","Crypto market 2026"]
for _i, _ex in enumerate(EXAMPLES):
    if chip_cols[_i].button(_ex, key=f"chip_{_i}"):
        query   = _ex
        run_btn = True

st.markdown("<br>", unsafe_allow_html=True)

# ─── Pipeline bar ─────────────────────────────────────────────────────────────
_pipe_ph = st.empty()

def render_pipeline(done: list = None):
    done = done or []
    cols = _pipe_ph.columns(5)
    for i, (icon, key, name, _) in enumerate(AGENTS):
        if key in done:
            cls, sfx = "pipe-done",  " ✓"
        else:
            cls, sfx = "pipe-idle",  ""
        cols[i].markdown(f'<div class="pipe-step {cls}">{icon} {name}{sfx}</div>',
                         unsafe_allow_html=True)

render_pipeline()

# ─── Past Session Viewer ──────────────────────────────────────────────────────
if st.session_state.loaded_report and not (run_btn and query.strip()):
    _lq = st.session_state.loaded_query or ""
    _lr = st.session_state.loaded_report

    st.markdown(
        f'<div class="session-banner">'
        f'<div class="session-label">📂 Past Session</div>'
        f'<div class="session-query">{_lq}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    _dc1, _dc2, _ = st.columns([1, 1, 4])
    _dc1.download_button("⬇ .md", data=_lr, file_name="report.md",
                         mime="text/markdown", use_container_width=True)
    _dc2.download_button("⬇ .txt", data=_lr, file_name="report.txt",
                         mime="text/plain", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_lr)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("✕  Clear & run new research", key="clear_sess"):
        st.session_state.loaded_report = None
        st.session_state.loaded_query  = None
        st.rerun()
    st.stop()

# ─── Run Pipeline ─────────────────────────────────────────────────────────────
if run_btn and query.strip():

    graph = build_graph()
    init  = ResearchState(
        query=query, sub_tasks=[], search_results=[], data_results=[],
        chart_data=[], draft_report="", critique="", final_report="",
        revision_count=0, messages=[]
    )

    done_agents = []
    accumulated = dict(init)

    with st.status("🤖 Running research pipeline…", expanded=True) as _status:
        try:
            for step in graph.stream(init):
                node = list(step.keys())[0]
                snap = step[node]

                for k, v in snap.items():
                    if k == "messages" and isinstance(v, list):
                        accumulated["messages"] = accumulated.get("messages", []) + v
                    else:
                        accumulated[k] = v

                done_agents.append(node)
                render_pipeline(done=done_agents)

                if node == "orchestrator":
                    tasks = snap.get("sub_tasks", [])
                    st.markdown(f"**🧠 Orchestrator** — {len(tasks)} sub-tasks:")
                    for t in tasks:
                        tag = "🔍" if "[SEARCH]" in t else "📊" if "[DATA]" in t else "✍️"
                        st.markdown(f"&nbsp;&nbsp;&nbsp;{tag} `{t}`")
                elif node == "search":
                    n = len(snap.get("search_results", []))
                    tavily = any("Tavily" in str(m.content) for m in snap.get("messages", []))
                    st.markdown(f"**🔍 Search** — {n} searches · {'🌐 Tavily live' if tavily else '🤖 simulated'}")
                elif node == "data":
                    st.markdown(f"**📊 Data** — {len(snap.get('data_results',[]))} analyses · {len(snap.get('chart_data',[]))} charts")
                elif node == "writer":
                    rev = snap.get("revision_count", 0)
                    st.markdown(f"**✍️ Writer** — {'revision #'+str(rev) if rev else 'first draft'} · {len(snap.get('draft_report','')):,} chars")
                elif node == "critic":
                    ok = snap.get("critique","").upper().startswith("APPROVE")
                    st.markdown(f"**✅ Critic** — {'APPROVED ✓' if ok else 'revision requested'}")

            _status.update(label="✅ Research complete!", state="complete")

        except Exception as e:
            _status.update(label="❌ Error", state="error")
            st.error(f"**Pipeline error:** {e}")
            st.info("Check your .env — make sure OPENAI_API_KEY is set.")
            st.stop()

    final = accumulated
    if not final.get("final_report") and final.get("draft_report"):
        final["final_report"] = final["draft_report"]

    if not final.get("final_report"):
        st.error("No report was generated. Please try again.")
        st.stop()

    # ── Stats row ─────────────────────────────────────────────────────────────
    st.markdown("---")
    critique  = final.get("critique", "")
    approved  = critique.upper().startswith("APPROVE")
    n_src     = len(final.get("search_results", []))
    n_data    = len(final.get("data_results", []))
    n_charts  = len(final.get("chart_data", []))
    n_words   = len(final.get("final_report","").split())

    hc = st.columns([3,1,1,1,1])
    hc[0].markdown(
        f'<span class="{"badge-approved" if approved else "badge-revised"}">{"✓ Critic Approved" if approved else "⚠ Max revisions"}</span>',
        unsafe_allow_html=True
    )
    for col, val, lbl in [(hc[1],n_src,"Searches"),(hc[2],n_data,"Analyses"),(hc[3],n_charts,"Charts"),(hc[4],f"{n_words:,}","Words")]:
        col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t_report, t_data, t_sources, t_log = st.tabs(["📄 Report","📊 Data & Charts","🔍 Sources","📋 Agent Log"])

    # ── Report ────────────────────────────────────────────────────────────────
    with t_report:
        _r = final["final_report"]
        _d1, _d2, _ = st.columns([1,1,4])
        _d1.download_button("⬇ Download .md", data=_r, file_name="report.md",
                            mime="text/markdown", use_container_width=True)
        _d2.download_button("⬇ Download .txt", data=_r, file_name="report.txt",
                            mime="text/plain", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(_r)

    # ── Data & Charts ─────────────────────────────────────────────────────────
    with t_data:
        data_results = final.get("data_results", [])
        chart_data   = final.get("chart_data", [])

        if not data_results:
            st.info("No data analyses generated for this query.")
        else:
            for _i, dr in enumerate(data_results):
                if isinstance(dr, dict):
                    task    = dr.get("task", f"Analysis {_i+1}")
                    source  = dr.get("source", "llm")
                    sql     = dr.get("sql")
                    tbl_md  = dr.get("table_md", "")
                    insight = dr.get("insight", "")
                    chart   = dr.get("chart")
                else:
                    task    = f"Analysis {_i+1}"
                    source  = "llm"
                    sql     = None
                    insight = ""
                    chart   = chart_data[_i] if _i < len(chart_data) else None
                    tbl_md  = re.sub(r'\{[\s\S]*?\}', '', str(dr)).strip()

                badge = '<span class="badge-sql">📂 Live SQLite</span>' if source=="sqlite" else '<span class="badge-llm">🤖 LLM Analysis</span>'
                st.markdown(f'<div class="analysis-card"><div class="analysis-title">{task}</div><div style="margin-top:4px;">{badge}</div></div>',
                            unsafe_allow_html=True)

                if chart and chart.get("x") and chart.get("y"):
                    cc, ct = st.columns(2)
                    with cc:
                        fig = go.Figure(go.Bar(
                            x=chart.get("x",[]), y=chart.get("y",[]),
                            marker=dict(color=chart.get("y",[]), colorscale="Purples", showscale=False)
                        ))
                        fig.update_layout(
                            title=dict(text=chart.get("title",""), font=dict(size=13, color="#e2e8f0")),
                            xaxis=dict(title=chart.get("x_label",""), tickfont=dict(color="#64748b"), gridcolor="#1e2436"),
                            yaxis=dict(title=chart.get("y_label",""), tickfont=dict(color="#64748b"), gridcolor="#1e2436"),
                            paper_bgcolor="#161b27", plot_bgcolor="#161b27",
                            height=300, margin=dict(l=10,r=10,t=40,b=10)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with ct:
                        if tbl_md: st.markdown(tbl_md)
                        if insight: st.markdown(f'<div class="insight-box">💡 {insight}</div>', unsafe_allow_html=True)
                else:
                    if tbl_md:    st.markdown(tbl_md)
                    if insight:   st.markdown(f'<div class="insight-box">💡 {insight}</div>', unsafe_allow_html=True)

                if sql:
                    with st.expander("View SQL Query"):
                        st.code(sql, language="sql")
                st.markdown("<br>", unsafe_allow_html=True)

    # ── Sources ───────────────────────────────────────────────────────────────
    with t_sources:
        search_results = final.get("search_results", [])
        if not search_results:
            st.info("No web search results for this query.")
        else:
            for sr in search_results:
                lines = str(sr).split("\n")
                title = lines[0].replace("### Query:","").replace("#","").strip() if lines else "Search Result"
                body  = "\n".join(lines[1:]).strip()
                st.markdown(f'<div class="source-card"><div class="source-title">🔍 {title}</div><div class="source-body">{body[:900]}</div></div>',
                            unsafe_allow_html=True)

    # ── Agent Log ─────────────────────────────────────────────────────────────
    with t_log:
        sub_tasks = final.get("sub_tasks", [])
        messages  = final.get("messages",  [])
        critique  = final.get("critique",  "")

        if sub_tasks:
            st.markdown("**Sub-task Breakdown**")
            for t in sub_tasks:
                clr = "#60a5fa" if "[SEARCH]" in t else "#34d399" if "[DATA]" in t else "#a78bfa"
                st.markdown(f'<div style="padding:7px 12px;margin:3px 0;background:#1a2035;border-radius:6px;border-left:3px solid {clr};font-size:14px;color:#94a3b8;">{t}</div>',
                            unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**Agent Messages**")
        for msg in messages:
            txt = msg.content if hasattr(msg, 'content') else str(msg)
            st.markdown(f'<div style="padding:6px 12px;margin:3px 0;background:#161b27;border-radius:6px;font-size:13px;color:#475569;border:1px solid #1e2436;">• {txt}</div>',
                        unsafe_allow_html=True)

        if critique:
            st.markdown("<br>**Critic Feedback**")
            st.markdown(f'<div style="padding:12px 16px;background:#0d1a0d;border:1px solid #166534;border-radius:8px;font-size:14px;color:#4ade80;line-height:1.7;">{critique}</div>',
                        unsafe_allow_html=True)

    # ── Save to memory ────────────────────────────────────────────────────────
    if memory_available():
        if save_research(query, final["final_report"]):
            st.markdown('<div style="text-align:right;font-size:12px;color:#2e3650;margin-top:8px;">💾 Saved to memory</div>',
                        unsafe_allow_html=True)

elif run_btn and not query.strip():
    st.warning("Please enter a research question first.")