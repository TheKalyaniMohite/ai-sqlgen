# app.py  — streamlined, production-friendly UI for your AI SQL Copilot

import os
import re
import time
import sqlite3
import html
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ===== LLM client (OpenAI) =====
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # app still works without OpenAI for limited fallbacks

# ------------------------------------------------------------
# Config / env
# ------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "data" / "telco_churn.csv"
DEFAULT_DB_URI = "file:data/telco.db?mode=ro"

# ------------------------------------------------------------
# Global styles (home gradient + dark Schema/Results)
# ------------------------------------------------------------
st.set_page_config(page_title="AI SQL Copilot", layout="wide")

BASE_CSS = """
<style>
:root{
  --bg-dark: #0E1116;
  --canvas:  #262B33;
  --panel:   #2C313A;
  --panel-br:#3B424E;
  --text:    #E9EDF4;
  --muted:   #A8B0BD;
  --accent:  #6A4AF8;
    --glass:   rgba(255,255,255,0.92);
    --glass-dark: rgba(26,29,34,0.92);
}

html, body{font-family:"Inter", "SF Pro Display", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;}

/* ===== Home (gradient) ===== */
.stApp { background: linear-gradient(180deg, #F3F4F6 0%, #1F2329 100%) !important; }
.stApp > header, .main, .block-container { background: transparent !important; }
.block-container{padding:2.6rem 2.8rem 4.2rem!important;max-width:1180px;margin:0 auto;}
.section-title{font-size:1.1rem;font-weight:700;text-transform:uppercase;letter-spacing:.18em;color:#4B5563;margin:2.6rem 0 1rem;text-align:center}
.section-subtitle{color:#4B5563;max-width:760px;margin:0 auto 2.2rem;text-align:center;font-weight:500}
.breadcrumb{display:flex;justify-content:flex-end;align-items:center;font-size:.82rem;letter-spacing:.18em;color:#94A3B8;padding:.45rem 0;text-transform:uppercase}
.breadcrumb span{margin-left:.5rem;font-weight:700;color:#F3F5FB;letter-spacing:.04em;text-transform:none}

/* Hero */
.hero-wrap{display:flex;justify-content:center;margin-top:1.6rem;margin-bottom:1.2rem;padding:0 1rem}
.hero{max-width:1100px;text-align:center!important}
.hero .title-line1{display:block;font-weight:900;letter-spacing:.02em;color:#1F2329;font-size:clamp(2.6rem,7.2vw,4.8rem);line-height:1.05;margin:0}
.hero .title-line2{display:block;font-weight:800;letter-spacing:.02em;color:#000;font-size:clamp(2.3rem,6.8vw,4.4rem);line-height:1.05;margin:.35rem 0 1.2rem}
.hero .subtitle{color:#1f2937;opacity:.9;max-width:980px;margin:0 auto}

/* Dual CTAs */
.cta{display:flex;justify-content:center;gap:1rem;margin-top:1.8rem;flex-wrap:wrap}
.cta .stButton > button{
  background:#0F172A;color:#E6EAF5;border:1px solid #1F2937;border-radius:12px;
    padding:.9rem 1.9rem;font-weight:700;box-shadow:2px 4px 0 rgba(0,0,0,.25);min-width:230px
}
.cta .stButton > button:hover{transform:translateY(-1px);box-shadow:3px 6px 0 rgba(0,0,0,.25)}

.feature-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1.4rem;margin:3rem auto 0;max-width:1020px;padding:0 1rem}
.feature-card{background:var(--glass);border:1px solid rgba(15,23,42,0.08);border-radius:18px;padding:1.5rem 1.6rem;box-shadow:0 18px 40px rgba(15,23,42,0.12);text-align:left;transition:transform .25s ease, box-shadow .25s ease}
.feature-card:hover{transform:translateY(-4px);box-shadow:0 24px 42px rgba(15,23,42,0.14)}
.feature-icon{display:inline-flex;width:44px;height:44px;border-radius:14px;background:#1F2937;color:#F8FAFC;align-items:center;justify-content:center;font-size:1.35rem;margin-bottom:1rem}
.feature-card h4{margin:0 0 .4rem;font-size:1.2rem;font-weight:700;color:#0f172a}
.feature-card p{margin:0;color:#42536a;font-size:.98rem;line-height:1.5}

.steps-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1.4rem;margin:3.4rem auto 0;max-width:1020px;padding:0 1rem}
.step-card{background:rgba(255,255,255,0.82);border:1px solid rgba(148,163,184,0.22);border-radius:18px;padding:1.5rem 1.6rem;box-shadow:0 12px 28px rgba(15,23,42,0.08);display:flex;flex-direction:column}
.step-card strong{display:flex;align-items:center;justify-content:center;width:48px;height:48px;border-radius:14px;background:#1f2937;color:#F8FAFC;font-size:1.05rem;font-weight:800;letter-spacing:.12em;margin-bottom:1rem}
.step-card .step-title{font-weight:700;font-size:1.1rem;color:#111827;letter-spacing:.01em;text-transform:none;margin-bottom:.65rem}
.step-card .step-desc{font-size:.95rem;line-height:1.6;color:#475569}
.step-card .step-desc .step-key{display:inline-block;padding:.18rem .5rem;border-radius:8px;background:#E9E1FF;color:#4338CA;font-weight:700;text-transform:uppercase;letter-spacing:.16em;font-size:.72rem;margin-right:.4rem}

/* ===== Shared dark page (Schema/Results) ===== */
.dark-root html, .dark-root body, .dark-root .appview-container, .dark-root .stApp{background:var(--canvas)!important}
.dark-root .main, .dark-root .block-container, .dark-root .stAppViewContainer, .dark-root .stMain{background:var(--canvas)!important}
.dark-root .stMarkdown, .dark-root p, .dark-root span, .dark-root div{color:var(--text)!important}
.dark-root .feature-card{background:var(--glass-dark);border:1px solid rgba(110,119,139,0.24);box-shadow:0 16px 34px rgba(0,0,0,0.35)}
.dark-root .step-card{background:rgba(26,29,34,0.82);border:1px solid rgba(71,81,99,0.36);box-shadow:0 16px 36px rgba(5,7,10,0.45)}
.dark-root .step-card strong{background:rgba(106,74,248,0.35);color:#E9E4FF}
.dark-root .step-card .step-title{color:#F2F4FA}
.dark-root .step-card .step-desc{color:var(--muted)}
.dark-root .step-card .step-desc .step-key{background:rgba(106,74,248,0.28);color:#CFC7FF}
.dark-root [data-testid="stChatMessage"] .stMarkdown{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08)}
.dark-root [data-testid="stChatInput"]{
    background:transparent!important;
}
.dark-root [data-testid="stChatInput"]::before{color:#F6F8FF;text-shadow:0 12px 28px rgba(0,0,0,0.6)}
.dark-root [data-testid="stChatInput"]::after{color:rgba(217,225,240,0.88)}
.dark-root [data-testid="stChatInput"] form{
    background:rgba(17,21,31,0.9)!important;
    border:1px solid rgba(103,116,144,0.3)!important;
    box-shadow:0 28px 60px rgba(0,0,0,0.55)!important;
    min-height:124px!important;
}
.dark-root [data-testid="stChatInput"] textarea{
    background:rgba(11,15,26,0.92)!important;
    color:#E8ECF7!important;
    border:1px solid rgba(124,136,165,0.3)!important;
}
.dark-root [data-testid="stChatInput"] textarea::placeholder{color:rgba(182,192,214,0.75)!important}
.dark-root [data-testid="stChatInput"] div[role="button"]{background:var(--accent)!important}
.dark-root .section-title, .dark-root .section-subtitle{color:var(--muted)}

.dark-root .stButton > button{background:var(--panel);border:1px solid var(--panel-br);color:var(--text);border-radius:12px;font-weight:700}
.dark-root .stButton > button:hover{background:#333945}
.dark-root .stDownloadButton button{background:var(--accent);color:#fff;border-radius:12px;border:none;font-weight:700}
.dark-root .stDownloadButton button:hover{filter:brightness(1.05)}

/* Chips & schema text */
.chip{display:inline-block;background:#1C4532;color:#D1FAE5;border-radius:8px;padding:.18rem .5rem;margin:.16rem .22rem 0 0;
  font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.92rem}
.kv{color:var(--muted)}

/* Chat area */
.chat-wrap{max-width:920px;margin:1rem auto 0;padding-bottom:.4rem}
[data-testid="stChatMessage"] .stMarkdown{
    background:rgba(15,23,42,0.05);border:1px solid rgba(15,23,42,0.10);
    border-radius:14px;padding:.85rem 1rem
}
footer,[data-testid="stStatusWidget"],[data-testid="stBottomBlockContainer"],[data-testid="stFooter"],
footer::before,footer::after,[data-testid="stFooter"]::before,[data-testid="stFooter"]::after{
    background:transparent!important;box-shadow:none!important;border:none!important;border-top:none!important;margin:0!important;padding:0!important;
}
[data-testid="stFooter"], footer{height:0!important;min-height:0!important;overflow:hidden!important}
[data-testid="stFooter"] *, footer *{display:none!important}
[data-testid="stDecoration"], .stAppDeployButton{display:none!important}
[data-testid="stBottomBlockContainer"]{padding-bottom:.6rem!important;background:transparent!important;border:none!important;box-shadow:none!important}
[data-testid="stChatInput"]{
    position:relative;
    margin:1.85rem auto 0!important;
    max-width:900px;
    padding-top:2.9rem;
    background:transparent!important;
    border:none!important;
    outline:none!important;
}
[data-testid="stChatInput"]::before{
    content:"Ask me about the dataset";
    position:absolute;
    top:-.45rem;
    left:0;
    font-size:1.15rem;
    font-weight:800;
    letter-spacing:.02em;
    color:#F7F9FF;
    text-shadow:0 10px 26px rgba(9,11,17,0.55);
}
[data-testid="stChatInput"]::after{
    content:"Probe column definitions, value ranges, outliers, or missing patterns before you run a query.";
    position:absolute;
    top:1.3rem;
    left:0;
    right:0;
    font-size:.92rem;
    line-height:1.45;
    color:rgba(226,232,247,0.88);
}
[data-testid="stChatInput"] form{
    display:flex;
    gap:.85rem;
    align-items:flex-end;
    padding:1.2rem 1.4rem;
    border-radius:22px;
    background:rgba(32,36,45,0.9)!important;
    border:1px solid rgba(164,178,206,0.18)!important;
    box-shadow:0 28px 60px rgba(5,7,12,0.52)!important;
    backdrop-filter:blur(18px);
    outline:none!important;
    transition:none!important;
    min-height:124px!important;
}
[data-testid="stChatInput"] form > div:first-child{flex:1}
[data-testid="stChatInput"] textarea{
    background:rgba(28,32,41,0.92)!important;color:#F2F5FF!important;border:1px solid rgba(160,174,205,0.28)!important;border-radius:16px!important;
    box-shadow:none!important;outline:none!important;padding:.95rem 1.1rem!important;font-size:1.04rem!important;min-height:86px!important;line-height:1.45!important;
}
[data-testid="stChatInput"] textarea:focus{border:1px solid rgba(160,174,205,0.42)!important;box-shadow:0 0 0 2px rgba(106,74,248,0.2)!important}
[data-testid="stChatInput"] textarea::placeholder{color:rgba(198,206,224,0.75)!important}
[data-testid="stChatInput"] div[role="button"]{background:var(--accent)!important;color:#fff!important;border-radius:16px!important;height:52px!important;width:52px!important;box-shadow:0 16px 30px rgba(106,74,248,0.35)!important}
[data-testid="stChatInput"] div[role="button"] svg{width:22px;height:22px}

/* ===== Sticky top utility bar on Results ===== */
.sticky-top{
  position: sticky; top: 0; z-index: 999;
  background: linear-gradient(180deg, rgba(14,17,22,.98), rgba(38,43,51,.98));
    backdrop-filter: blur(6px);
    border-bottom: 1px solid #333944;
    padding: .72rem .45rem .7rem .45rem;
    margin: -1rem 0 1.2rem 0;
}
.util-row{display:flex; gap:.65rem; align-items:center; flex-wrap:wrap}
.util-row .btn{height:42px;padding:0 16px;border:1px solid var(--panel-br);border-radius:12px;background:var(--panel);color:var(--text);font-weight:800}
.util-row .btn:hover{background:#333945}
.util-row .spacer{flex:1 1 auto}
.util-row .question{padding:.48rem .9rem;border:1px solid #313845;border-radius:14px;background:#141821;color:var(--text);font-weight:600;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

.question-banner{max-width:1040px;margin:1.2rem auto 1.2rem}
.question-chip{display:block;padding:.65rem 1.15rem;border:1px solid rgba(74,84,104,0.55);border-radius:16px;background:rgba(20,24,32,0.92);box-shadow:0 18px 34px rgba(8,10,17,0.38);color:var(--text);font-weight:600;line-height:1.5}
.question-chip strong{display:inline-block;margin-right:.6rem;font-size:.78rem;letter-spacing:.26em;text-transform:uppercase;color:var(--muted)}

.dark-root .stButton > button.btn-ghost{background:transparent;border:1px solid rgba(103,112,132,0.48);color:var(--muted);font-weight:600}
.dark-root .stButton > button.btn-ghost:hover{border-color:var(--accent);color:var(--text)}

/* Tabs look native, but tighten spacing */
.block-container .stTabs [data-baseweb="tab-list"]{gap:.9rem;padding-bottom:.25rem}
.block-container .stTabs [data-baseweb="tab"]{padding:.45rem 1.5rem;border-radius:12px 12px 0 0}
.block-container .stTabs [data-baseweb="tab"] button{font-weight:700;letter-spacing:.02em}

.chart-controls{background:rgba(34,39,47,0.75);border:1px solid rgba(72,80,96,0.65);border-radius:14px;padding:1rem 1.2rem;margin-bottom:1rem}
.chart-controls .stSelectbox, .chart-controls .stRadio{margin-bottom:0!important}

.dark-root [data-testid="stFileUploader"]{background:rgba(20,24,32,0.94);border:1px solid rgba(86,96,118,0.52);border-radius:18px;padding:1.6rem 1.8rem;margin:1.8rem auto;max-width:760px;box-shadow:0 22px 46px rgba(0,0,0,0.45)}
[data-testid="stFileUploader"]{background:rgba(24,28,36,0.94);border:1px solid rgba(102,112,133,0.46);border-radius:18px;padding:1.6rem 1.8rem;margin:1.8rem auto;max-width:760px;box-shadow:0 20px 48px rgba(15,18,28,0.28);color:#E8ECF7}
[data-testid="stFileUploader"] label{font-weight:700;color:#F2F5FF}
[data-testid="stFileUploader"] p{color:rgba(226,232,245,0.82)!important}
.dark-root [data-testid="stFileUploader"] label{color:#F2F5FF}
.dark-root [data-testid="stFileUploader"] p{color:rgba(226,232,245,0.82)!important}

.metric-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin:1.6rem auto 2.6rem;max-width:980px}
.metric-card{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:1.1rem 1.3rem;backdrop-filter:blur(14px);box-shadow:0 16px 34px rgba(0,0,0,0.25);display:flex;flex-direction:column;gap:.4rem;min-height:120px}
.metric-label{font-size:.78rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);font-weight:700}
.metric-value{font-size:1.85rem;font-weight:800;color:var(--text)}
.metric-desc{color:var(--muted);font-size:.92rem;line-height:1.35}

.frame{background:var(--panel);border:1px solid var(--panel-br);border-radius:14px;padding:24px 26px;box-shadow:0 20px 48px rgba(0,0,0,0.38);margin-bottom:1.6rem}
.frame h3{margin:.1rem 0 1rem}
.frame.frame--glass{background:rgba(39,44,53,0.85);border:1px solid rgba(88,98,118,0.45);backdrop-filter:blur(10px)}
.frame.frame--light{background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12)}

.dark-root .stDownloadButton button{border-radius:12px;padding:.55rem 1.2rem;font-weight:700}
.dark-root .stDownloadButton button svg{margin-right:.5rem}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

DARK_OVERRIDE = """
<style>
html, body, .appview-container, .stApp{background:#0E1116!important}
.main, .block-container, .stAppViewContainer, .stMain{background:#262B33!important}
</style>
"""

# ------------------------------------------------------------
# UI helper data & builders
# ------------------------------------------------------------

FEATURE_CARDS = [
    (
        "🎯",
        "Ask anything in plain English",
        "Pose questions exactly as you would to an analyst. The copilot translates every prompt into safe, reviewable SQL.",
    ),
    (
        "🧾",
        "See the SQL before it runs",
        "Inspect, tweak, and export the generated query in a single click so you can reuse it elsewhere with confidence.",
    ),
    (
        "📊",
        "Discover insights instantly",
        "Summaries, tables, and charts are arranged side-by-side so you can move from raw data to visuals without leaving the page.",
    ),
]

WORKFLOW_STEPS = [
    (
        "Upload or use sample data",
        "<span class='step-key'>Data</span> Start with the bundled Telco dataset or bring your own CSV—columns are auto-cleaned for analysis.",
    ),
    (
        "Chat about the schema",
        "<span class='step-key'>Schema</span> Ask about column meanings, data quality, and distributions to understand the landscape before querying.",
    ),
    (
        "Run SQL and iterate",
        "<span class='step-key'>Iterate</span> Generate SQL, review results, download tables, and pivot to new follow-up questions within the same flow.",
    ),
]


def _humanize_number(value):
    if value is None:
        return "–"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    abs_num = abs(num)
    if abs_num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    if abs_num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    if abs_num >= 1_000:
        return f"{num / 1_000:.1f}K"
    if num.is_integer():
        return f"{int(num):,}"
    return f"{num:,.2f}"


def _build_metric_cards(metrics: list[tuple[str, str, str]]) -> str:
    cards = [
        (
            "<div class=\"metric-card\">"
            f"<span class=\"metric-label\">{label}</span>"
            f"<span class=\"metric-value\">{value}</span>"
            f"<span class=\"metric-desc\">{desc}</span>"
            "</div>"
        )
        for label, value, desc in metrics
    ]
    return "<div class=\"metric-cards\">" + "".join(cards) + "</div>"


def dataset_summary_html(df: pd.DataFrame, label: str | None = None) -> str:
    rows, cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    missing_vals = int(df.isna().sum().sum())
    duplicates = None
    if rows <= 200_000:
        duplicates = int(df.duplicated().sum())

    metrics: list[tuple[str, str, str]] = [
        ("Rows", _humanize_number(rows), f"Records{' in ' + label if label else ''}"),
        ("Columns", _humanize_number(cols), "Available fields"),
        ("Missing values", _humanize_number(missing_vals), "Across the entire dataset"),
    ]

    if duplicates is not None:
        metrics.append(("Duplicate rows", _humanize_number(duplicates), "Potential duplicates"))

    metrics.append(("Approx. memory", f"{memory_mb:.2f} MB", "Currently loaded in session"))
    return _build_metric_cards(metrics)


def feature_cards_html() -> str:
    cards = [
        "<div class=\"feature-card\">"
        f"<div class=\"feature-icon\">{icon}</div>"
        f"<h4>{title}</h4>"
        f"<p>{copy}</p>"
        "</div>"
        for icon, title, copy in FEATURE_CARDS
    ]
    return "<div class=\"feature-grid\">" + "".join(cards) + "</div>"


def steps_cards_html() -> str:
    cards = []
    for idx, (title, description) in enumerate(WORKFLOW_STEPS, start=1):
        cards.append(
            "<div class=\"step-card\">"
            f"<strong>{idx:02d}</strong>"
            f"<span class=\"step-title\">{title}</span>"
            f"<span class=\"step-desc\">{description}</span>"
            "</div>"
        )
    return "<div class=\"steps-grid\">" + "".join(cards) + "</div>"

# ------------------------------------------------------------
# LLM + SQL helpers
# ------------------------------------------------------------
SYSTEM_PROMPT = """You are a data analyst who writes safe, valid SQLite SQL queries.

Rules:
- Output ONLY a SQL query, no explanations or prose.
- Use ONLY tables and columns that exist in the schema provided to you. Do not invent names.
- Generate a SINGLE SELECT statement (no multiple statements).
- NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, PRAGMA, REINDEX, or comments.
- If joins are needed, write explicit joins using the tables in the schema.
- If the user asks for a rate/percentage, compute it with COUNTs and CAST/ROUND.
- When reasonable, add ORDER BY and LIMIT.
"""

def extract_sql(text: str) -> str:
    m = re.search(r"```sql\s*(.*?)```", text, flags=re.S | re.I)
    if m: return m.group(1).strip()
    return text.strip()

def is_safe_sql(sql: str) -> bool:
    s = sql.strip()
    if s.endswith(";"): s = s[:-1].strip()
    low = s.lower()
    if not low.startswith("select"): return False
    if ";" in s: return False
    banned = ["insert","update","delete","drop","alter","create","attach","pragma","reindex","--","/*","*/"]
    return not any(b in low for b in banned)

def ensure_limit(sql: str, n: int = 200) -> str:
    s = sql.strip().rstrip(";")
    if re.search(r"\blimit\b", s, flags=re.I): return s
    return f"{s}\nLIMIT {n}"

def set_query_timeout(conn, seconds: float = 5.0):
    start = time.time()
    def abort_if_slow(): return 1 if (time.time() - start) > seconds else 0
    conn.set_progress_handler(abort_if_slow, 10000)

def clear_query_timeout(conn):
    conn.set_progress_handler(None, 0)

def get_full_schema(conn) -> str:
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()]
    if not tables: return "No tables found."
    lines = []
    for t in tables:
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({t})")]
        lines.append(f"Table `{t}` columns: " + ", ".join(cols))
    return "\n".join(lines)

def llm_generate_sql(question: str, schema_text: str) -> str:
    if not OPENAI_API_KEY or OpenAI is None:
        return "SELECT * FROM uploaded_csv LIMIT 50"
    client = OpenAI(api_key=OPENAI_API_KEY)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{schema_text}\n\nUser question: {question}"},
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0.2,
        )
        return extract_sql(resp.choices[0].message.content)
    except Exception as e:
        safe_err = str(e).replace("'", "''")
        return f"SELECT 'LLM error' AS error, '{safe_err}' AS detail"

def try_run_sql_with_repair(sql: str, conn, schema_text: str, max_retries: int = 2):
    attempt_sql, last_err, repairs = sql, None, 0
    for _ in range(max_retries + 1):
        try:
            set_query_timeout(conn, 5.0)
            df = pd.read_sql_query(attempt_sql, conn)
            clear_query_timeout(conn)
            return df, attempt_sql, None, repairs
        except Exception as e:
            clear_query_timeout(conn)
            last_err = str(e)
            if repairs >= max_retries or not (OPENAI_API_KEY and OpenAI is not None):
                break
            fixed = llm_generate_sql(
                "The following SELECT-only SQL failed. Return ONLY a corrected SELECT query for SQLite.\n\n"
                f"Schema:\n{schema_text}\n\nOriginal SQL:\n{attempt_sql}\n\nError:\n{last_err}",
                schema_text
            )
            attempt_sql = extract_sql(fixed).strip()
            if not is_safe_sql(attempt_sql): break
            attempt_sql = ensure_limit(attempt_sql, 200)
            repairs += 1
    return pd.DataFrame(), attempt_sql, last_err, repairs

def generate_insight_paragraph(df: pd.DataFrame) -> str | None:
    if df is None or df.empty or not (OPENAI_API_KEY and OpenAI is not None):
        return None
    client = OpenAI(api_key=OPENAI_API_KEY)
    sample_csv = df.head(50).to_csv(index=False)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a concise analytics writer for business stakeholders."},
                {"role": "user", "content":
                    "Write a brief (2–3 sentences) summary of key insights in this result table. "
                    "Be factual and avoid speculation. Mention standout categories or large differences. "
                    "Here is a CSV sample of the results:\n\n" + sample_csv}
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

# ------------------------------------------------------------
# CSV -> SQLite utility
# ------------------------------------------------------------
def build_sqlite_from_df(df: pd.DataFrame, table_name: str = "uploaded_csv"):
    conn = sqlite3.connect(":memory:")
    df2 = df.copy()
    df2.columns = [
        c.strip().replace(" ", "_").replace("-", "_").replace("/", "_")
        for c in df2.columns
    ]
    df2.to_sql(table_name, conn, index=False, if_exists="replace")
    return conn, table_name

def schema_html(df: pd.DataFrame, table_name: str) -> str:
    rows, cols = df.shape
    chips = " ".join([f'<span class="chip">{c}</span>' for c in df.columns])
    return (
        f'<p class="kv"><b>Table name:</b> <code>{table_name}</code></p>'
        f'<p class="kv"><b>Columns ({cols}):</b></p>{chips}'
        f'<p class="kv" style="margin-top:.6rem"><b>Number of rows and columns:</b> {rows} rows × {cols} columns</p>'
    )

# ------------------------------------------------------------
# State init
# ------------------------------------------------------------
for k, v in {
    "mode": "home",
    "upload_df": None,
    "uploaded_csv": None,
    "schema_chat": [],
    "show_uploader": False,
    "ready_to_continue": False,
    "last_question": None,
    "final_sql": None,
    "results_df": None,
    "auto_insights": None,
    "results_tab": "Summary",
}.items():
    if k not in st.session_state: st.session_state[k] = v

# ------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------
def hero():
    st.markdown("""
      <div class="hero-wrap">
        <div class="hero">
          <span class="title-line1">AI POWERED</span>
          <span class="title-line2">SQL QUERY GENERATOR</span>
          <p class="subtitle">
            Ask a question in plain English. I’ll generate SELECT-only SQL for your uploaded data or the bundled Telco database and run it.
          </p>
        </div>
      </div>
    """, unsafe_allow_html=True)

def apply_dark():
    st.markdown(DARK_OVERRIDE, unsafe_allow_html=True)

def reset_chat_and_results():
    st.session_state.last_question = None
    st.session_state.final_sql = None
    st.session_state.results_df = None
    st.session_state.auto_insights = None
    st.session_state.results_tab = "Summary"
    st.session_state.schema_chat = []
    st.session_state.mode = "schema"

# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
def page_home():
    hero()
    st.markdown('<div class="cta">', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        if st.button("Use sample Telco CSV", use_container_width=True, key="home_default"):
            if DEFAULT_CSV_PATH.exists():
                try:
                    df = pd.read_csv(DEFAULT_CSV_PATH)
                    st.session_state.upload_df = df
                    st.session_state.uploaded_csv = DEFAULT_CSV_PATH.name
                    reset_chat_and_results()
                    st.session_state.mode = "schema"
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load default CSV: {e}")
            else:
                st.error(f"Default CSV not found at {DEFAULT_CSV_PATH}")

    with col2:
        if st.button("Import your CSV", use_container_width=True, key="home_import"):
            st.session_state.show_uploader = True
            st.session_state.ready_to_continue = False
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.show_uploader:
        st.info("Large files may take a few seconds while we cache a temporary SQLite database.")
        up = st.file_uploader("Choose a CSV file", type=["csv"], key="home_uploader")
        if up is not None:
            try:
                df = pd.read_csv(up)
                st.success(f"Uploaded: {up.name}")
                st.session_state.upload_df = df
                st.session_state.uploaded_csv = up.name
                st.session_state.ready_to_continue = True
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

        action_cols = st.columns([0.22, 0.22, 0.56])
        with action_cols[0]:
            if st.session_state.ready_to_continue and st.button("Continue →", key="continue_to_schema"):
                st.session_state.show_uploader = False
                st.session_state.ready_to_continue = False
                reset_chat_and_results()
                st.session_state.mode = "schema"
                st.rerun()
        with action_cols[1]:
            if st.button("Cancel", key="cancel_upload", type="secondary"):
                st.session_state.show_uploader = False
                st.session_state.ready_to_continue = False
                st.session_state.upload_df = None
                st.session_state.uploaded_csv = None
    else:
        st.markdown('<div class="section-title">Why teams love this copilot</div>', unsafe_allow_html=True)
        st.markdown(feature_cards_html(), unsafe_allow_html=True)
        st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Three quick steps to go from question to insight.</div>', unsafe_allow_html=True)
        st.markdown(steps_cards_html(), unsafe_allow_html=True)

def page_schema():
    apply_dark()

    nav_home, nav_reset, nav_meta = st.columns([0.18, 0.22, 0.60])
    with nav_home:
        if st.button("← Home", key="schema_home", use_container_width=True, help="Back to Home", type="secondary"):
            st.session_state.mode = "home"
            st.rerun()
    with nav_reset:
        if st.button("Reset conversation", key="schema_reset", use_container_width=True, help="Clear previous chat", type="secondary"):
            reset_chat_and_results()
            st.rerun()
    with nav_meta:
        dataset_name = st.session_state.get("uploaded_csv") or "uploaded_csv"
        st.markdown(f"<div class='breadcrumb'>Dataset: <span>{dataset_name}</span></div>", unsafe_allow_html=True)

    df = st.session_state.upload_df
    if df is None:
        st.warning("No data loaded. Choose **Use default .csv** or **Import .csv** on Home.")
        return

    raw_name = st.session_state.get("uploaded_csv") or "uploaded_csv"
    table_name = Path(raw_name).stem

    st.markdown(dataset_summary_html(df, table_name), unsafe_allow_html=True)
    st.markdown(f"<div class='frame frame--glass'>{schema_html(df, table_name)}</div>", unsafe_allow_html=True)

    preview_toggle = st.checkbox("Preview first 10 rows", value=False, key="schema_preview")
    if preview_toggle:
        st.dataframe(df.head(10), use_container_width=True, height=360)
        st.caption("Preview capped to 10 rows for quick validation.")

    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

    for m in st.session_state.schema_chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    q = st.chat_input("Ask about columns, types, unique values, missingness…")
    if q:
        st.session_state.schema_chat.append({"role": "user", "content": q})

        conn, tbl = build_sqlite_from_df(df, table_name)
        schema_text = get_full_schema(conn)
        gen = llm_generate_sql(q, schema_text)
        cand = extract_sql(gen)
        if not is_safe_sql(cand):
            st.error("Generated SQL was blocked for safety (must be a single SELECT).")
            return
        cand = ensure_limit(cand, 200)

        out_df, final_sql, last_err, repairs = try_run_sql_with_repair(cand, conn, schema_text, 2)
        conn.close()

        st.session_state.last_question = q
        st.session_state.final_sql = final_sql
        st.session_state.results_df = out_df
        st.session_state.auto_insights = generate_insight_paragraph(out_df)
        st.session_state.results_tab = "Summary"
        st.session_state.mode = "results"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

def page_results():
    apply_dark()

    nav_home, nav_new_chat, nav_spacer = st.columns([0.18, 0.18, 0.64])
    with nav_home:
        if st.button("← Home", key="results_home", use_container_width=True, help="Back to Home", type="secondary"):
            st.session_state.mode = "home"
            st.rerun()
    with nav_new_chat:
        if st.button("New chat", key="results_new_chat", use_container_width=True, help="Start fresh", type="secondary"):
            reset_chat_and_results()
            st.rerun()

    # ===== Tabs: Summary / SQL / Table / Chart
    df = st.session_state.results_df
    if df is None:
        st.warning("No results yet. Ask a question from the Schema page.")
        return

    st.markdown(dataset_summary_html(df, "result set"), unsafe_allow_html=True)

    if st.session_state.last_question:
        safe_question = html.escape(st.session_state.last_question)
        st.markdown(
            "<div class='question-banner'><div class='question-chip'><strong>Question</strong>"
            f"<span>{safe_question}</span></div></div>",
            unsafe_allow_html=True,
        )

    tab_summary, tab_sql, tab_table, tab_chart = st.tabs(["Summary", "SQL Query", "Table", "Chart"])

    with tab_summary:
        st.markdown("### Summary")
        if st.session_state.auto_insights:
            st.write(st.session_state.auto_insights)
        else:
            st.write("This section summarizes the intent of your question and the shape of the result set.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_sql:
        st.markdown("### SQL Query")
        final_sql = st.session_state.final_sql or "-- (empty) --"
        st.code(final_sql, language="sql")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_table:
        st.markdown("### Results")
        table_height = min(520, max(320, 38 + 28 * len(df))) if not df.empty else 240
        st.dataframe(df, use_container_width=True, height=table_height)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_chart:
        st.markdown("### Chart")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            x_options = ["Row index"] + [col for col in df.columns if col != numeric_cols[0]]
            ctrl_type, ctrl_x, ctrl_y = st.columns([0.24, 0.38, 0.38])
            with ctrl_type:
                chart_type = st.radio("Chart type", ["Line", "Bar", "Area"], horizontal=True, key="chart_type")
            with ctrl_x:
                x_axis = st.selectbox("X-axis", x_options, key="chart_x")
            with ctrl_y:
                y_axis = st.selectbox("Metric", numeric_cols, key="chart_y")

            if x_axis == "Row index":
                plot_df = df[[y_axis]].reset_index().rename(columns={"index": "Row"}).set_index("Row")
            else:
                x_series = df[x_axis]
                if pd.api.types.is_numeric_dtype(x_series) or pd.api.types.is_datetime64_any_dtype(x_series):
                    plot_df = df[[x_axis, y_axis]].dropna().set_index(x_axis).sort_index()
                else:
                    plot_df = df[[x_axis, y_axis]].dropna().groupby(x_axis, as_index=False)[y_axis].mean().set_index(x_axis)

            if plot_df.empty:
                st.info("Not enough data to visualise with the current selection.")
            else:
                if chart_type == "Line":
                    st.line_chart(plot_df, use_container_width=True)
                elif chart_type == "Bar":
                    st.bar_chart(plot_df, use_container_width=True)
                else:
                    st.area_chart(plot_df, use_container_width=True)
            st.caption("Adjust the axes to explore different relationships.")
        else:
            st.info("No numeric columns detected to plot.")
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Router
# ------------------------------------------------------------
mode = st.session_state.mode
if mode == "home":
    page_home()
elif mode == "schema":
    page_schema()
elif mode == "results":
    page_results()
else:
    page_home()
