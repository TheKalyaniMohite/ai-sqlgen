import os
import re
import time
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---- LLM client (OpenAI) ----
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ------------- Config -------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DB_PATH = Path("data/telco.db")
DEFAULT_DB_URI = "file:data/telco.db?mode=ro"  # read-only when using the bundled telco db
TABLE = "customers"  # used only by the fallback heuristics when no API key is present

# ------------- Helpers -------------
def get_full_schema(conn) -> str:
    """Return a descriptive schema of ALL user tables and their columns."""
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]
    if not tables:
        return "No tables found."

    lines = []
    for t in tables:
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({t})")]
        lines.append(f"Table `{t}` columns: " + ", ".join(cols))
    return "\n".join(lines)


def set_query_timeout(conn, seconds: float = 5.0):
    """Abort long-running queries after `seconds` using SQLite progress handler."""
    start = time.time()

    def abort_if_slow():
        return 1 if (time.time() - start) > seconds else 0  # non-zero aborts

    conn.set_progress_handler(abort_if_slow, 10000)  # check every ~10k ops


def clear_query_timeout(conn):
    """Remove any active progress handler."""
    conn.set_progress_handler(None, 0)


def try_run_sql_with_repair(sql: str, conn, schema_text: str, max_retries: int = 2):
    """
    Try to execute SQL. On error, ask the LLM to fix it using the error + schema, and retry.
    Returns: (df, final_sql, last_error_or_None, num_repairs_done)
    """
    attempt_sql = sql
    last_err = None
    repairs = 0

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

            # Ask LLM to produce a corrected SELECT
            fix_prompt = (
                "The following SELECT-only SQL failed. "
                "Return ONLY a corrected SELECT query for SQLite.\n\n"
                f"Schema:\n{schema_text}\n\n"
                f"Original SQL:\n{attempt_sql}\n\n"
                f"Error:\n{last_err}"
            )
            fixed = llm_generate_sql(fix_prompt, schema_text)
            attempt_sql = extract_sql(fixed).strip()
            # Keep safety
            if not is_safe_sql(attempt_sql):
                break
            attempt_sql = ensure_limit(attempt_sql, 200)
            repairs += 1

    return pd.DataFrame(), attempt_sql, last_err, repairs


def generate_insight_paragraph(df: pd.DataFrame) -> str | None:
    """Ask the LLM to write a short, business-friendly summary of the results."""
    if df is None or df.empty or not (OPENAI_API_KEY and OpenAI is not None):
        return None

    sample_csv = df.head(50).to_csv(index=False)
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a concise analytics writer for business stakeholders."},
                {
                    "role": "user",
                    "content": (
                        "Write a brief (2–3 sentences) summary of key insights in this result table. "
                        "Be factual and avoid speculation. Mention standout categories or large differences. "
                        "Here is a CSV sample of the results:\n\n" + sample_csv
                    ),
                },
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


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
    """Accept plain SQL or fenced code blocks."""
    m = re.search(r"```sql\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    return text.strip()


def is_safe_sql(sql: str) -> bool:
    """Allow a single SELECT statement. Block DDL/DML and multi-statements."""
    s = sql.strip()
    # allow a single trailing semicolon if present
    if s.endswith(";"):
        s = s[:-1].strip()
    low = s.lower()

    # Must start with SELECT
    if not low.startswith("select"):
        return False

    # Block additional semicolons (multi-statements)
    if ";" in s:
        return False

    # Block dangerous keywords
    banned = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "attach",
        "pragma",
        "reindex",
        "--",
        "/*",
        "*/",
    ]
    return not any(b in low for b in banned)


def ensure_limit(sql: str, n: int = 200) -> str:
    s = sql.strip().rstrip(";")
    if re.search(r"\blimit\b", s, flags=re.I):
        return s
    return f"{s}\nLIMIT {n}"


def llm_generate_sql(question: str, schema_text: str) -> str:
    """Generate SQL via OpenAI or simple fallbacks if no key."""
    if not OPENAI_API_KEY or OpenAI is None:
        # Fallback heuristics for a couple of common intents (assumes telco customers table)
        ql = question.lower()
        if "contract" in ql and "rate" in ql:
            return """
            SELECT contract,
                   SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churn_yes,
                   COUNT(*) AS total,
                   ROUND(100.0 * SUM(CASE WHEN churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
            FROM customers
            GROUP BY contract
            ORDER BY churn_rate_pct DESC
            LIMIT 200
            """.strip()
        if "payment" in ql:
            return """
            SELECT paymentmethod,
                   SUM(CASE WHEN churn='Yes' THEN 1 ELSE 0 END) AS churn_yes,
                   COUNT(*) AS total,
                   ROUND(100.0 * SUM(CASE WHEN churn='Yes' THEN 1 ELSE 0 END)/COUNT(*), 2) AS churn_rate_pct
            FROM customers
            GROUP BY paymentmethod
            ORDER BY churn_rate_pct DESC
            LIMIT 200
            """.strip()
        # Basic fallback
        return f"SELECT * FROM {TABLE} LIMIT 50"

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
        text = resp.choices[0].message.content
        return extract_sql(text)
    except Exception as e:
        # Return a harmless query with the error in a column so app doesn't crash
        safe_err = str(e).replace("'", "''")
        return f"SELECT 'LLM error' AS error, '{safe_err}' AS detail"


# ------------- UI -------------
st.set_page_config(page_title="AI SQL Generator", layout="wide")
st.title("🧠➡️🗃️ AI-Powered SQL Query Generator")
st.caption(
    "Ask a question in plain English. I’ll generate **SELECT-only** SQL for your uploaded data "
    "or the bundled Telco database and run it."
)

# --- Choose data source: uploads take precedence; else use bundled telco.db (read-only) ---
uploaded = st.file_uploader(
    "Upload CSV file(s) to analyze (optional)", type=["csv"], accept_multiple_files=True
)

conn = None
active_source = "default-db"
if uploaded:
    # Build an in-memory SQLite DB from uploaded CSVs
    conn = sqlite3.connect(":memory:")
    for uf in uploaded:
        df_up = pd.read_csv(uf)
        # simple snake_case normalization for column names
        df_up.columns = [
            c.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
            for c in df_up.columns
        ]
        table_name = Path(uf.name).stem.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        df_up.to_sql(table_name, conn, index=False, if_exists="replace")
    active_source = "uploaded-csvs"
else:
    # Fallback to your bundled telco DB (read-only)
    if not DB_PATH.exists():
        st.error("No database found. Either upload one or ensure data/telco.db exists (run init_db.py).")
        st.stop()
    conn = sqlite3.connect(DEFAULT_DB_URI, uri=True)
    active_source = "telco-db"

st.caption(f"Data source: **{active_source}**")

# Use the full multi-table schema (works for both upload and telco-db)
schema_text = get_full_schema(conn)
with st.expander("See table schema"):
    st.code(schema_text)

examples = [
    "Show churn rate by contract type",
    "Breakdown of churn by payment method",
    "Average monthly charges for churned vs not churned customers",
    "Churn rate by tenure buckets (0-12, 13-24, 25-48, 49+)",
]
st.write("**Examples:**")
st.write(" • " + "\n • ".join(examples))

question = st.text_input("Your question", placeholder="e.g., Show churn rate by contract type")
run = st.button("Generate SQL and Run")

# ---------- Run block ----------
if run and question.strip():
    # 1) Generate SQL (or fallback)
    generated_sql = llm_generate_sql(question, schema_text)
    candidate_sql = extract_sql(generated_sql)

    # 2) Safety gate
    if not is_safe_sql(candidate_sql):
        st.error("Generated SQL was blocked for safety. It must be a single SELECT statement.")
        st.stop()

    # 3) Ensure a row cap, then try run with auto-repair
    candidate_sql = ensure_limit(candidate_sql, 200)
    df, final_sql, last_err, repairs = try_run_sql_with_repair(
        candidate_sql, conn, schema_text, max_retries=2
    )

    # 4) Show SQL actually executed
    st.subheader("Final SQL (after repairs)" if repairs else "Generated SQL")
    st.code(final_sql, language="sql")

    # 5) Results / errors
    if last_err and df.empty:
        st.info(f"Tried {repairs} repair attempt(s). Last error: {last_err}")

    st.subheader("Results")
    st.dataframe(df, use_container_width=True)
    st.caption(f"{len(df)} row(s) shown." if not df.empty else "No rows returned.")

    # 6) Download + Plot
    if not df.empty:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results as CSV", csv_bytes, file_name="results.csv", mime="text/csv")

        st.markdown("### Plot Results")
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c])]

        if num_cols and cat_cols:
            xcol = st.selectbox("X (category)", cat_cols, index=0, key="plot_x")
            ycol = st.selectbox("Y (numeric)", num_cols, index=0, key="plot_y")
            plot_df = df[[xcol, ycol]].groupby(xcol, as_index=False).sum()
            st.bar_chart(plot_df.set_index(xcol)[ycol], use_container_width=True)
        else:
            st.caption("Add a categorical and a numeric column to enable plotting.")

        # 7) Auto-insights
        insights = generate_insight_paragraph(df)
        if insights:
            st.markdown("### Auto-insights")
            st.write(insights)
