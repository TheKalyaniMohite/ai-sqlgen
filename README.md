# AI-Powered SQL Query Generator (Streamlit + OpenAI)

Natural language → safe, single-statement SELECT SQL → results table + CSV download + quick chart.  
Works with your uploaded CSVs or the bundled Telco churn database (SQLite).

Tech:Python, Streamlit, SQLite, pandas, OpenAI API.  
Safety:SELECT-only guardrails, banned keywords, row LIMIT, self-repair on SQL errors.

Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
