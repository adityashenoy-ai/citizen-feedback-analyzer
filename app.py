# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import math
from openai import OpenAI
import pydeck as pdk

# ------------- Streamlit page config -------------
st.set_page_config(page_title="AI Citizen Feedback Analyzer (Advanced)", layout="wide")
st.title("🇮🇳 AI Citizen Feedback Analyzer — Advanced")
st.markdown("NER · Root-cause modeling · Dashboards · Map · Actionable recommendations")

# ------------- Initialize OpenAI client -------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY not found in Streamlit Secrets. Add OPENAI_API_KEY before running.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------- Geo reference for states (lat, lon) -------------
STATE_COORDS = {
    "Karnataka": [12.9716, 77.5946],
    "Maharashtra": [19.0760, 72.8777],
    "Rajasthan": [26.9124, 75.7873],
    "Uttar Pradesh": [26.8467, 80.9462],
    "Tamil Nadu": [13.0827, 80.2707],
    "Delhi": [28.7041, 77.1025],
    "Kerala": [10.8505, 76.2711],
    "Bihar": [25.0961, 85.3131],
    "Gujarat": [23.0225, 72.5714],
    "West Bengal": [22.5726, 88.3639],
}

# ---------------- Utility functions ----------------
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # normalize columns we expect
    expected_cols = {c.lower(): c for c in df.columns}
    # try to rename common variations
    if "complaint_text" not in df.columns:
        for k in ["complaint", "text", "description", "complaint_text"]:
            if k in expected_cols:
                df = df.rename(columns={expected_cols[k]: "complaint_text"})
                break
    if "state" not in df.columns:
        for k in ["state", "province", "region"]:
            if k in expected_cols:
                df = df.rename(columns={expected_cols[k]: "state"})
                break
    if "district" not in df.columns and "district" in expected_cols:
        df = df.rename(columns={expected_cols["district"]: "district"})
    # fill missing columns
    if "district" not in df.columns:
        df["district"] = ""
    if "complaint_id" not in df.columns:
        df.insert(0, "complaint_id", range(1, len(df) + 1))
    return df

def safe_sample_texts(df: pd.DataFrame, max_items=200):
    """Return up to max_items complaint_texts joined as a JSON list string for LLM prompts"""
    texts = df["complaint_text"].dropna().astype(str).tolist()
    # reduce length if any text is extremely long
    sampled = texts[:max_items]
    return json.dumps([{"id": i+1, "text": sampled[i]} for i in range(len(sampled))], ensure_ascii=False)

def call_llm(prompt, model="gpt-4o-mini", temperature=0.2, max_tokens=1200):
    """Call the OpenAI LLM (chat completion) and return the assistant content."""
    # Keep retry logic
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=temperature
            )
            # new client returns structure with .choices list
            text = None
            # attempt several paths for compatibility
            if hasattr(resp, "choices") and len(resp.choices) > 0:
                # some SDKs return choices[0].message.content
                ch = resp.choices[0]
                if hasattr(ch, "message") and hasattr(ch.message, "content"):
                    text = ch.message.content
                elif isinstance(ch, dict) and "message" in ch and "content" in ch["message"]:
                    text = ch["message"]["content"]
                elif "text" in ch:
                    text = ch["text"]
            if text is None:
                # fallback
                text = str(resp)
            return text
        except Exception as e:
            sleep_t = 1 + 2*attempt
            time.sleep(sleep_t)
            last_exc = e
    st.error(f"LLM call failed after retries: {last_exc}")
    return None

# ---------------- LLM prompt builders ----------------
def build_ner_prompt(sample_json_list):
    prompt = f"""
You are an expert NLP system. Given a JSON array of complaint objects like:
{sample_json_list}

For each object, extract named entities and labels. Return one JSON object with these keys:
- entities: an array of objects {{id, issues: [top_issue_terms], organizations: [org names], locations: [location mentions], dates: [dates found], persons: [any names], severity: [low/medium/high if implied]}}
Follow strict JSON only, no explanation text.
Limit each arrays to top 5 items.
"""
    return prompt

def build_sentiment_prompt(sample_json_list):
    prompt = f"""
You are an expert sentiment analyzer for Indian citizen complaints. Given a JSON array of complaint objects:
{sample_json_list}

Return a JSON object with a sentiment summary:
- sentiments: [{{id, sentiment: "positive"/"neutral"/"negative", score: float_between_0_1}}]
Also return overall_counts: {{positive: int, neutral: int, negative: int}}
Return strict JSON only.
"""
    return prompt

def build_rootcause_prompt(cluster_examples_json, cluster_name):
    prompt = f"""
You are a policy consultant. Given these example complaints in cluster '{cluster_name}':
{cluster_examples_json}

1) Provide top 5 root causes (short phrases).  
2) For each root cause provide: impact (short), probability (high/medium/low), recommended actions (3 bullets), quick KPIs to track improvement.

Return strict JSON: {{ cluster: "<name>", root_causes: [ {{cause: "...", impact: "...", probability: "...", actions: ["..."], kpis: ["..."]}} ] }}
"""
    return prompt

# ---------------- Streamlit layout ----------------
with st.sidebar:
    st.header("Upload / Settings")
    uploaded_file = st.file_uploader("Upload complaints CSV (columns: complaint_text, state, district, date)", type=["csv"])
    st.markdown("---")
    st.write("LLM Model")
    model = st.selectbox("Model", options=["gpt-4o-mini","gpt-4o"], index=0)
    max_sample = st.slider("LLM sample size (how many complaint rows to send to LLM)", 50, 500, 200, step=50)
    run_ner = st.checkbox("Run NER extraction (LLM)", value=True)
    run_sentiment = st.checkbox("Run Sentiment Analysis (LLM)", value=True)
    run_rootcause = st.checkbox("Run Root-Cause Modeling (LLM)", value=True)

# Main
if uploaded_file is None:
    st.info("Upload a CSV to begin. Use the sample_data_1000.csv if you uploaded earlier.")
    st.stop()

# Load
df = load_csv(uploaded_file)
st.success(f"Loaded {len(df)} complaints")

# Add lat/lon from state
df["state"] = df["state"].fillna("").astype(str)
df["district"] = df["district"].fillna("").astype(str)
df["lat"] = df["state"].apply(lambda s: STATE_COORDS.get(s, [None, None])[0])
df["lon"] = df["state"].apply(lambda s: STATE_COORDS.get(s, [None, None])[1])

# Top-level metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total complaints", len(df))
col2.metric("Unique states", df["state"].nunique())
col3.metric("Unique districts", df["district"].nunique())
col4.metric("Sample sent to LLM", min(max_sample, len(df)))

# ---------------- Dashboards ----------------
st.markdown("## Dashboard")

# complaints per state
state_counts = df["state"].value_counts().rename_axis("state").reset_index(name="count")
st.write("### Complaints by State")
st.bar_chart(state_counts.set_index("state")["count"])

# top issues (simple keyword frequency)
st.write("### Top words in complaints (quick view)")
# quick cleaning
top_words = (
    " ".join(df["complaint_text"].dropna().astype(str).str.lower().tolist())
    .replace(",", " ")
    .replace(".", " ")
    .split()
)
# remove short words
top_words = [w for w in top_words if len(w) > 3]
from collections import Counter
word_counts = Counter(top_words)
top_15 = word_counts.most_common(15)
top15_df = pd.DataFrame(top_15, columns=["word", "count"])
st.table(top15_df)

# interactive map (heatmap + scatter)
st.write("### Map: Complaint density (Heatmap) & Points")
map_df = df.dropna(subset=["lat", "lon"])
if len(map_df) > 0:
    # Hexagon / Heatmap layer
    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=map_df,
        get_position='[lon, lat]',
        aggregation='"MEAN"',
        threshold=0.3,
        radiusPixels=60
    )
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=map_df.sample(min(len(map_df), 1000)),
        get_position='[lon, lat]',
        get_radius=50000,
        pickable=True,
        auto_highlight=True
    )
    view_state = pdk.ViewState(latitude=22.9734, longitude=78.6569, zoom=4)
    deck = pdk.Deck(layers=[heat_layer, scatter], initial_view_state=view_state, tooltip={"text":"State: {state}\nDistrict: {district}\nComplaint: {complaint_text}"})
    st.pydeck_chart(deck)
else:
    st.warning("No geo data (state mapping) available for map.")

# ---------------- LLM-powered analysis ----------------
st.markdown("## AI Analysis (LLM-powered)")

sample_json = safe_sample_texts(df, max_items=max_sample)

# 1) NER
ner_results = None
if run_ner:
    st.write("Running NER extraction...")
    ner_prompt = build_ner_prompt(sample_json)
    ner_raw = call_llm(ner_prompt, model=model, temperature=0.0)
    try:
        # Expecting JSON output - try to find the JSON in the response
        ner_json = json.loads(ner_raw.strip())
        ner_results = ner_json.get("entities", ner_json)
        st.success("NER extraction parsed successfully.")
    except Exception as e:
        st.error("Failed to parse NER JSON from LLM. Showing raw output below.")
        st.code(ner_raw[:2000])

# 2) Sentiment
sentiment_results = None
if run_sentiment:
    st.write("Running sentiment analysis...")
    sent_prompt = build_sentiment_prompt(sample_json)
    sent_raw = call_llm(sent_prompt, model=model, temperature=0.0)
    try:
        sent_json = json.loads(sent_raw.strip())
        sentiment_results = sent_json
        st.success("Sentiment analysis parsed successfully.")
    except Exception as e:
        st.error("Failed to parse sentiment JSON from LLM. Showing raw output below.")
        st.code(sent_raw[:2000])

# visualize sentiment if parsed
if sentiment_results:
    try:
        overall = sentiment_results.get("overall_counts")
        if overall:
            st.write("### Sentiment distribution (LLM)")
            s_df = pd.DataFrame(list(overall.items()), columns=["sentiment","count"]).set_index("sentiment")
            st.bar_chart(s_df)
    except Exception:
        pass

# 3) Root-cause modeling: cluster complaints into top 4 clusters (using quick keyword clusters)
st.write("### Root-cause modeling")
# Very simple cluster: use top words to group; then call LLM per cluster for RCA
# create clusters by matching top 5 frequent words
keywords = [w for w, _ in top_15]
clusters = {}
for kw in keywords[:5]:
    mask = df["complaint_text"].str.lower().str.contains(kw, na=False)
    sample = df[mask].head(30)["complaint_text"].tolist()
    if len(sample) > 0:
        clusters[kw] = sample

rootcause_results = {}
if run_rootcause and len(clusters) > 0:
    st.write(f"Found {len(clusters)} clusters for RCA.")
    for cname, examples in clusters.items():
        st.write(f"Analyzing cluster: **{cname}** (examples: {len(examples)})")
        examples_json = json.dumps([{"text": e} for e in examples], ensure_ascii=False)
        root_prompt = build_rootcause_prompt(examples_json, cname)
        rc_raw = call_llm(root_prompt, model=model, temperature=0.2)
        try:
            rc_json = json.loads(rc_raw.strip())
            rootcause_results[cname] = rc_json
            st.success(f"RCA for cluster '{cname}' parsed.")
        except Exception:
            st.error(f"Failed to parse RCA JSON for cluster {cname}. Showing raw output:")
            st.code(rc_raw[:2000])

# show rootcause summary table
if rootcause_results:
    summary_rows = []
    for k, v in rootcause_results.items():
        rcs = v.get("root_causes", []) if isinstance(v, dict) else []
        for rc in rcs:
            summary_rows.append({
                "cluster": k,
                "cause": rc.get("cause")[:80] if isinstance(rc.get("cause"), str) else str(rc.get("cause")),
                "probability": rc.get("probability"),
                "actions": " | ".join(rc.get("actions", [])[:3]) if isinstance(rc.get("actions", list)) else ""
            })
    if summary_rows:
        st.write("#### Root Cause Summary")
        st.dataframe(pd.DataFrame(summary_rows))

# --------------- Downloadable reports ---------------
st.markdown("## Export & Reports")
if st.button("Generate executive report (Markdown)"):
    # create a simple markdown report from results
    report = "# Executive Report — Citizen Feedback Analysis\n\n"
    report += f"**Total complaints:** {len(df)}\n\n"
    if sentiment_results and "overall_counts" in sentiment_results:
        o = sentiment_results["overall_counts"]
        report += "## Sentiment\n"
        report += f"- Positive: {o.get('positive',0)}\n- Neutral: {o.get('neutral',0)}\n- Negative: {o.get('negative',0)}\n\n"
    if rootcause_results:
        report += "## Root Causes (Top clusters)\n"
        for k,v in rootcause_results.items():
            report += f"### Cluster: {k}\n"
            for rc in v.get("root_causes", []):
                report += f"- **Cause:** {rc.get('cause')}\n  - Probability: {rc.get('probability')}\n  - Actions:\n"
                for a in rc.get("actions", [])[:3]:
                    report += f"    - {a}\n"
            report += "\n"
    st.download_button("Download Executive Report (.md)", data=report, file_name="executive_report.md", mime="text/markdown")

st.markdown("---")
st.caption("Advanced features powered by an LLM — NER, sentiment & RCA outputs depend on model quality and prompt framing. Use with policy & privacy considerations for real data.")
