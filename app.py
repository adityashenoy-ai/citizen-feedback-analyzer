# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import time
from openai import OpenAI
import pydeck as pdk
from collections import Counter

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(page_title="AI Citizen Feedback Analyzer (Advanced)", layout="wide")
st.title("🇮🇳 AI Citizen Feedback Analyzer — Advanced")
st.caption("NER · Sentiment · Root-Cause Modeling · Dashboards · Maps")

# ---------------------------------------------------
# Init OpenAI Client
# ---------------------------------------------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Add OPENAI_API_KEY to Streamlit Secrets")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------------------------------------------
# State → Geo Coordinates
# ---------------------------------------------------
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

# ---------------------------------------------------
# Utility: Load CSV
# ---------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file):
    df = pd.read_csv(file)

    # normalize columns
    df.columns = [c.lower() for c in df.columns]

    if "complaint_text" not in df.columns:
        for alt in ["complaint", "text", "description"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "complaint_text"})

    if "state" not in df.columns:
        if "region" in df.columns:
            df = df.rename(columns={"region": "state"})
        else:
            df["state"] = ""

    if "district" not in df.columns:
        df["district"] = ""

    if "complaint_id" not in df.columns:
        df.insert(0, "complaint_id", range(1, len(df) + 1))

    return df

# ---------------------------------------------------
# Utility: LLM call with retry
# ---------------------------------------------------
def call_llm(prompt, model="gpt-4o-mini", temperature=0.0):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            content = resp.choices[0].message.content
            return content
        except Exception as e:
            time.sleep(1 + attempt)
            last_error = e
    return None

# ---------------------------------------------------
# Extract JSON robustly
# ---------------------------------------------------
def extract_json_from_text(text):
    if not text:
        raise ValueError("Empty LLM response")

    # 1) Try fenced code block ```json ... ```
    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            pass

    # 2) Try first {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except:
            # try basic fix
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            return json.loads(fixed)

    # 3) Try first [...]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except:
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            return json.loads(fixed)

    raise ValueError("No valid JSON found")

# ---------------------------------------------------
# Build Prompts
# ---------------------------------------------------
def build_ner_prompt(sample_json):
    return f"""
You are an expert NER system. Extract structured entities from the following complaints:

{sample_json}

Return JSON ONLY (wrapped in ```json):

```json
{{
  "entities": [
    {{
      "id": 1,
      "issues": ["..."],
      "organizations": ["..."],
      "locations": ["..."],
      "dates": ["..."],
      "persons": ["..."],
      "severity": "low"
    }}
  ]
}}
```json

**Rules**
- No explanation, ONLY JSON
- Limit arrays to 5 items
- severity must be: low / medium / high
"""

def build_sentiment_prompt(sample_json):
    return f"""
You are a sentiment analyzer. Analyze the following complaint texts:

{sample_json}

Return JSON ONLY:

```json
{{
  "sentiments": [
    {{"id": 1, "sentiment": "negative", "score": 0.88}}
  ],
  "overall_counts": {{
    "positive": 0,
    "neutral": 0,
    "negative": 0
  }}
}}
```json
"""

def build_rootcause_prompt(cluster_json, name):
    return f"""
You are a policy expert. Analyze cluster '{name}'.

Examples:
{cluster_json}

Return JSON ONLY:

```json
{{
  "cluster": "{name}",
  "root_causes": [
    {{
      "cause": "...",
      "probability": "high",
      "impact": "medium",
      "actions": ["...", "...", "..."],
      "kpis": ["...","..."]
    }}
  ]
}}
```json
"""

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
with st.sidebar:
    st.header("Upload / Settings")
    file = st.file_uploader("Upload CSV", type=["csv"])
    model = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o"])
    max_sample = st.slider("Sample size to LLM", 50, 400, 150, step=50)
    run_ner = st.checkbox("Run NER", True)
    run_sent = st.checkbox("Run Sentiment", True)
    run_rca = st.checkbox("Run Root Cause Analysis", True)

if not file:
    st.info("Upload a CSV to begin")
    st.stop()

# ---------------------------------------------------
# Load file
# ---------------------------------------------------
df = load_csv(file)
df["state"] = df["state"].astype(str)
df["district"] = df["district"].astype(str)
df["lat"] = df["state"].apply(lambda x: STATE_COORDS[x][0] if x in STATE_COORDS else None)
df["lon"] = df["state"].apply(lambda x: STATE_COORDS[x][1] if x in STATE_COORDS else None)

# ---------------------------------------------------
# Dashboard
# ---------------------------------------------------
st.success(f"Loaded {len(df)} complaints")

col1, col2, col3 = st.columns(3)
col1.metric("Complaints", len(df))
col2.metric("States", df["state"].nunique())
col3.metric("Sample to LLM", max_sample)

# Complaints per state
st.subheader("Complaints per State")
count_df = df["state"].value_counts().reset_index()
st.bar_chart(count_df.set_index("index"))

# Top words
st.subheader("Top Keywords")
all_words = " ".join(df["complaint_text"].astype(str).str.lower()).split()
all_words = [w for w in all_words if len(w) > 4]
top = Counter(all_words).most_common(20)
st.table(pd.DataFrame(top, columns=["word", "count"]))

# Map
st.subheader("Complaint Density Map")
map_df = df.dropna(subset=["lat","lon"])

if len(map_df):
    heat = pdk.Layer(
        "HeatmapLayer",
        data=map_df,
        get_position='[lon, lat]',
        radiusPixels=60,
    )
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=map_df.head(1000),
        get_position='[lon, lat]',
        get_radius=50000,
        get_color=[255,0,0],
        pickable=True
    )
    view = pdk.ViewState(latitude=22.9, longitude=78.6, zoom=4)
    deck = pdk.Deck(layers=[heat, scatter], initial_view_state=view,
        tooltip={"text":"{state}\n{district}\n{complaint_text}"})
    st.pydeck_chart(deck)

# ---------------------------------------------------
# Prepare sample for LLM
# ---------------------------------------------------
sample = df.head(max_sample)[["complaint_id","complaint_text"]]
sample_json = json.dumps(
    [{"id": int(r["complaint_id"]), "text": r["complaint_text"]} for _, r in sample.iterrows()],
    ensure_ascii=False
)

# ---------------------------------------------------
# NER
# ---------------------------------------------------
if run_ner:
    st.subheader("Named Entity Recognition (NER)")
    prompt = build_ner_prompt(sample_json)
    raw = call_llm(prompt, model=model, temperature=0.0)

    try:
        parsed = extract_json_from_text(raw)
        ner_entities = parsed.get("entities", parsed)
        st.success("NER Parsed Successfully")
        st.json(ner_entities)
    except:
        st.error("NER JSON Parsing Failed")
        st.code(raw)

# ---------------------------------------------------
# Sentiment
# ---------------------------------------------------
if run_sent:
    st.subheader("Sentiment Analysis")
    prompt = build_sentiment_prompt(sample_json)
    raw = call_llm(prompt, model=model, temperature=0.0)

    try:
        parsed = extract_json_from_text(raw)
        st.success("Sentiment Parsed Successfully")
        st.json(parsed)

        if "overall_counts" in parsed:
            dist = parsed["overall_counts"]
            dist_df = pd.DataFrame.from_dict(dist, orient="index", columns=["count"])
            st.bar_chart(dist_df)
    except:
        st.error("Sentiment JSON parse failed")
        st.code(raw)

# ---------------------------------------------------
# Root Cause Analysis
# ---------------------------------------------------
if run_rca:
    st.subheader("Root Cause Modeling")

    # cluster using top keywords
    keywords = [w for w,c in top[:5]]
    clusters = {}
    for kw in keywords:
        rows = df[df["complaint_text"].str.contains(kw, case=False, na=False)].head(20)
        if len(rows):
            clusters[kw] = rows["complaint_text"].tolist()

    if not clusters:
        st.warning("No clusters formed")
    else:
        for kw, examples in clusters.items():
            st.write(f"### Cluster: {kw}")
            cl_json = json.dumps([{"text": t} for t in examples], ensure_ascii=False)

            prompt = build_rootcause_prompt(cl_json, kw)
            raw = call_llm(prompt, model=model)

            try:
                parsed = extract_json_from_text(raw)
                st.json(parsed)
            except:
                st.error(f"Failed to parse RCA JSON for cluster {kw}")
                st.code(raw)

# ---------------------------------------------------
# End
# ---------------------------------------------------
st.markdown("---")
st.caption("Built by Streamlit + OpenAI — fully LLM-powered analytics.")
