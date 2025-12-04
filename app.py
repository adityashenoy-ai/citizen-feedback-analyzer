import streamlit as st
import pandas as pd
import openai
from openai import OpenAI

# Streamlit UI
st.title("🇮🇳 AI-Based Citizen Feedback Analyzer for Indian GovTech Portals")
st.subheader("Upload citizen complaint/feedback data → Get insights, clusters, summaries & action plans")

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# File upload
uploaded_file = st.file_uploader("Upload CSV file containing citizen complaints", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    sample_rows = df.head(50).to_json(orient="records")

    st.write("### 🔍 AI Analysis")
    if st.button("Generate Insights"):
        with st.spinner("Analyzing citizen feedback..."):
            prompt = f"""
            You are analyzing citizen complaints from Indian govt portals (CPGRAMS, state grievance systems,
            RTI responses, municipal feedback).

            Sample dataset:
            {sample_rows}

            Perform the following:
            1. Identify major complaint categories
            2. Detect sentiment distribution
            3. Extract common pain points
            4. Map complaints to relevant govt departments
            5. Identify recurring root causes
            6. Suggest corrective actions
            7. Provide a 5-point executive summary

            Use clean markdown formatting.
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            st.markdown(response.choices[0].message.content)
