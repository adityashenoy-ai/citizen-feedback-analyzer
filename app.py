import streamlit as st
import pandas as pd
import numpy as np
import openai
import os

# Title
st.title("🇮🇳 AI-Based Citizen Feedback Analyzer for Indian GovTech Portals")
st.subheader("Upload citizen complaint/feedback data → Get insights, clusters, summaries & action plans")

# OpenAI API Key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# File Upload
uploaded_file = st.file_uploader("Upload CSV file containing citizen complaints", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # Convert data into a string chunk for LLM
    sample_rows = df.head(50).to_json(orient="records")

    st.write("### 🔍 AI Analysis")
    if st.button("Generate Insights"):
        with st.spinner("Analyzing citizen feedback..."):
            prompt = f"""
            You are an AI system analyzing citizen complaints from Indian Govt portals 
            such as CPGRAMS, state grievance systems, RTI responses, and municipal feedback systems.

            Here is a sample dataset (50 rows) of complaints:
            {sample_rows}

            Perform the following analysis:

            1. Identify major complaint categories (Transport, Electricity, Health, Education, Police, Pension, Water, Land, Municipal Works, etc.)
            2. Detect sentiment distribution (positive, neutral, negative)
            3. Extract the most common pain points
            4. Map complaints to relevant Govt departments/ministries
            5. Identify recurring root causes
            6. Suggest corrective actions for Govt
            7. Generate a short 5-point executive summary for a Govt Secretary

            Provide your output in clean markdown format.
            """

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            st.markdown(response["choices"][0]["message"]["content"])


