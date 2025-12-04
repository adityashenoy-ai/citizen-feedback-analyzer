import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import pydeck as pdk

# Streamlit UI
st.title("🇮🇳 AI-Based Citizen Feedback Analyzer for Indian GovTech Portals")
st.subheader("Upload citizen complaint/feedback data → Get insights, clusters, summaries, visualizations & action plans")

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ----------------------------
#  GEO DATA FOR INDIAN STATES
# ----------------------------
state_coords = {
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

# File upload
uploaded_file = st.file_uploader("Upload CSV file containing citizen complaints", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    # -------------------------------------------------------
    # ADD LAT/LON FOR MAP VISUALIZATION
    # -------------------------------------------------------
    df["lat"] = df["state"].apply(lambda x: state_coords[x][0] if x in state_coords else None)
    df["lon"] = df["state"].apply(lambda x: state_coords[x][1] if x in state_coords else None)

    st.write("### 🗺️ Interactive Map of Complaints")

    # REMOVE rows with missing coordinates
    map_df = df.dropna(subset=["lat", "lon"])

    # Pydeck layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_radius=60000,
        get_color=[255, 0, 0, 160],
        pickable=True
    )

    # Map view
    view_state = pdk.ViewState(
        latitude=22.9734,
        longitude=78.6569,
        zoom=4,
        height=600
    )

    # Render map
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "State: {state}\nDistrict: {district}\nComplaint: {complaint_text}"}
    )

    st.pydeck_chart(r)

    # -------------------------------------------------------
    # AI ANALYSIS
    # -------------------------------------------------------
    st.write("### 🔍 AI Analysis")
    sample_rows = df.head(50).to_json(orient="records")

    if st.button("Generate Insights"):
        with st.spinner("Analyzing citizen feedback with AI..."):
            prompt = f"""
            You are analyzing citizen complaints from Indian government portals
            (CPGRAMS, state grievance systems, RTI responses, municipal portals).

            Sample dataset (50 rows):
            {sample_rows}

            Perform:
            1. Identify key complaint categories.
            2. Detect sentiment distribution.
            3. Extract most common pain points.
            4. Map complaints to govt departments.
            5. Identify recurring root causes.
            6. Suggest corrective actions.
            7. Provide a 5-point executive summary.
            8. Offer data-driven recommendations for policymakers.

            Respond in clean, formatted markdown.
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            st.markdown(response.choices[0].message.content)
