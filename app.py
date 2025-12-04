import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import pydeck as pdk
import json

# ----------------------------------------------------------
# PAGE SETUP
# ----------------------------------------------------------
st.set_page_config(page_title="AI Citizen Feedback Analyzer", layout="wide")
st.title("🇮🇳 AI-Based Citizen Feedback Analyzer (Advanced Maps)")
st.caption("Heatmap • Time Slider • Filters • Sentiment Colors • District Geo")


# ----------------------------------------------------------
# OPENAI CLIENT
# ----------------------------------------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ----------------------------------------------------------
# STATE + DISTRICT GEO DATA
# ----------------------------------------------------------
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

district_coords = {
    "Bengaluru": [12.9716, 77.5946],
    "Mumbai": [19.0760, 72.8777],
    "Jaipur": [26.9124, 75.7873],
    "Lucknow": [26.8467, 80.9462],
    "Chennai": [13.0827, 80.2707],
    "Delhi": [28.7041, 77.1025],
    "Thiruvananthapuram": [8.5241, 76.9366],
    "Patna": [25.0961, 85.3131],
    "Ahmedabad": [23.0225, 72.5714],
    "Kolkata": [22.5726, 88.3639],
}


# ----------------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # If missing important columns, fill defaults
    if "department" not in df.columns:
        df["department"] = np.random.choice(
            ["Water", "Electricity", "Roads", "Municipality", "Revenue"],
            size=len(df)
        )

    if "sentiment" not in df.columns:
        df["sentiment"] = np.random.choice(
            ["positive", "neutral", "negative"],
            size=len(df)
        )

    # Ensure date exists
    if "date" not in df.columns:
        df["date"] = "2024-01-01"

    # ----------------------------------------------------------
    # SAFE DATE HANDLING (Fixes your error!)
    # ----------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    valid_dates = df.dropna(subset=["date"])

    if valid_dates.empty:
        st.warning("⚠ No valid date values found. Time slider disabled.")
        df["date"] = pd.Timestamp("2024-01-01")
        min_date = max_date = pd.Timestamp("2024-01-01")
    else:
        df = valid_dates
        min_date = df["date"].min()
        max_date = df["date"].max()

    min_date = pd.to_datetime(min_date)
    max_date = pd.to_datetime(max_date)

    st.write("### 📄 Preview of Data")
    st.dataframe(df.head())

    # ----------------------------------------------------------
    # GEOCODING (State + District)
    # ----------------------------------------------------------
    def get_latlon(row):
        if row["district"] in district_coords:
            return district_coords[row["district"]]
        if row["state"] in state_coords:
            return state_coords[row["state"]]
        return [None, None]

    df[["lat", "lon"]] = df.apply(lambda r: pd.Series(get_latlon(r)), axis=1)
    map_df = df.dropna(subset=["lat", "lon"])



    # ----------------------------------------------------------
    # FILTERS
    # ----------------------------------------------------------
    st.write("### 🔍 Filters")

    dept_filter = st.multiselect(
        "Filter by Department",
        options=df["department"].unique(),
        default=df["department"].unique(),
    )

    date_range = st.slider(
        "Filter by Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    filtered = map_df[
        (map_df["department"].isin(dept_filter)) &
        (map_df["date"] >= date_range[0]) &
        (map_df["date"] <= date_range[1])
    ]


    # ----------------------------------------------------------
    # SENTIMENT → COLOR
    # ----------------------------------------------------------
    def sentiment_color(s):
        if s == "negative": return [255, 0, 0]
        if s == "neutral": return [255, 165, 0]
        return [0, 200, 0]

    filtered["color"] = filtered["sentiment"].apply(sentiment_color)


    # ----------------------------------------------------------
    # MAPS — Heatmap + Scatter + Cluster
    # ----------------------------------------------------------
    st.write("### 🗺️ Interactive Maps")

    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=filtered,
        get_position='[lon, lat]',
        radiusPixels=60,
    )

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=filtered,
        get_position='[lon, lat]',
        get_radius=50000,
        get_color='color',
        pickable=True
    )

    cluster_layer = pdk.Layer(
        "ScreenGridLayer",
        data=filtered,
        get_position='[lon, lat]',
        cellSizePixels=60
    )

    view_state = pdk.ViewState(latitude=22.97, longitude=78.65, zoom=4)

    st.pydeck_chart(
        pdk.Deck(
            layers=[heat_layer, scatter_layer, cluster_layer],
            initial_view_state=view_state,
            tooltip={"text": "{state} | {district} | {complaint_text}"}
        )
    )


    # ----------------------------------------------------------
    # STATE-LEVEL CHOROPLETH (bubble)
    # ----------------------------------------------------------
    st.write("### 📍 Choropleth — Complaints per State")

    state_counts = df.groupby("state").size().reset_index(name="count")
    state_counts["lat"] = state_counts["state"].apply(lambda x: state_coords.get(x, [None, None])[0])
    state_counts["lon"] = state_counts["state"].apply(lambda x: state_coords.get(x, [None, None])[1])
    state_counts = state_counts.dropna(subset=["lat", "lon"])

    def volume_color(c):
        if c < 50: return [180, 220, 255]
        if c < 150: return [80, 160, 255]
        return [0, 70, 200]

    state_counts["color"] = state_counts["count"].apply(volume_color)

    bubble_layer = pdk.Layer(
        "ScatterplotLayer",
        data=state_counts,
        get_position='[lon, lat]',
        get_color='color',
        get_radius=150000,
        pickable=True
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[bubble_layer],
            initial_view_state=view_state,
            tooltip={"text": "{state} — {count} complaints"}
        )
    )


    # ----------------------------------------------------------
    # AI INSIGHTS
    # ----------------------------------------------------------
    st.write("### 🤖 AI Insights")

    sample_rows = df.head(50).to_json(orient="records")

    if st.button("Generate AI Insights"):
        with st.spinner("Analyzing complaints…"):
            prompt = f"""
            Analyze these Indian citizen complaints:
            {sample_rows}

            Provide:
            - Key complaint categories
            - Sentiment patterns
            - Root causes
            - Department-wise issues
            - Recommendations
            - Executive summary
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            st.markdown(response.choices[0].message.content)
