import streamlit as st
import pandas as pd
from openai import OpenAI
import pydeck as pdk
import json
import numpy as np

# ----------------------------------------------------------
# 1. BASIC UI
# ----------------------------------------------------------
st.title("🇮🇳 AI-Based Citizen Feedback Analyzer (Advanced Map Version)")
st.caption("Heatmaps • Time Slider • Sentiment Coloring • Choropleth • Filters")


# ----------------------------------------------------------
# 2. OPENAI CLIENT
# ----------------------------------------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ----------------------------------------------------------
# 3. GEO DATA — STATES + DISTRICTS
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

# Basic district-level centroid dictionary
district_coords = {
    "Bengaluru": [12.9716, 77.5946],
    "Mumbai": [19.0760, 72.8777],
    "Jaipur": [26.9124, 75.7873],
    "Chennai": [13.0827, 80.2707],
    "Lucknow": [26.8467, 80.9462],
    "Delhi": [28.7041, 77.1025],
    "Thiruvananthapuram": [8.5241, 76.9366],
    "Patna": [25.0961, 85.3131],
    "Ahmedabad": [23.0225, 72.5714],
    "Kolkata": [22.5726, 88.3639],
}


# ----------------------------------------------------------
# 4. FILE UPLOAD
# ----------------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure date column exists for time slice
    if "date" not in df.columns:
        df["date"] = "2024-01-01"

    df["date"] = pd.to_datetime(df["date"], errors='coerce')

    st.write("### 📄 Preview")
    st.dataframe(df.head())


    # ----------------------------------------------------------
    # 5. SENTIMENT COLORING (auto or provided)
    # ----------------------------------------------------------
    if "sentiment" not in df.columns:
        df["sentiment"] = np.random.choice(["positive", "neutral", "negative"], size=len(df))

    def sentiment_to_color(sent):
        if sent == "negative":
            return [255, 0, 0]   # red
        if sent == "neutral":
            return [255, 165, 0] # orange
        return [0, 200, 0]       # green

    df["color"] = df["sentiment"].apply(sentiment_to_color)


    # ----------------------------------------------------------
    # 6. ADD LAT/LON USING STATE & DISTRICT
    # ----------------------------------------------------------
    def get_latlon(row):
        dist = str(row["district"])
        state = str(row["state"])

        if dist in district_coords:
            return district_coords[dist]
        if state in state_coords:
            return state_coords[state]
        return [None, None]

    df[["lat", "lon"]] = df.apply(
        lambda row: pd.Series(get_latlon(row)),
        axis=1
    )

    map_df = df.dropna(subset=["lat", "lon"])


    # ----------------------------------------------------------
    # 7. FILTERS — Department & Date (Time Slider)
    # ----------------------------------------------------------
    st.write("### 🔍 Filters")

    # DEPARTMENT FILTER
    if "department" not in df.columns:
        df["department"] = np.random.choice(
            ["Water", "Electricity", "Roads", "Municipality", "Revenue"],
            size=len(df)
        )

    dept_filter = st.multiselect(
        "Filter by Department",
        options=df["department"].unique(),
        default=df["department"].unique()
    )

    # DATE SLIDER
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.slider("Time Filter (by Date)",
                           min_value=min_date,
                           max_value=max_date,
                           value=(min_date, max_date))

    filtered = map_df[
        (map_df["department"].isin(dept_filter)) &
        (map_df["date"] >= date_range[0]) &
        (map_df["date"] <= date_range[1])
    ]


    # ----------------------------------------------------------
    # 8. HEATMAP + SCATTER MAP + CLUSTER MAP
    # ----------------------------------------------------------
    st.write("### 🗺️ Maps (Heatmap, Scatter, Clusters)")

    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=filtered,
        get_position='[lon, lat]',
        radiusPixels=60
    )

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=filtered,
        get_position='[lon, lat]',
        get_color='color',
        get_radius=50000,
        pickable=True
    )

    cluster_layer = pdk.Layer(
        "ScreenGridLayer",
        data=filtered,
        get_position='[lon, lat]',
        cellSizePixels=70,
        colorRange=[
            [255, 255, 204],
            [255, 237, 160],
            [254, 217, 118],
            [254, 178, 76],
            [253, 141, 60],
            [240, 59, 32],
            [189, 0, 38],
        ]
    )

    view = pdk.ViewState(latitude=22.9, longitude=78.6, zoom=4)

    st.pydeck_chart(
        pdk.Deck(
            layers=[heat_layer, scatter_layer, cluster_layer],
            initial_view_state=view,
            tooltip={"text": "{state} | {district} | {complaint_text}\nSentiment: {sentiment}"}
        )
    )


    # ----------------------------------------------------------
    # 9. CHOROPLETH MAP (State-level complaint count)
    # ----------------------------------------------------------
    st.write("### 🗺️ Choropleth — Complaints per State")

    # group
    state_counts = df.groupby("state").size().reset_index(name="count")

    # join geo coords
    state_counts["lat"] = state_counts["state"].apply(lambda x: state_coords[x][0] if x in state_coords else None)
    state_counts["lon"] = state_counts["state"].apply(lambda x: state_coords[x][1] if x in state_coords else None)

    state_counts = state_counts.dropna(subset=["lat", "lon"])

    # color by volume
    def volume_to_color(count):
        if count < 50: return [200, 230, 255]
        if count < 150: return [100, 180, 255]
        if count < 300: return [50, 130, 255]
        return [0, 80, 200]

    state_counts["color"] = state_counts["count"].apply(volume_to_color)

    polygon_layer = pdk.Layer(
        "ScatterplotLayer",
        data=state_counts,
        get_position='[lon, lat]',
        get_color='color',
        get_radius=150000,
        pickable=True
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[polygon_layer],
            initial_view_state=view,
            tooltip={"text": "{state} — Complaints: {count}"}
        )
    )


    # ----------------------------------------------------------
    # 10. AI INSIGHTS (same as original)
    # ----------------------------------------------------------
    st.write("### 🤖 AI Analysis")

    sample_rows = df.head(50).to_json(orient="records")

    if st.button("Generate Insights"):
        with st.spinner("Thinking..."):
            prompt = f"""
            Analyze these Indian citizen complaints:
            {sample_rows}

            Give:
            - Key complaint categories
            - Sentiment patterns
            - Department-wise issues
            - Root causes
            - Recommendations
            - 5-point executive summary
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            st.markdown(response.choices[0].message.content)
