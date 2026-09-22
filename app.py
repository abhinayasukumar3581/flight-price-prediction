import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import date, time

MODEL_PATH = "flight_price_best_model.joblib"


# ------------------------------------------------------------
# Feature engineering — kept consistent with the training notebook
# ------------------------------------------------------------
def create_features(data):
    data = data.copy()

    journey_date = pd.to_datetime(
        data["Date_of_Journey"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    data["Journey_Day"] = journey_date.dt.day
    data["Journey_Month"] = journey_date.dt.month
    data["Journey_Weekday"] = journey_date.dt.dayofweek

    for col, prefix in [("Dep_Time", "Dep"), ("Arrival_Time", "Arrival")]:
        time_parts = data[col].astype(str).str.extract(r"(\d{1,2}):(\d{2})")

        data[f"{prefix}_Hour"] = pd.to_numeric(
            time_parts[0], errors="coerce"
        )
        data[f"{prefix}_Minute"] = pd.to_numeric(
            time_parts[1], errors="coerce"
        )

    duration_text = data["Duration"].astype(str)

    hours = pd.to_numeric(
        duration_text.str.extract(r"(\d+)\s*h")[0],
        errors="coerce"
    ).fillna(0)

    minutes = pd.to_numeric(
        duration_text.str.extract(r"(\d+)\s*m")[0],
        errors="coerce"
    ).fillna(0)

    data["Duration_Minutes"] = hours * 60 + minutes

    stops_text = data["Total_Stops"].fillna("1 stop").astype(str)

    data["Stops_Num"] = pd.to_numeric(
        stops_text.str.extract(r"(\d+)")[0],
        errors="coerce"
    )

    data.loc[
        stops_text.str.lower().eq("non-stop"),
        "Stops_Num"
    ] = 0

    route_parts = data["Route"].fillna("None").astype(str).str.split("→")

    for i in range(5):
        data[f"Route_{i+1}"] = (
           route_parts.str[i].fillna("None").astype(str).str.strip()
        )

    drop_cols = [
        "Date_of_Journey",
        "Route",
        "Dep_Time",
        "Arrival_Time",
        "Duration",
        "Total_Stops"
    ]

    return data.drop(columns=drop_cols, errors="ignore")


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Flight Price Prediction")
st.write(
    "Enter the flight details below to estimate the ticket price "
    "using the trained machine-learning model."
)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model_bundle = load_model()
    model = model_bundle["model"]
    model_name = model_bundle.get("model_name", "Trained ML Model")
except Exception as e:
    st.error("The trained model could not be loaded.")
    st.code(str(e))
    st.info(
        "Make sure flight_price_best_model.joblib is in the same folder "
        "as app.py and that the requirements.txt versions are installed."
    )
    st.stop()


st.caption(f"Model loaded: {model_name}")

# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
st.subheader("Enter Flight Details")

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox(
        "Airline",
        [
            "IndiGo",
            "Air India",
            "Jet Airways",
            "SpiceJet",
            "Multiple carriers",
            "GoAir",
            "Vistara",
            "Air Asia",
            "Vistara Premium economy",
            "Jet Airways Business",
            "Multiple carriers Premium economy",
            "Trujet"
        ]
    )

    journey_date = st.date_input(
        "Date of Journey",
        value=date(2019, 3, 15)
    )

    source = st.selectbox(
        "Source",
        [
            "Banglore",
            "Kolkata",
            "Delhi",
            "Chennai",
            "Mumbai"
        ]
    )

    destination = st.selectbox(
        "Destination",
        [
            "New Delhi",
            "Banglore",
            "Cochin",
            "Kolkata",
            "Delhi",
            "Hyderabad"
        ]
    )

    stops = st.selectbox(
        "Total Stops",
        [
            "non-stop",
            "1 stop",
            "2 stops",
            "3 stops",
            "4 stops"
        ]
    )

with col2:
    route = st.text_input(
        "Route",
        value=f"{source} → {destination}",
        help="Use airport/city names separated by →, for example: "
             "Banglore → Delhi"
    )

    departure_time = st.time_input(
        "Departure Time",
        value=time(10, 0)
    )

    arrival_time = st.time_input(
        "Arrival Time",
        value=time(13, 0)
    )

    duration_hours = st.number_input(
        "Duration — Hours",
        min_value=0,
        max_value=48,
        value=2,
        step=1
    )

    duration_minutes = st.number_input(
        "Duration — Additional Minutes",
        min_value=0,
        max_value=59,
        value=50,
        step=1
    )

    additional_info = st.text_input(
        "Additional Information",
        value="No info"
    )

duration_string = f"{int(duration_hours)}h {int(duration_minutes)}m"

if st.button("🔮 Predict Flight Price", type="primary", use_container_width=True):

    if not route.strip():
        st.warning("Please enter a route.")
        st.stop()

    raw_input = pd.DataFrame([{
        "Airline": airline,
        "Date_of_Journey": journey_date.strftime("%d/%m/%Y"),
        "Source": source,
        "Destination": destination,
        "Route": route,
        "Dep_Time": departure_time.strftime("%H:%M"),
        "Arrival_Time": arrival_time.strftime("%H:%M"),
        "Duration": duration_string,
        "Total_Stops": stops,
        "Additional_Info": additional_info if additional_info.strip() else "No info"
    }])

    try:
        processed_input = create_features(raw_input)

        # Keep exactly the feature columns used by the training notebook.
        expected_columns = model_bundle["feature_columns"]

        for column in expected_columns:
            if column not in processed_input.columns:
                processed_input[column] = np.nan

        processed_input = processed_input[expected_columns]

        prediction = float(model.predict(processed_input)[0])

        st.success("Prediction completed!")

        st.metric(
            label="Estimated Flight Price",
            value=f"₹ {prediction:,.0f}"
        )

        st.info(
            "This is a machine-learning estimate. Actual fares can vary "
            "based on availability, booking time, demand and airline pricing."
        )

        with st.expander("View processed input"):
            st.dataframe(processed_input)

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)


st.markdown("---")
st.caption("Flight Price Prediction • Machine Learning Deployment")
