import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Video Game Sales Prediction",
    page_icon="🎮",
    layout="wide"
)

# Load dataset
df = pd.read_csv("vgsales.csv")

# Load model
model = joblib.load("models/linear_regression_model.pkl")

st.title("🎮 Video Game Sales Prediction Dashboard")

st.markdown(
    "Predict Global Video Game Sales using a Linear Regression model."
)

# Sidebar
st.sidebar.header("Filters")

genre = st.sidebar.selectbox(
    "Genre",
    sorted(df["Genre"].dropna().unique())
)

publisher = st.sidebar.selectbox(
    "Publisher",
    sorted(df["Publisher"].dropna().unique())
)

filtered = df[
    (df["Genre"] == genre) &
    (df["Publisher"] == publisher)
]

st.subheader("Filtered Dataset")

st.dataframe(filtered)

st.subheader("Sales Distribution")

fig = px.bar(
    filtered,
    x="Name",
    y="Global_Sales",
    color="Platform"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Predict Global Sales")

year = st.number_input(
    "Release Year",
    1980,
    2035,
    2015
)

na = st.number_input(
    "North America Sales",
    0.0,
    50.0,
    1.0
)

eu = st.number_input(
    "Europe Sales",
    0.0,
    50.0,
    1.0
)

jp = st.number_input(
    "Japan Sales",
    0.0,
    50.0,
    1.0
)

if st.button("Predict"):

    sample = pd.DataFrame({
        "Year": [year],
        "NA_Sales": [na],
        "EU_Sales": [eu],
        "JP_Sales": [jp]
    })

    prediction = model.predict(sample)[0]

    st.success(
        f"Predicted Global Sales: {prediction:.2f} million units"
    )

st.markdown("---")
st.caption("Developed using Python, Streamlit, Plotly and Scikit-learn.")
