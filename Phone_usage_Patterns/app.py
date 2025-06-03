import streamlit as st
from utils import load_data, encode_columns
from eda import show_eda
from model_train import train_model, predict_primary_use
from clustering import show_clusters
import pandas as pd

st.set_page_config(page_title="Phone Usage Pattern Analysis", layout="wide")

df = load_data()
df, label_encoders = encode_columns(df)

st.sidebar.title("📱 Dashboard")
page = st.sidebar.radio("Go to", ["EDA", "Classification", "Clustering"])

if page == "EDA":
    show_eda(df)

elif page == "Classification":
    st.header("🧠 Predict Primary Use")

    clf, feature_cols, _, _, _, _ = train_model(df, label_encoders)

    st.subheader("🎯 Enter User Info")
    with st.form("user_input_form"):
        age = st.slider("Age", 10, 80, 30)
        gender = st.selectbox("Gender", label_encoders['gender'].classes_)
        os = st.selectbox("OS", label_encoders['os'].classes_)
        screen_time = st.slider("Screen Time", 0.0, 15.0, 5.0)
        data_usage = st.slider("Data Usage", 0.0, 100.0, 10.0)
        call_duration = st.slider("Calls Duration", 0.0, 300.0, 60.0)
        apps = st.slider("Apps Installed", 0, 200, 50)
        social_time = st.slider("Social Media Time", 0.0, 10.0, 2.0)
        ecom = st.slider("E-commerce Spend", 0, 10000, 2000)
        stream_time = st.slider("Streaming Time", 0.0, 10.0, 2.0)
        game_time = st.slider("Gaming Time", 0.0, 10.0, 2.0)
        recharge = st.slider("Monthly Recharge", 0, 5000, 1000)

        submitted = st.form_submit_button("Predict")

    if submitted:
        user_input = pd.DataFrame([[
            age,
            label_encoders['gender'].transform([gender])[0],
            0,  # dummy for location
            0,  # dummy for phone brand
            label_encoders['os'].transform([os])[0],
            screen_time, data_usage, call_duration, apps,
            social_time, ecom, stream_time, game_time, recharge
        ]], columns=feature_cols)

        pred = predict_primary_use(clf, user_input)
        pred_label = label_encoders['primary_use'].inverse_transform([pred])[0]
        st.success(f"📌 Predicted Primary Use: **{pred_label}**")

elif page == "Clustering":
    show_clusters(df)
