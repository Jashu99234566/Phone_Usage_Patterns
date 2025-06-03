import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def show_clusters(df):
    st.write("🔍 Using KMeans to group users based on their usage patterns")

    X = df.drop(columns=['user_id', 'primary_use'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    st.subheader("📌 Cluster Distribution")
    st.bar_chart(df['cluster'].value_counts())

    st.subheader("📍 Sample Clustered Data")
    st.dataframe(df[['user_id', 'cluster']].head())
