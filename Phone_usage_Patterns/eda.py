import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def show_eda(df):
    st.subheader("📌 Dataset Overview")
    st.dataframe(df.head())

    st.subheader("🔍 Correlation Heatmap")

    # Only select numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("🎯 Screen Time vs Primary Use")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='primary_use', y='screen_time_(hrs/day)', data=df, ax=ax2)
    st.pyplot(fig2)
