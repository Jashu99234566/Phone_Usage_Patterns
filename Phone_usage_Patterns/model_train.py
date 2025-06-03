import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(df, label_encoders):
    features = df.drop(columns=['user_id', 'primary_use'])
    target = df['primary_use']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    return clf, features.columns, X_train, y_train, X_test, y_test

def predict_primary_use(clf, input_data):
    return clf.predict(input_data)[0]
