import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data():
    df = pd.read_csv("data/phone_usage_india.csv")
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df

def encode_columns(df):
    label_encoders = {}
    categorical_cols = ['gender', 'location', 'phone_brand', 'os', 'primary_use']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders
