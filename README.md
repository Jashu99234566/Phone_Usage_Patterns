# 📱 Phone Usage Pattern Analysis App

## 🧠 Overview

This Streamlit-based web application analyzes mobile phone usage behavior among Indian users. It combines **Exploratory Data Analysis (EDA)**, **Machine Learning classification**, and **KMeans clustering** to extract actionable insights from user-level phone activity data.

---

## 📎 Features

### 1. EDA Module (`eda.py`)

* Visualizes dataset structure
* Displays correlation heatmap of numeric features
* Compares screen time across different primary usage types (e.g., gaming, streaming)

### 2. Classification Module (`model_train.py`)

* Uses a **Random Forest Classifier** to predict a user's primary use (e.g., gaming, calling, social media) based on:

  * Age, gender, OS, screen time, data usage, number of apps, recharge value, etc.
* Includes an interactive form to input custom user data and get predictions

### 3. Clustering Module (`clustering.py`)

* Uses **KMeans Clustering** to group users into behavioral clusters based on scaled numeric features
* Shows cluster distributions and example user assignments

---

## 🔧 Tech Stack

* **Frontend**: Streamlit
* **ML Models**: Scikit-learn (RandomForest, KMeans)
* **Data Handling**: Pandas, LabelEncoder
* **Visualization**: Seaborn, Matplotlib

---

## 📂 File Structure

| File             | Description                                                                    |
| ---------------- | ------------------------------------------------------------------------------ |
| `app.py`         | Main Streamlit app with navigation between EDA, Classification, and Clustering |
| `utils.py`       | Data loading and label encoding utilities                                      |
| `eda.py`         | EDA visualizations                                                             |
| `model_train.py` | ML model training and prediction logic                                         |
| `clustering.py`  | KMeans-based user clustering                                                   |

---

## 📊 Dataset

* `data/phone_usage_india.csv` (**not included**)
* Contains anonymized user-level mobile usage data with columns like `gender`, `os`, `screen_time`, `apps_installed`, etc.

---

## 🎯 Use Cases

* Identify key phone usage segments
* Predict how a new user is likely to use their phone
* Cluster users for marketing or UX personalization

---

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place `phone_usage_india.csv` inside a `data/` folder
4. Run the app: `streamlit run app.py`

---

## 🙏 Acknowledgments

Made with ❤️ using Python and Streamlit for behavioral data analysis.
