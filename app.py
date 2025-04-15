
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("final_model_data.csv")

# Encode age group and education
X = pd.get_dummies(df[['age_group', 'educ']], drop_first=True)
y = df['selfemp_mis8']

# Balance the data
rus = RandomUnderSampler()
X_res, y_res = rus.fit_resample(X, y)

# Train the model
model = LogisticRegression()
model.fit(X_res, y_res)

# Streamlit UI
st.title("Self-Employment Trends in California (2009–2019)")
st.markdown("Explore trends by age group and education level.")

# User inputs
age_input = st.selectbox("Select Age Group", ['Young', 'Mid', 'Older'])
educ_input = st.selectbox("Select Education Level", sorted(df['educ'].unique()))

# Prepare input for prediction
input_df = pd.DataFrame({
    'educ': [educ_input],
    'age_group': [age_input]
})
input_encoded = pd.get_dummies(input_df, drop_first=True)
# Align with training columns
for col in X.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X.columns]  # reorder

# Predict
prob = model.predict_proba(input_encoded)[0][1]
st.metric(label="Predicted Probability of Being Self-Employed", value=f"{prob:.2%}")

# Show trends
st.subheader("Self-Employment Rate by Age Group")
grouped = df.groupby("age_group")["selfemp_mis8"].mean()
st.bar_chart(grouped)
