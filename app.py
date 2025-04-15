import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("final_model_data.csv")

# Descriptive age labels for display
age_labels_display = {
    "Young (21–35 yrs)": "Young",
    "Mid (36–45 yrs)": "Mid",
    "Older (46–65 yrs)": "Older"
}

# Create label column for visualization
label_map = {v: k for k, v in age_labels_display.items()}
df["age_group_label"] = df["age_group"].map(label_map)

# Prepare input data for training
X = pd.get_dummies(df[["age_group", "educ"]], drop_first=True)
y = df["selfemp_mis8"]

# Balance the data
rus = RandomUnderSampler()
X_res, y_res = rus.fit_resample(X, y)

# Train model
model = LogisticRegression()
model.fit(X_res, y_res)

# Streamlit UI
st.title("📊 Self-Employment Trends in California (2009–2019)")
st.markdown("Explore how self-employment varies by **age group** and **education level**.")

# User Inputs
age_display = st.selectbox("Select Age Group", list(age_labels_display.keys()), key="age_group")
age_input = age_labels_display[age_display]  # used for model input

educ_input = st.selectbox("Select Education Level (coded 1–4)", sorted(df["educ"].unique()), key="education_level")

# Prepare input for prediction
input_df = pd.DataFrame({
    "educ": [educ_input],
    "age_group": [age_input]
})
input_encoded = pd.get_dummies(input_df, drop_first=True)

# Align with model training columns
for col in X.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X.columns]

# Predict and display result
prob = model.predict_proba(input_encoded)[0][1]
st.metric(label="Predicted Probability of Being Self-Employed", value=f"{prob:.2%}")

# Bar Chart of Self-Employment Rate by Age Group
st.subheader("📈 Average Self-Employment Rate by Age Group")
grouped = df.groupby("age_group_label")["selfemp_mis8"].mean().sort_index()
st.bar_chart(grouped)

# Optional: Expand to view raw data
with st.expander("📄 View Raw Data"):
    st.dataframe(df.head(100))

