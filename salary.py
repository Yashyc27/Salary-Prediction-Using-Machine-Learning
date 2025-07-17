import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Salary Prediction App")

@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv('Salary_Data.csv')
    except FileNotFoundError:
        st.error("Error: 'Salary_Data.csv' not found. Please make sure the file is in the same directory as the Streamlit app.")
        st.stop()
    
    df.dropna(inplace=True)

    job_title_count = df['Job Title'].value_counts()
    job_title_edited = job_title_count[job_title_count <= 25]
    df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in job_title_edited.index else x)

    job_category_map = {
        'Software Engineer': 1, 'Data Analyst': 1, 'Software Developer': 1,
        'Web Developer': 1, 'Junior Software Developer': 1, 'Junior Web Developer': 1,
        'Junior Software Engineer': 1, 'Full Stack Engineer': 1, 'Front end Developer': 1,
        'Front End Developer': 1, 'Back end Developer': 1, 'Data Scientist': 1,
        'Senior Data Scientist': 1, 'Research Scientist': 1, 'Senior Research Scientist': 1,
        'Content Marketing Manager': 1, 'Software Engineer Manager': 1,
        'Senior Software Engineer': 1, 'Product Designer': 1,
        'Sales Associate': 2, 'Marketing Analyst': 2, 'Marketing Coordinator': 2,
        'Marketing Manager': 2, 'Sales Executive': 2, 'Sales Representative': 2,
        'Junior Sales Representative': 2, 'Junior Sales Associate': 2,
        'Junior Marketing Manager': 2, 'Receptionist': 2, 'Junior HR Generalist': 2,
        'Junior HR Coordinator': 2, 'Senior HR Generalist': 2,
        'Human Resources Coordinator': 2, 'Human Resources Manager': 2,
        'Financial Analyst': 2, 'Operations Manager': 2, 'Financial Manager': 2,
        'Research Director': 2, 'Director of HR': 2,
        'Product Manager': 3, 'Sales Manager': 3, 'Sales Director': 3,
        'Marketing Director': 3, 'Director of Marketing': 3,
        'Senior Human Resources Manager': 3, 'Senior Product Marketing Manager': 3,
        'Director of Data Science': 3,
    }
    df['Job Category Code'] = df['Job Title'].map(job_category_map).fillna(0).astype(int)

    education_map = {
        "High School": 0,
        "Bachelor's Degree": 1, "Bachelor's": 1,
        "Master's Degree": 2, "Master's": 2,
        "PhD": 3, "phD": 3
    }
    df['Education Level'] = df['Education Level'].map(education_map).fillna(0).astype(int)

    gender_encoder = LabelEncoder()
    df['Gender_Encoded'] = gender_encoder.fit_transform(df['Gender'])

    return df, gender_encoder, list(df['Job Title'].unique()), list(education_map.keys())

df_processed, gender_encoder, unique_job_titles, unique_education_levels_input = load_and_preprocess_data()

X = df_processed[['Age', 'Gender_Encoded', 'Education Level', 'Job Category Code', 'Years of Experience']]
y = df_processed['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

st.title("💰 Salary Prediction App")
st.markdown("Predict your potential salary using our Random Forest model!")

st.sidebar.header("About the Model")
st.sidebar.markdown("""
This application uses a Random Forest Regression model to predict salaries based on various factors.
The model was trained on a dataset containing Age, Gender, Education Level, Job Title, Years of Experience, and Salary.
""")

st.sidebar.header("Explore Data Insights")

# Graphs
st.sidebar.subheader("Salary Distribution")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.histplot(df_processed['Salary'], kde=True, ax=ax1, color='#6a0572')
ax1.set_title('Distribution of Salaries')
ax1.set_xlabel('Salary')
ax1.set_ylabel('Count')
st.sidebar.pyplot(fig1)

st.sidebar.subheader("Salary vs. Years of Experience")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.scatterplot(x='Years of Experience', y='Salary', data=df_processed, ax=ax2, alpha=0.6, color='#2a9d8f')
ax2.set_title('Salary vs. Years of Experience')
ax2.set_xlabel('Years of Experience')
ax2.set_ylabel('Salary')
st.sidebar.pyplot(fig2)

st.sidebar.subheader("Salary by Education Level")
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.boxplot(x='Education Level', y='Salary', data=df_processed, ax=ax3, palette='viridis')
ax3.set_title('Salary by Education Level')
ax3.set_xlabel('Education Level')
ax3.set_ylabel('Salary')
st.sidebar.pyplot(fig3)

st.header("Input Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=20, max_value=65, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

with col2:
    education_level = st.selectbox("Education Level", unique_education_levels_input)
    years_of_experience = st.slider("Years of Experience", min_value=0, max_value=40, value=5)

job_title = st.selectbox("Job Title", unique_job_titles)

if st.button("Predict Salary", help="Click to get your estimated salary!"):
    gender_encoded = gender_encoder.transform([gender])[0] if gender in gender_encoder.classes_ else gender_encoder.transform(['Other'])[0]

    education_map_input = {
        "High School": 0,
        "Bachelor's Degree": 1, "Bachelor's": 1,
        "Master's Degree": 2, "Master's": 2,
        "PhD": 3, "phD": 3
    }
    education_level_mapped = education_map_input.get(education_level, 0)

    job_category_map_input = {
        'Software Engineer': 1, 'Data Analyst': 1, 'Software Developer': 1,
        'Web Developer': 1, 'Junior Software Developer': 1, 'Junior Web Developer': 1,
        'Junior Software Engineer': 1, 'Full Stack Engineer': 1, 'Front end Developer': 1,
        'Front End Developer': 1, 'Back end Developer': 1, 'Data Scientist': 1,
        'Senior Data Scientist': 1, 'Research Scientist': 1, 'Senior Research Scientist': 1,
        'Content Marketing Manager': 1, 'Software Engineer Manager': 1,
        'Senior Software Engineer': 1, 'Product Designer': 1,
        'Sales Associate': 2, 'Marketing Analyst': 2, 'Marketing Coordinator': 2,
        'Marketing Manager': 2, 'Sales Executive': 2, 'Sales Representative': 2,
        'Junior Sales Representative': 2, 'Junior Sales Associate': 2,
        'Junior Marketing Manager': 2, 'Receptionist': 2, 'Junior HR Generalist': 2,
        'Junior HR Coordinator': 2, 'Senior HR Generalist': 2,
        'Human Resources Coordinator': 2, 'Human Resources Manager': 2,
        'Financial Analyst': 2, 'Operations Manager': 2, 'Financial Manager': 2,
        'Research Director': 2, 'Director of HR': 2,
        'Product Manager': 3, 'Sales Manager': 3, 'Sales Director': 3,
        'Marketing Director': 3, 'Director of Marketing': 3,
        'Senior Human Resources Manager': 3, 'Senior Product Marketing Manager': 3,
        'Director of Data Science': 3,
    }

    job_title_count_inference = df_processed['Job Title'].value_counts()
    job_title_edited_inference = job_title_count_inference[job_title_count_inference <= 25].index
    if job_title in job_title_edited_inference:
        job_title_for_prediction = 'Others'
    else:
        job_title_for_prediction = job_title

    job_category_code = job_category_map_input.get(job_title_for_prediction, 0)

    input_data = pd.DataFrame([[age, gender_encoded, education_level_mapped, job_category_code, years_of_experience]],
                              columns=['Age', 'Gender_Encoded', 'Education Level', 'Job Category Code', 'Years of Experience'])

    predicted_salary = rf_model.predict(input_data)[0]
    st.success(f"Predicted Salary: ${predicted_salary:,.2f}")


