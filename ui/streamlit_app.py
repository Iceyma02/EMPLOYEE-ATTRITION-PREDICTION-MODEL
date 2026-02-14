# ui/streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# Page config
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

# Get API URL from environment (for deployment)
API_URL = (https://employee-attrition-prediction-model.onrender.com)
# Title
st.title("📊 Employee Attrition Risk Intelligence System")
st.markdown("---")

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/group.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predictor", "Dashboard", "About"])

# Check API health
@st.cache_data(ttl=60)
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False

api_healthy = check_api_health()

if not api_healthy:
    st.sidebar.error("⚠️ API is not reachable. Make sure the API is running.")
else:
    st.sidebar.success("✅ API Connected")

if page == "Predictor":
    st.header("🔮 Individual Employee Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age", 18, 60, 32)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        distance = st.slider("Distance from Home (miles)", 1, 30, 5)
        
        st.subheader("Job Information")
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"
        ])
        job_level = st.slider("Job Level", 1, 5, 2)
        overtime = st.selectbox("Overtime", ["No", "Yes"])
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    
    with col2:
        st.subheader("Compensation")
        monthly_income = st.number_input("Monthly Income ($)", 2000, 20000, 5000, step=500)
        daily_rate = st.slider("Daily Rate ($)", 100, 1500, 500)
        hourly_rate = st.slider("Hourly Rate ($)", 30, 100, 60)
        percent_hike = st.slider("Percent Salary Hike (%)", 11, 25, 15)
        stock_option = st.slider("Stock Option Level", 0, 3, 1)
        
        st.subheader("Satisfaction Metrics")
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        env_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        work_life = st.slider("Work Life Balance (1-4)", 1, 4, 3)
        relationship_sat = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Experience")
        total_years = st.slider("Total Working Years", 0, 40, 10)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_in_role = st.slider("Years in Current Role", 0, 18, 3)
        years_with_manager = st.slider("Years with Current Manager", 0, 17, 2)
        years_since_promo = st.slider("Years Since Last Promotion", 0, 15, 1)
        num_companies = st.slider("Number of Companies Worked", 0, 9, 2)
    
    with col4:
        st.subheader("Education & Training")
        education = st.slider("Education Level (1-5)", 1, 5, 3)
        education_field = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree", 
            "Human Resources", "Other"
        ])
        training_times = st.slider("Training Times Last Year", 0, 6, 2)
        performance_rating = st.slider("Performance Rating (3-4)", 3, 4, 3)
    
    if st.button("🔮 Predict Attrition Risk", type="primary", use_container_width=True):
        
        # Prepare the data
        employee_data = {
            "Age": age,
            "BusinessTravel": business_travel,
            "DailyRate": daily_rate,
            "Department": department,
            "DistanceFromHome": distance,
            "Education": education,
            "EducationField": education_field,
            "EnvironmentSatisfaction": env_satisfaction,
            "Gender": gender,
            "HourlyRate": hourly_rate,
            "JobInvolvement": 3,  # Default
            "JobLevel": job_level,
            "JobRole": job_role,
            "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": 15000,  # Default
            "NumCompaniesWorked": num_companies,
            "OverTime": overtime,
            "PercentSalaryHike": percent_hike,
            "PerformanceRating": performance_rating,
            "RelationshipSatisfaction": relationship_sat,
            "StockOptionLevel": stock_option,
            "TotalWorkingYears": total_years,
            "TrainingTimesLastYear": training_times,
            "WorkLifeBalance": work_life,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": years_since_promo,
            "YearsWithCurrManager": years_with_manager
        }
        
        with st.spinner("Calculating risk..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=employee_data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    st.markdown("---")
                    col_res1, col_res2, col_res3 = st.columns(3)
                    
                    with col_res1:
                        risk_color = {
                            "High": "#ff4b4b",
                            "Medium": "#ffa500",
                            "Low": "#00cc96"
                        }
                        st.markdown(f"### Risk Level")
                        st.markdown(f"<h1 style='color: {risk_color[result['risk_level']]};'>{result['risk_level']}</h1>", 
                                  unsafe_allow_html=True)
                    
                    with col_res2:
                        st.markdown(f"### Probability")
                        st.markdown(f"<h1>{result['probability']*100:.1f}%</h1>", 
                                  unsafe_allow_html=True)
                    
                    with col_res3:
                        st.markdown(f"### Confidence")
                        st.markdown(f"<h1>{result['confidence_score']*100:.1f}%</h1>", 
                                  unsafe_allow_html=True)
                    
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['probability'] * 100,
                        title={'text': "Attrition Risk %"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': risk_color[result['risk_level']]},
                            'steps': [
                                {'range': [0, 30], 'color': "#e6f7e6"},
                                {'range': [30, 60], 'color': "#fff4e6"},
                                {'range': [60, 100], 'color': "#ffe6e6"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("💡 Recommended Actions")
                    
                    if result['risk_level'] == "High":
                        st.error("""
                        **IMMEDIATE INTERVENTION NEEDED:**
                        - Schedule stay interview within 7 days
                        - Review compensation against market rate
                        - Check for promotion eligibility
                        - Assess workload and overtime
                        - Consider mentorship program
                        """)
                    elif result['risk_level'] == "Medium":
                        st.warning("""
                        **PROACTIVE RETENTION SUGGESTED:**
                        - Monitor engagement over next 30 days
                        - Career development discussion
                        - Review work-life balance options
                        - Check team dynamics
                        """)
                    else:
                        st.success("""
                        **LOW RISK - MAINTAIN:**
                        - Regular check-ins (quarterly)
                        - Continue current engagement practices
                        - Consider for high-potential program
                        - Document what's working well
                        """)
                    
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"Connection Error: {e}")

elif page == "Dashboard":
    st.header("📈 Workforce Analytics Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", "1,470", "IBM Dataset")
    with col2:
        st.metric("Overall Attrition", "16.1%", "-2.3%")
    with col3:
        st.metric("High Risk Today", "12", "+3")
    with col4:
        st.metric("Model Accuracy", "87.7%", "+2.1%")
    
    # Feature importance from EDA
    st.subheader("🔑 Key Drivers of Attrition")
    
    importance_data = pd.DataFrame({
        'Feature': ['Overtime', 'Monthly Income', 'Years at Company', 
                   'Age', 'Job Satisfaction', 'Distance from Home'],
        'Importance': [0.246, -0.160, -0.134, -0.159, -0.103, 0.078]
    })
    
    fig = px.bar(importance_data, x='Importance', y='Feature', 
                 orientation='h', 
                 color='Importance',
                 color_continuous_scale=['green', 'yellow', 'red'],
                 title="Feature Correlation with Attrition")
    st.plotly_chart(fig, use_container_width=True)
    
    # Two columns for insights
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("⏰ Overtime Impact")
        ot_data = pd.DataFrame({
            'Status': ['No Overtime', 'Overtime'],
            'Attrition Rate': [10.4, 30.5]
        })
        fig = px.pie(ot_data, values='Attrition Rate', names='Status',
                    title="Attrition Rate by Overtime Status",
                    color_discrete_sequence=['#00cc96', '#ff4b4b'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("💰 Income Analysis")
        st.markdown("""
        | Group | Avg Income | 
        |-------|------------|
        | Stayers | **$6,833** |
        | Leavers | **$4,787** |
        | **Gap** | **-29.9%** |
        """)
    
    # Risk by role
    st.subheader("👥 Attrition Rate by Job Role")
    role_data = pd.DataFrame({
        'Job Role': ['Sales Representative', 'Lab Technician', 'Human Resources',
                    'Sales Executive', 'Research Scientist'],
        'Attrition Rate': [39.8, 24.0, 23.0, 18.0, 14.0]
    })
    fig = px.bar(role_data, x='Attrition Rate', y='Job Role',
                 orientation='h', color='Attrition Rate',
                 color_continuous_scale='Reds',
                 title="Top 5 High-Risk Roles")
    st.plotly_chart(fig, use_container_width=True)

else:  # About page
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## Employee Attrition Prediction System
    
    This system was built for People Data and Analytics in HR Department.
    
    ### 🎯 Business Problem
    Employee attrition costs organizations 150% of annual salary per departure. 
    This system predicts individual attrition risk with **87.7% accuracy** and provides 
    actionable retention recommendations.
    
    ### 📊 Key Insights
    - **Overtime employees are 2.9x more likely to leave**
    - **Leavers earn 29.9% less than stayers**
    - **Sales Representatives have 39.8% attrition rate**
    - **Single employees at 2.0x higher risk**
    
    ### 🛠️ Technologies Used
    - **Python** (Pandas, NumPy, Scikit-learn, XGBoost)
    - **FastAPI** for production API
    - **Streamlit** for HR dashboard
    - **MLflow** for experiment tracking
    - **Docker** for containerization
    
    ### 💰 Business Impact
    With this model, the organization can save:
    - **$860,000** with 10% attrition reduction
    - **$1.72M** with 20% attrition reduction
    - **$5.7M** by targeting high-risk employees
    
    ### 👨‍💻 About the Developer
    Built with 🔥 for Data Science and Analytics.
    
    [GitHub Repository](https://github.com/Iceyma02/EMPLOYEE-ATTRITION-PREDICTION-MODEL)
    """)
    
    st.image("https://img.icons8.com/color/96/000000/api-settings.png", width=100)
    st.success("API Status: Connected" if api_healthy else "API Status: Disconnected")
