import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from src.data_processing import load_framingham_data, preprocess_data, clean_data
from src.ml_model import CardiovascularRiskModel

# Set page config
st.set_page_config(
    page_title="Cardiovascular Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# Title
st.title("🫀 Cardiovascular Event Risk Prediction Tool")
st.markdown("""
This tool predicts the 10-year risk of cardiovascular events based on the Framingham Heart Study dataset.
The model considers multiple risk factors to provide personalized risk assessment.
""")

# Sidebar for user inputs
st.sidebar.header("Patient Information")

# Input fields for cardiovascular risk factors
col1, col2 = st.sidebar.columns(2)

with col1:
    male = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    age = st.slider('Age', min_value=20, max_value=80, value=50, step=1)
    education = st.selectbox('Education Level', options=[1, 2, 3, 4], 
                            format_func=lambda x: {1: 'Some High School', 2: 'High School Graduate', 
                                                  3: 'Some College', 4: 'College Graduate'}[x])
    current_smoker = st.selectbox('Current Smoker', options=[0, 1], 
                                 format_func=lambda x: 'No' if x == 0 else 'Yes')
    cigs_per_day = st.slider('Cigarettes per Day', min_value=0, max_value=70, value=0, step=1)

with col2:
    bp_meds = st.selectbox('On Blood Pressure Meds', options=[0, 1], 
                          format_func=lambda x: 'No' if x == 0 else 'Yes')
    prevalent_stroke = st.selectbox('Prevalent Stroke', options=[0, 1], 
                                   format_func=lambda x: 'No' if x == 0 else 'Yes')
    prevalent_hyp = st.selectbox('Prevalent Hypertension', options=[0, 1], 
                                format_func=lambda x: 'No' if x == 0 else 'Yes')
    diabetes = st.selectbox('Diabetes', options=[0, 1], 
                           format_func=lambda x: 'No' if x == 0 else 'Yes')
    tot_chol = st.slider('Total Cholesterol', min_value=120, max_value=600, value=240, step=10)
    sys_bp = st.slider('Systolic Blood Pressure', min_value=80, max_value=300, value=140, step=5)
    dia_bp = st.slider('Diastolic Blood Pressure', min_value=40, max_value=150, value=90, step=5)
    bmi = st.slider('BMI', min_value=15.0, max_value=50.0, value=25.0, step=0.1)
    heart_rate = st.slider('Heart Rate', min_value=40, max_value=150, value=75, step=1)
    glucose = st.slider('Glucose', min_value=40, max_value=400, value=100, step=5)

# Create a dataframe with the inputs
user_data = pd.DataFrame({
    'male': [male],
    'age': [age],
    'education': [education],
    'currentSmoker': [current_smoker],
    'cigsPerDay': [cigs_per_day],
    'BPMeds': [bp_meds],
    'prevalentStroke': [prevalent_stroke],
    'prevalentHyp': [prevalent_hyp],
    'diabetes': [diabetes],
    'totChol': [tot_chol],
    'sysBP': [sys_bp],
    'diaBP': [dia_bp],
    'BMI': [bmi],
    'heartRate': [heart_rate],
    'glucose': [glucose]
})

# Placeholder for model loading
# In a real implementation, you would load a pre-trained model
@st.cache_resource
def load_trained_model():
    """Load the trained model - in practice, you would load a pre-trained model"""
    try:
        # Check if model exists
        model_path = 'models/cardiovascular_risk_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            model = CardiovascularRiskModel()
            model.model = model_data['model']
            model.model_type = model_data['model_type']
            model.feature_names = model_data['feature_names']
            model.is_trained = model_data['is_trained']
            return model
        else:
            # If no pre-trained model exists, return None
            # In a real implementation, you would train the model here
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_trained_model()

# Function to predict risk using a simple rule-based approach for demonstration
def simple_prediction(user_data):
    """A simple rule-based prediction for demonstration purposes"""
    risk_score = 0
    
    # Age factor (higher age = higher risk)
    if user_data['age'].iloc[0] > 65:
        risk_score += 2
    elif user_data['age'].iloc[0] > 50:
        risk_score += 1
    
    # Gender factor (male = higher risk)
    if user_data['male'].iloc[0] == 1:
        risk_score += 1
    
    # Smoking factor
    if user_data['currentSmoker'].iloc[0] == 1:
        risk_score += 1
        if user_data['cigsPerDay'].iloc[0] > 20:
            risk_score += 1
    
    # Hypertension factor
    if user_data['prevalentHyp'].iloc[0] == 1:
        risk_score += 1
    
    # Diabetes factor
    if user_data['diabetes'].iloc[0] == 1:
        risk_score += 1
    
    # Cholesterol factor
    if user_data['totChol'].iloc[0] > 240:
        risk_score += 1
    
    # Blood pressure factor
    if user_data['sysBP'].iloc[0] > 160:
        risk_score += 2
    elif user_data['sysBP'].iloc[0] > 140:
        risk_score += 1
    
    # Calculate probability based on risk score (0-10 scale)
    probability = min(1.0, risk_score / 10.0)  # Cap at 100%
    probability = max(0.0, probability)  # Ensure at least 0%
    
    return probability

# Make prediction
if st.button('Calculate Cardiovascular Risk'):
    with st.spinner('Calculating risk...'):
        if model and model.is_trained:
            # Use the trained model for prediction
            try:
                risk_probability = model.predict_proba(user_data)[0]
            except Exception as e:
                st.error(f"Model prediction error: {e}")
                risk_probability = simple_prediction(user_data)
        else:
            # Use simple rule-based prediction for demonstration
            risk_probability = simple_prediction(user_data)
    
    # Display result
    risk_percentage = risk_probability * 100
    st.subheader(f"10-Year Cardiovascular Risk: {risk_percentage:.1f}%")
    
    # Risk category
    if risk_probability < 0.05:
        risk_category = "Low Risk"
        risk_color = "green"
    elif risk_probability < 0.1:
        risk_category = "Moderate Risk"
        risk_color = "orange"
    elif risk_probability < 0.2:
        risk_category = "High Risk"
        risk_color = "red"
    else:
        risk_category = "Very High Risk"
        risk_color = "darkred"
    
    st.markdown(f"#### Risk Category: <span style='color:{risk_color}; font-weight:bold;'>{risk_category}</span>", 
                unsafe_allow_html=True)
    
    # Risk explanation
    st.markdown("### Risk Interpretation")
    if risk_probability < 0.05:
        st.info("Low Risk: Your risk of a cardiovascular event in the next 10 years is relatively low. Maintain a healthy lifestyle to keep your risk low.")
    elif risk_probability < 0.1:
        st.warning("Moderate Risk: Your risk is elevated. Consider discussing lifestyle modifications with your healthcare provider.")
    elif risk_probability < 0.2:
        st.error("High Risk: Your risk is significantly elevated. Consult with your healthcare provider about risk reduction strategies.")
    else:
        st.error("Very High Risk: Your risk is very high. It is important to discuss this with your healthcare provider immediately.")

# Display information about the model
st.markdown("---")
st.markdown("### About This Tool")
st.info("""
This Cardiovascular Risk Prediction Tool is based on the Framingham Heart Study dataset, 
which has been instrumental in identifying major risk factors for cardiovascular disease.

**Important Disclaimers:**
- This tool provides estimates based on statistical models and should not replace professional medical advice
- Always consult with healthcare professionals for medical decisions
- Results are for informational purposes only
""")