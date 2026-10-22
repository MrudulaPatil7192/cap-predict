import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Prediction App",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Capstone Prediction App")
st.write("Enter the required features below to get a prediction.")

# Load your trained model (Ensure your model file, e.g., 'model.pkl', is in your repository)
@st.cache_resource
def load_model():
    try:
        return joblib.load('model.pkl')
    except Exception as e:
        return None

model = load_model()

# Input form
with st.form("prediction_form"):
    st.subheader("Input Features")
    
    # Customize these inputs based on your model's expected features
    feature_1 = st.number_input("Feature 1", value=0.0)
    feature_2 = st.number_input("Feature 2", value=0.0)
    
    submit_button = st.form_submit_button(label="Make Prediction")

if submit_button:
    if model is None:
        st.error("Model file (`model.pkl`) not found or failed to load. Please make sure your model file is uploaded in the repository.")
    else:
        try:
            # Example prediction logic (adjust according to your preprocessing/model)
            input_data = pd.DataFrame([[feature_1, feature_2]], columns=['feature_1', 'feature_2'])
            prediction = model.predict(input_data)
            
            st.success("Prediction generated successfully!")
            st.metric(label="Predicted Result", value=str(prediction[0]))
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
