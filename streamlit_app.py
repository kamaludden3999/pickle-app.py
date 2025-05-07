pip install streamlit 
import pickle

# Save the model
with open("logistic_model.pkl", "wb") as file:
    pickle.dump(logistic_regression_model, file)
    import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title of the app
st.title("Logistic Regression Model Deployment")

# Description
st.write("This app uses a logistic regression model to make predictions based on user inputs.")

# Define user inputs
# Example: Assuming the model has three numerical features
st.sidebar.header("Input Features")
feature1 = st.sidebar.number_input("Feature 1", min_value=0.0, max_value=100.0, value=50.0)
feature2 = st.sidebar.number_input("Feature 2", min_value=0.0, max_value=100.0, value=50.0)
feature3 = st.sidebar.number_input("Feature 3", min_value=0.0, max_value=100.0, value=50.0)

# Make predictions
if st.sidebar.button("Predict"):
    features = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    # Display results
    st.write(f"Prediction: {'Class 1' if prediction[0] == 1 else 'Class 0'}")
    st.write(f"Probability: {probability[0][1]:.2f} (Class 1)")

# Footer
st.write("Developed with Streamlit")
