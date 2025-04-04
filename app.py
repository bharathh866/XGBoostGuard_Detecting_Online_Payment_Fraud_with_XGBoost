
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Custom CSS for the navigation bar

# Load your saved model and feature names
with open('trained_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

with open('selected_features.sav', 'rb') as features_file:
    feature_names = pickle.load(features_file)

# Create the Streamlit web app
st.title("XGBoostGuard Detecting Online Payment Fraud with XGBoost")

# Sidebar navigation
nav_selection = st.sidebar.radio("Navigation", ["Model", "About"])

if nav_selection == "Model":
    st.header("Online Fraud Detection Model")
    
    # User input section
    st.subheader("Enter Transaction Details:")
    
   
    step = st.number_input("Sequence Step of the Transaction", min_value=1.0)
    amount = st.number_input("Transaction Amount", min_value=0.0)
    old_balance_org = st.number_input("Old Balance of Origin Account", min_value=0.0)
    new_balance_org = st.number_input("New Balance of Origin Account", min_value=0.0)
    old_balance_dest = st.number_input("Old Balance of Destination Account", min_value=0.0)
    new_balance_dest = st.number_input("New Balance of Destination Account", min_value=0.0)
    
    # Predict button
    if st.button("Predict"):
        # Prepare user input as a DataFrame
        user_input = pd.DataFrame({
            'step': [step],
            'amount': [amount],
            'oldbalanceOrg': [old_balance_org],
            'newbalanceOrig': [new_balance_org],
            'oldbalanceDest': [old_balance_dest],
            'newbalanceDest': [new_balance_dest],
            'isFlaggedFraud': 0
        })
        
        # Make predictions using the model
        prediction = model.predict(user_input)
        
        # Display the prediction
        if prediction[0] == 1:
            st.error("The transaction is predicted as fraud.")
        else:
            st.success("The transaction is predicted as non-fraud.")
            
elif nav_selection == "About":
    st.header("About This Project")
    
    # Project Description
    st.subheader("Project Description")
    
    st.write(project_description)
    
    # Steps Followed
    st.subheader("Steps I've Followed")
    steps_followed = """
    1. Data Preparation:
       - I began by importing the necessary libraries and installing the required packages.
       - I loaded the dataset from a CSV file named 'onlinefraud.csv'.
       - To understand the data, I checked its shape and displayed the first few rows.
       - I also dealt with missing data by removing rows with missing values.
       - Visualized feature relationships using a correlation matrix heatmap to see how different features correlate.

    2. Feature Selection and Data Splitting:
       - I divided the data into two parts: input features (X) and the target variable (y).
       - For model training and evaluation, I split the data into training and testing sets.
       - Since XGBoost requires numeric input, I excluded non-numeric columns ('nameOrig', 'nameDest', 'type') from the data.

    3. Model Building:
       - I created an XGBoost classifier model, which is known for its effectiveness in handling complex data like this.
       - I then trained the model using the training data, allowing it to learn patterns in the data.

    4. Model Evaluation:
       - To assess the model's performance, I used various evaluation techniques.
       - I made predictions on the test set and calculated the accuracy of the model to see how often it correctly classified transactions.
       - Additionally, I generated a classification report that included metrics like precision, recall, and F1-score to assess the model's performance.
       - The confusion matrix and the Receiver Operating Characteristic (ROC) curve provided visual insights into how well the model distinguishes between fraudulent and non-fraudulent transactions.

    5. User Input and Prediction:
       - I added a user interaction feature where a user can input data for a new transaction.
       - The trained model is used to predict whether the user's input is a fraudulent transaction or not.

    6. Precision-Recall Curve:
       - I included a Precision-Recall curve to evaluate the precision and recall trade-off of the model.
    """
    st.write(steps_followed)

    # Graphs and Visualizations
    st.subheader("Graphs and Visualizations")
    graphs_description = """
    The correlation matrix heatmap helps to understand how features are related to each other.
    The confusion matrix provides insights into the model's classification performance.
    The ROC curve and AUC give a holistic view of how well the model separates fraudulent and non-fraudulent transactions.
    The Precision-Recall curve assesses precision and recall, which is essential in detecting rare fraud cases.
    """
    st.write(graphs_description)
    st.image('xgboost1.png', caption='Image 1', use_column_width=True)
    st.image('xgboost2.png', caption='Image 2', use_column_width=True)
    st.image('xgboost3.png', caption='Image 3', use_column_width=True)
    st.image('xgboost4.png', caption='Image 4', use_column_width=True)
    
    # Accuracy
    st.subheader("Accuracy")
    accuracy_description = """
    I used accuracy as a primary metric to determine the model's overall correctness in classifying transactions. However, I'm aware that accuracy alone might not be enough, especially for imbalanced datasets. for this Model Accuracy: 99.00% 
    """
    st.write(accuracy_description)

    # Future Improvements
    st.subheader("Future Improvements")
    future_improvements = """
    To make my project even better, I'm considering the following:
    - Fine-tuning the model's hyperparameters for better performance.
    - Exploring feature engineering to potentially create new features that can improve fraud detection.
    - Implementing anomaly detection techniques alongside classification to enhance detection of unusual fraud patterns.
    - Integrating the model into a real-time transaction processing system for immediate fraud detection.
    - Ensuring model interpretability, so I can understand and explain why the model makes specific predictions.
    - Exploring data augmentation techniques to balance the dataset by generating synthetic data for the minority class (fraud).
    By addressing these aspects, I aim to improve the robustness and effectiveness of my fraud detection model, making it more accurate and reliable in identifying fraudulent transactions.
    """
    st.write(future_improvements)


    # GitHub link
    st.subheader("Connect and Learn More")
    st.write("You can learn more about this project on my GitHub repository.")
    st.markdown("[GitHub Repository](https://github.com/likhith1409/XGBoostGuard_Detecting_Online_Payment_Fraud_with_XGBoost")






