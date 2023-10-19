# XGBoostGuard - Detecting Online Payment Fraud with XGBoost

Streamlit link: https://xgboostguard.streamlit.app/

## Description

XGBoostGuard is a fraud detection model built using the powerful XGBoost machine learning algorithm. This model is designed to automatically identify potentially fraudulent financial transactions based on various transaction-related features.

## Table of Contents

- [About This Project](#xgboostguard---detecting-online-payment-fraud-with-xgboost)
  - [Description](#description)
- [Table of Contents](#table-of-contents)
- [Steps I've Followed](#steps-ive-followed)
  - [Data Preparation](#data-preparation)
  - [Feature Selection and Data Splitting](#feature-selection-and-data-splitting)
  - [Model Building](#model-building)
  - [Model Evaluation](#model-evaluation)
  - [User Input and Prediction](#user-input-and-prediction)
  - [Precision-Recall Curve](#precision-recall-curve)
- [Graphs and Visualizations](#graphs-and-visualizations)
  - [Accuracy](#accuracy)
- [Future Improvements](#future-improvements)

## Steps I've Followed

### Data Preparation

1. **Libraries and Packages**: I began by importing the necessary libraries and installing the required packages.

2. **Data Loading**: I loaded the dataset from a CSV file named 'onlinefraud.csv'.https://drive.google.com/file/d/1O90u1b67QqpEEcQT-iEOs0J72zmo06w6/view?usp=sharing

3. **Data Exploration**:
    - Checked the dataset's shape to understand its size.
    - Displayed the first few rows to get an initial look at the data.
    - Dealt with missing data by removing rows with missing values.

4. **Data Visualization**: Visualized feature relationships using a correlation matrix heatmap to see how different features correlate.

### Feature Selection and Data Splitting

1. **Data Splitting**:
    - Divided the data into two parts: input features (X) and the target variable (y).
    - Split the data into training and testing sets for model training and evaluation.

2. **Feature Engineering**:
    - Excluded non-numeric columns ('nameOrig', 'nameDest', 'type') from the data since XGBoost requires numeric input.

### Model Building

1. Created an XGBoost classifier model, known for its effectiveness in handling complex data.

2. Trained the model using the training data, allowing it to learn patterns in the data.

### Model Evaluation

1. To assess the model's performance, I used various evaluation techniques:
    - Made predictions on the test set and calculated the accuracy of the model.
    - Generated a classification report that included metrics like precision, recall, and F1-score.
    - Utilized the confusion matrix and the Receiver Operating Characteristic (ROC) curve to visualize the model's performance in distinguishing between fraudulent and non-fraudulent transactions.

### User Input and Prediction

1. Added a user interaction feature where a user can input data for a new transaction.

2. Used the trained model to predict whether the user's input is a fraudulent transaction or not.

### Precision-Recall Curve

1. Included a Precision-Recall curve to evaluate the precision and recall trade-off of the model.

## Graphs and Visualizations

The following visualizations provide valuable insights into the model's performance:
- **Correlation Matrix Heatmap**: Helps understand how features are related to each other.
- **Confusion Matrix**: Provides insights into the model's classification performance.
- **ROC Curve and AUC**: Give a holistic view of how well the model separates fraudulent and non-fraudulent transactions.
- **Precision-Recall Curve**: Assesses precision and recall, essential in detecting rare fraud cases.

![xgboost1](https://github.com/likhith1409/XGBoostGuard_Detecting_Online_Payment_Fraud_with_XGBoost/assets/91020626/b401df13-3098-483d-8c7c-ddb563498069)
![xgboost2](https://github.com/likhith1409/XGBoostGuard_Detecting_Online_Payment_Fraud_with_XGBoost/assets/91020626/51dfe959-9215-41ea-a007-698f6a976ea7)
![xgboost3](https://github.com/likhith1409/XGBoostGuard_Detecting_Online_Payment_Fraud_with_XGBoost/assets/91020626/87033fd3-33ec-4687-8534-c7a2fd9b873b)
![xgboost4](https://github.com/likhith1409/XGBoostGuard_Detecting_Online_Payment_Fraud_with_XGBoost/assets/91020626/2fae760a-4a3f-4c9b-9021-1586f1642ecc)

### Accuracy

I used accuracy as a primary metric to determine the model's overall correctness in classifying transactions. However, I'm aware that accuracy alone might not be enough, especially for imbalanced datasets. For this model, accuracy is 99.00%.

## Future Improvements

To make my project even better, I'm considering the following:

1. **Hyperparameter Fine-Tuning**: Fine-tuning the model's hyperparameters for better performance.

2. **Feature Engineering**: Exploring feature engineering to potentially create new features that can improve fraud detection.

3. **Anomaly Detection**: Implementing anomaly detection techniques alongside classification to enhance detection of unusual fraud patterns.

4. **Real-Time Integration**: Integrating the model into a real-time transaction processing system for immediate fraud detection.

5. **Model Interpretability**: Ensuring model interpretability, so I can understand and explain why the model makes specific predictions.

6. **Data Augmentation**: Exploring data augmentation techniques to balance the dataset by generating synthetic data for the minority class (fraud). By addressing these aspects, I aim to improve the robustness and effectiveness of my fraud detection model, making it more accurate and reliable in identifying fraudulent transactions.




