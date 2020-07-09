import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing

st.write("""
# Diabetes Prediction App
This app predicts Diabetes using data from https://www.kaggle.com/uciml/pima-indians-diabetes-database.
And it's part of my portolio: https://nbviewer.jupyter.org/github/BrunoPTeruya/Portfolio/blob/master/Diabetes%20Pima.ipynb
""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
def user_input_features():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17 , 1)
    Glucose = st.sidebar.slider('Glucose', 0.0, 200.0, 32.0)
    BloodPressure = st.sidebar.slider('BloodPressure', 0.0, 122.0, 20.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 8.0)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.0, 2.42, 1.0)
    data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
diabetes_raw = pd.read_csv('diabetes_clean.csv')
diabetes = diabetes_raw.drop(columns=['Outcome'])
df = pd.concat([input_df,diabetes],axis=0)

# Displays the user input features
st.subheader('User Input features')
st.write(input_df)
# Reads in saved classification model

load_clf = pickle.load(open('diabetes_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

st.subheader('Prediction')
types = np.array([0, 1])
st.write(types[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
