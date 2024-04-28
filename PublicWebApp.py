import pickle
import streamlit as st
from  streamlit_option_menu import option_menu
import numpy as np


#loading the saved model
diabetes_model = pickle.load(open("trained_Diabetes.sav",'rb'))
Heart_disease_model = pickle.load(open("trained_Heart.sav",'rb'))
breast_cancer_model = pickle.load(open("trained_Breast.sav",'rb'))
with open("sclaer_breast.sav", 'rb') as file:
    scaler = pickle.load(file)
# sidebar for navigate

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction Systems',
                           
                           ['Diabetes Prediction','Heart Disease Prediction','Breast Canser Prediction'],
                           
                           icons=['activity','heart-pulse-fill','virus'],
                           
                           default_index = 0)
    
    
    
    
    
    
# Prediciton Pages
if (selected == 'Diabetes Prediction'):

    #page title
    st.title('Diabetes Prediction using ML')
    
    
    
    #getting input data from user 
    #columns for input fields
    
    col1,col2,col3 = st.columns(3)
    
    with col1:
      Pregnancies = st.text_input('Number of Pregnancies')
      
    with col2:
      Glucose   = st.text_input('Glucose level')
        
    with col3:
      BloodPressure   = st.text_input('BloodPressure level')
      
    with col1:
      SkinThickness   = st.text_input('SkinThickness value')
      
    with col2:
      Insulin   = st.text_input('Insulin level')
      
    with col3:  
      BMI   = st.text_input('BMI value')
      
    with col1:
      DiabetesPedigreeFunction   = st.text_input('DiabetesPedigreeFunction')
        
    with col2:  
      Age   = st.text_input('Age')
    
    #code for prediction
    
    diab_dignosis = ''
    
    #creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI, DiabetesPedigreeFunction, Age]])
        
        if(diab_prediction[0] == 1):
             diab_dignosis = 'The person is Diabetic'
           
        else :
             diab_dignosis = 'The person is not Diabetic'
           
    st.success(diab_dignosis)     #this will display the result on screen
    
   
    
   


if (selected == 'Heart Disease Prediction'):
    
    
    st.title('Heart Disease Prediction using ML')
    
    col1,col2,col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (0 for female, 1 for male)')
    with col3:
        cp = st.text_input('Chest Pain Type (0-3)')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Cholesterol (mg/dl)')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1=true, 0=false)')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic Results (0-2)')
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1=yes, 0=no)')
    with col1:
        oldpeak = st.text_input('ST Depression Induced by Exercise Relative to Rest')
    with col2:
        slope = st.text_input('Slope of the Peak Exercise ST Segment (0-2)')
    with col3:
        ca = st.text_input('Number of Major Vessels Colored by Fluoroscopy (0-3)')
    with col1:
        thal = st.text_input('Thalassemia (0-2)')
        
    heart_diagnosis = ''
    
    if st.button('Heart Disease Test Result'):
        input_data = np.array([
            age, sex, cp, trestbps, chol, fbs, restecg, thalach,
            exang, oldpeak, slope, ca, thal
            ], dtype=float).reshape(1,-1)

        heart_prediction  = Heart_disease_model.predict(input_data)
        if (heart_prediction[0] == 1):
            heart_diagnosis = 'The person is having heart disease'
        
        else:
            heart_diagnosis = 'The person does not have any heart diasease'
    st.success(heart_diagnosis)
    
    
    
    
    
    
    
if (selected == 'Breast Canser Prediction'):
    st.title('Breast Canser Prediction using ML')
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        mean_radius = st.text_input('mean radius')
    with col2:
        mean_texture = st.text_input('mean texture')
    with col3:
        mean_perimeter = st.text_input('mean perimeter')
    with col4:
        mean_area = st.text_input('mean area')
    with col5:
        mean_smoothness = st.text_input('mean smoothness')
    with col1:
        mean_compactness = st.text_input('mean compactness')
    with col2:
        mean_concavity = st.text_input('mean concavity')
    with col3:
        mean_concave_points = st.text_input('mean concave points')
    with col4:
        mean_symmetry = st.text_input('mean symmetry')
    with col5:
        mean_fractal_dimension = st.text_input('mean fractal dimension')
    with col1:
        radius_error = st.text_input('radius error')
    with col5:
        texture_error = st.text_input('texture error')
    with col1:
        perimeter_error = st.text_input('perimeter error')
    with col2:
        area_error = st.text_input('area error')
    with col3:
        smoothness_error = st.text_input('smoothness error')
    with col4:
        compactness_error = st.text_input('compactness error')
    with col5:
        concavity_error = st.text_input('concavity error')
    with col1:
        concave_points_error = st.text_input('concave points error')
    with col2:
        symmetry_error = st.text_input('symmetry error')
    with col3:
        fractal_dimension_error = st.text_input('fractal dimension error')
    with col4:
        worst_radius = st.text_input('worst radius')
    with col5:
        worst_texture = st.text_input('worst texture')
    with col1:
        worst_perimeter = st.text_input('worst perimeter')
    with col2:
        worst_area = st.text_input('worst area')
    with col3:
        worst_smoothness = st.text_input('worst smoothness')
    with col4:
        worst_compactness = st.text_input('worst compactness')
    with col5:
        worst_concavity = st.text_input('worst concavity')
    with col1:
        worst_concave_points = st.text_input('worst concave points')
    with col2:
        worst_symmetry = st.text_input('worst symmetry')
    with col3:
        worst_fractal_dimension = st.text_input('worst fractal dimension')
        
    cancer_diagnosis = ''
    
    if st.button('Breast Cancer Test Result'):
        input_data = np.array([
            mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
            mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
            mean_fractal_dimension, radius_error, texture_error, perimeter_error,
            area_error, smoothness_error, compactness_error, concavity_error,
            concave_points_error, symmetry_error, fractal_dimension_error,
            worst_radius, worst_texture, worst_perimeter, worst_area,
            worst_smoothness, worst_compactness, worst_concavity,
            worst_concave_points, worst_symmetry, worst_fractal_dimension
            ], dtype=float).reshape(1, -1)
    
        input_data_scaled = scaler.transform(input_data)
        cancer_prediction = breast_cancer_model.predict(input_data_scaled)
        
        if cancer_prediction[0] == 1:
            cancer_diagnosis = 'The person is having breast cancer'
        else:
                cancer_diagnosis = 'The person does not have any breast disease'
                
    st.success(cancer_diagnosis)
