import streamlit as st
import numpy as np
import pandas as pd
import pickle

from PIL import Image
from sklearn import datasets

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_option('deprecation.showfileUploaderEncoding', False)

img = Image.open('iuli_logo.png')
st.set_page_config(page_title = "Thesis Sample", page_icon = img)

st.title('Comparison of Algorithm Performance in Breast Cancer Detection')
st.markdown("""
            This app is the result of the implementation of the Breast Cancer which aims to find out the comparison of the results 
            of the accuracy of 3 classification algorithms with dataset. 
            """)

heart_data = pd.read_csv('heartdisease1.csv')
heartdisease = heart_data.drop(columns = ['target'])
df = pd.concat([heartdisease], axis = 0)

X = heart_data.drop(columns = ['target'])
y = heart_data['target']

st.subheader('Dataset')
st.markdown("""
            The data used is taken from the Kaggle website with a CSV file type consisting of a total of 1025 pieces of data. 
            The total data of 496 patients had coronary heart disease and 529 patients did not have coronary heart disease. 
            This data was taken in 1988 from a hospital and consists of four databases namely Cleveland, Hungary, Switzerland, and Long Beach V.
            """)
st.write(df)

st.subheader('Exploratory Data Analysis')
st.write('Shape of the data is ', X.shape)

st.sidebar.header('Model Selection')
algorithm = st.sidebar.selectbox('Select the algorithm model.', ('GA', 
                                                          'FL',
                                                          'NN'))
    
def newparam(choose_algorithm):
    params = dict()
    if choose_algorithm == 'FL':
        max_depthCART = st.sidebar.slider('max_depth', 2, 15)
        params['max_depthCART'] = max_depthCART
        
    elif choose_algorithm == 'NN':
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
        max_depthRF = st.sidebar.slider('max_depth', 2, 15)
        params['max_depthRF'] = max_depthRF
        
    return params

params = newparam(algorithm)
            
def classification(choose_algorithm, params):
    algo = None
    if choose_algorithm == 'GA':
        algo = GaussianNB()
    elif choose_algorithm == 'FL':
        algo = DecisionTreeClassifier(max_depth = params['max_depthCART'],
                                      random_state = 0)
    else:
        algo = RandomForestClassifier(n_estimators = params['n_estimators'],
                                      max_depth = params['max_depthRF'],
                                      random_state = 0)
    return algo

algo = classification(algorithm, params)

score = cross_val_score(algo, X, y, cv = 10)

st.subheader('Results')
st.write(f'Algorithm: {algorithm}')
st.write('Accuracy: ', round(score.mean()*100,2), '%')
st.write('Standard Deviation: ', round(score.std()*100,2), '%')
