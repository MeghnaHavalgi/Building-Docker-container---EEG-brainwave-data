#!/usr/bin/python3
# inference.py
# Xavier Vasques 13/04/2021


import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import load
from sklearn import preprocessing
from sklearn.metrics import classification_report



def inference():

    MODEL_DIR = '/Volumes/GL/TECH/IIITH/TA_Session_Material/Material/EEG-letters-main'
    MODEL_PATH_SVC = 'svc.joblib'
    MODEL_PATH_RF = 'rf.joblib'
    MODEL_PATH_DT = 'dt.joblib'
    MODEL_PATH_NN = 'nn.joblib'
        
    # Load, read and normalize training data
    testing = "test.csv"
    data_test = pd.read_csv(testing)
        
    y_test = data_test['# Letter'].values
    X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
   
    print("Shape of the test data")
    print(X_test.shape)
    print(y_test.shape)
    
    # Data normalization (0,1)
    X_test = preprocessing.normalize(X_test, norm='l2')
    
    # Models training
    
    # SVM Classifier Model
    print(MODEL_PATH_SVC)
    clf_svc = load(MODEL_PATH_SVC)
    print("SVM score and classification:")
    prediction_svc = clf_svc.predict(X_test)
    report_svc = classification_report(y_test, prediction_svc)

    print(clf_svc.score(X_test, y_test))
    print('SVC Prediction:', prediction_svc)
    print('SVC Classification Report:', report_svc)
    print('-'*100)

    # RF Classification Model
    print(MODEL_PATH_RF)
    clf_rf = load(MODEL_PATH_RF)
    print("Random Forest score and classification:")
    prediction_rf = clf_rf.predict(X_test)
    report_rf = classification_report(y_test, prediction_rf)

    print(clf_rf.score(X_test, y_test))
    print('RF Prediction:', prediction_rf)
    print('RF Classification Report:', report_rf)
    print('-'*100)

    # DT Classification Model
    #print(MODEL_PATH_DT)
    #clf_dt = load(MODEL_PATH_DT)
    #print("Decision Tree score and classification:")
    #prediction_dt = clf_dt.predict(X_test)
    #report_dt = classification_report(y_test, prediction_dt)

    #print(clf_dt.score(X_test, y_test))
    #print('DT Prediction:', prediction_dt)
    #print('DT Classification Report:', report_dt)
    #print('-'*100)      

    # NN Classifcation Model
    #clf_nn = load(MODEL_PATH_NN)
    #print("NN score and classification:")
    #prediction_nn = clf_nn.predict(X_test)
    #report_nn = classification_report(y_test, prediction_nn)


    #print(clf_nn.score(X_test, y_test))
    #print('NN Prediction:', prediction_nn)
    #print('NN Classification Report:', report_nn)
    
    
if __name__ == '__main__':
    inference()
