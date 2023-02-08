#!/usr/bin/python3
# tain.py
# Xavier Vasques 13/04/2021

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from joblib import dump
from sklearn import preprocessing

def train():

    # Load directory paths for persisting model

    #MODEL_DIR = '/Volumes/GL/TECH/IIITH/TA_Session_Material/Material/EEG-letters-main'
    MODEL_PATH_SVC = 'svc.joblib'
    MODEL_PATH_RF = 'rf.joblib'
    MODEL_PATH_DT = 'dt.joblib'
    MODEL_PATH_NN = 'nn.joblib'
      
    # Load, read and normalize training data
    training = "./train.csv"
    data_train = pd.read_csv(training)
        
    y_train = data_train['# Letter'].values
    X_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis = 1)

    print("Shape of the training data")
    print(X_train.shape)
    print(y_train.shape)
        
    # Data normalization (0,1)
    X_train = preprocessing.normalize(X_train, norm='l2')
    
    # Models training
    
    # Support Vector Classifier (Default parameters)
    clf_svc = SVC()
    clf_svc.fit(X_train, y_train)
    
    # Save model
    from joblib import dump
    dump(clf_svc, MODEL_PATH_SVC)

    # Random Forest Classifier (Default parameters)
    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train, y_train)
    
    # Save model
    from joblib import dump
    dump(clf_rf, MODEL_PATH_RF)

    # Decision Tree Classifier (Default parameters)
    #clf_dt = DecisionTreeClassifier()
    #clf_dt.fit(X_train, y_train)
    
    # Save model
    #from joblib import dump
    #dump(clf_dt, MODEL_PATH_DT)
        
    # Neural Networks multi-layer perceptron (MLP) algorithm
    #clf_NN = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(500,), random_state=0, max_iter=1000)
    #clf_NN.fit(X_train, y_train)
       
    # Secord model
    #from joblib import dump, load
    #dump(clf_NN, MODEL_PATH_NN)
        
if __name__ == '__main__':
    train()
