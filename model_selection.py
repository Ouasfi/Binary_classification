# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import pandas as pd
import numpy as np
import keras
from random import seed
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score, cohen_kappa_score, log_loss
from sklearn.model_selection import GridSearchCV , train_test_split , RandomizedSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential





# roc_auc_score, roc_curve
from sklearn.metrics import roc_auc_score, roc_curve

def load_data(filepath, target_name = None, test_size = 0):  

    """
    load data from filepath 

    Parameters :
    ___________

    filepath : string 
        filename or path where data is stored.
    target_name : string
        name of the variable to be predicted. 
    test_size : float
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset 
        to include in the test split. If int, represents the absolute number of test samples.
        If None, it will be set to 0.25.
    
    Return :
    _______
    df : pandas dataframe 
    X : np array 
        input array extracted from loaded data 
    Y : np array
        target array extracted from loaded data
     X_train, X_test, y_train, y_test : np arrays
       train-test split of inputs.  

    """ 
    df = pd.read_csv(filepath)
    if target_name is None:
        X = df.drop(df.columns[-1],axis=1)
        y = df[df.columns[-1]]

    else :
        X = df.drop(target_name, axis=1).values
        y = df[target_name].values
    
    if test_size !=0:
        X_train, X_test, y_train, y_test = train_test_split( X,y, 
                test_size = test_size, random_state = 101)
        print("X_train :", X_train.shape, " y_train :", y_train.shape, "X_test : ", X_test.shape, "y test : ",y_test.shape )
        return df, X, y, X_train, X_test, y_train, y_test

    return df, X, y




def search_pipeline(X_train_data, X_test_data, y_train_data, 
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       do_probabilities = False, search_mode = 'GridSearchCV', n_iterations = 0):
    """
    Parameters tuning for sklearn and keras models. **is_keras_model** should be set to True if a keras model is used. 

    """


    fitted_model = None
    is_keras_model = type(model) == keras.engine.sequential.Sequential
    if is_keras_model :
        keras_model = model # sinon, il y a une erreur bizarre qui apparait 
        model = KerasClassifier(build_fn=  lambda : keras_model, verbose=0)
    if(search_mode == 'GridSearchCV'):
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid, 
            cv=cv, 
            n_jobs=-1, 
            scoring=scoring_fit,
            verbose=2
        )
        fitted_model = gs.fit(X_train_data, y_train_data)

    elif (search_mode == 'RandomizedSearchCV'):
        gs = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid, 
            cv=cv,
            n_iter=n_iterations,
            n_jobs=-1, 
            scoring=scoring_fit,
            verbose=2
        )
        fitted_model = gs.fit(X_train_data, y_train_data)
    
    
    if(fitted_model != None):
        if do_probabilities:
            pred = fitted_model.predict_proba(X_test_data)
        else:
            pred = fitted_model.predict(X_test_data)
            
        return gs, fitted_model, pred

def get_best_parameters (grid ):
    print("les meilleurs paramètres sont " ,grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
    print("les meilleur score est " ,grid.best_score_) 

    return grid.best_params_

def plot_roc(y_test,y_pred,model):
    """
    plot roc curve


    Parameters :
    ___________

    y_test : np array
            target data used for testing the model performance
    y_pred : np array with the same same of y_test 
            predicted output of used "model"
    """
    # AUC score
    auc_score = roc_auc_score(y_test, y_pred)
     
    # fpr, tpr, threshold

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
      
    # ROC curve plot
    plt.plot([0, 1], [0, 1], 'k--')

    plt.plot(fpr, tpr, label= model + ' Classifier (AUC = {: .2f})'.format(auc_score))

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.legend(loc='lower right')

    #plt.title(model + ' Classifier ROC Curve')

    plt.show()

def accuracy(y_test, y_pred):

    """
    Parameters :
    ___________

    y_test : np array
            target data used for testing the model performance
    y_pred : np array with the same same of y_test 
            predicted output of used "model"

    """
     
    print("Confusion Matrix: \n",
    confusion_matrix(y_test,y_pred))
     
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
     
    print("Report : \n",
    classification_report(y_test, y_pred))


def cross_validation(model, X,Y,epochs=100, batch_size=70, n_splits=10, **kwargs):

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    is_keras_model = type(model) == keras.engine.sequential.Sequential
    

    if is_keras_model:
        print('Using Keras classifier')
        keras_model = model
        model = KerasClassifier(build_fn=lambda :keras_model, epochs=epochs, batch_size=batch_size, verbose=0)
    # evaluate using 1k-fold cross validation
    kfold = StratifiedKFold(n_splits=n_splits,  random_state=seed)
    results = cross_val_score(model, X, Y, cv=kfold)
    print('avg accuracy :', results.mean())
    return results
























    
