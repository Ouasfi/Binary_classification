# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import pandas as pd
import numpy as np
from random import seed
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn import model_selection




# roc_auc_score, roc_curve
from sklearn.metrics import roc_auc_score, roc_curve




def plot_roc(y_test,y_pred,model):

    """
    plot roc curve
    """

    # AUC score

    auc_score = roc_auc_score(y_test, y_pred)

        

    # fpr, tpr, threshold

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

        

    # ROC curve plot
    plt.plot([0, 1], [0, 1], 'k--')

    #plt.plot(fpr, tpr, label= model + ' Classifier (AUC = {: .2f})'.format(auc_score))

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.legend(loc='lower right')

    #plt.title(model + ' Classifier ROC Curve')

    plt.show()

def search_pipeline(X_train_data, X_test_data, y_train_data, 
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       do_probabilities = False, search_mode = 'GridSearchCV', n_iterations = 0):
    fitted_model = None
    
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
        rs = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid, 
            cv=cv,
            n_iter=n_iterations,
            n_jobs=-1, 
            scoring=scoring_fit,
            verbose=2
        )
        fitted_model = rs.fit(X_train_data, y_train_data)
    
    
    if(fitted_model != None):
        if do_probabilities:
            pred = fitted_model.predict_proba(X_test_data)
        else:
            pred = fitted_model.predict(X_test_data)
            
        return fitted_model, pred

def load_data():   
    filename = r"C:\Users\faull\OneDrive\Documents\Scolaire\IMT\MCE\Machine learning\Projet\data_classification\data_banknote_authentication.txt"
    df = pd.read_csv(filename, names=['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class'])
    X = df.drop('Class', axis=1)
    y = df['Class']  
    return df, X, y


df, X, y = load_data()
print(df.head(8))

  
X_train, X_test, y_train, y_test = train_test_split( X,y, 
                test_size = 0.50, random_state = 101) 
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
y_train = y_train[:,np.newaxis]
y_test = y_test[:,np.newaxis]
print("X_train :", np.shape(X_train), " y_train :", np.shape(y_train), "X_tesy : ", X_test, "y tesy : ",y_test )
model = SVC() 
#model.fit(X_train,y_train)
#pred = model.predict(X_test)
#print("pred : " , pred)
#model.fit(X_train, y_train) 
#  
## print prediction results 
#predictions = model.predict(X_test) 
#print(classification_report(y_test, predictions)) 


  
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000, 2000],  
              'gamma': [1, 0.1, 0.01,0.05,0.005, 0.001, 0.0001],
              #'degree' : [1,3],
              'kernel': ['rbf']}  
  
#grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
#  
## fitting the model for grid search 
#grid.fit(X_train, y_train) 
model, pred = search_pipeline(X_train, X_test, y_train, model, 
                                 param_grid, cv=5, scoring_fit='accuracy',
                                 search_mode = 'RandomizedSearchCV', n_iterations = 15)

# print best parameter after tuning 
print("les meilleurs paramètres sont " ,model.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print("les meilleur score est " ,model.best_score_) 
print(len(y_test), " et ",len(pred))
plot_roc(y_test,pred,model)
report = classification_report(y_test, pred)
print(report)

#results = model_selection.cross_val_score(model, X, y, scoring='accuracy')

#print("Accuracy: ",results.mean(), results.std())

#grid_predictions = grid.predict(X_test) 
#  
## print classification report 
#print(classification_report(y_test, grid_predictions)) 






    
