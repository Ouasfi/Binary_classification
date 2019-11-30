import numpy as np 
import pandas as pd
import models as m
from model_selection import *
from sklearn.preprocessing import label_binarize
import argparse

parser = argparse.ArgumentParser('Parameter tuning for classification on a defined dataset')
parser.add_argument('--path', type=str, default='none')
parser.add_argument('--target_name', type=str, default='Class')
parser.add_argument('--test_size', type=float, default=0.3)
parser.add_argument('--train', type=bool, default='true')
parser.add_argument('--batch_size', type=int, default=70)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--model_name', type=str, default='MLP')
parser.add_argument('--finetune', type=bool, default='False')


args = parser.parse_args()

path = args.path
target_name = args.target_name
test_size =args.test_size
train = args.train
epochs = args.epochs
batch_size = args.batch_size
model_name = args.model_name
finetune = args.finetune

df, X, y, X_train, X_test, y_train, y_test = load_data(path, target_name, test_size)
y_train = label_binarize(y_train, classes = ['ckd', 'notckd'])
y_test = label_binarize(y_test, classes = ['ckd', 'notckd'])


param_grid_MLP = {
              'epochs':[5, 10, 15 ],
              'batch_size':[ 70, 50, 60, 40],
              
              #'batch_size' :          [32, 128],
              #'optimizer' :           ['Adam'],
              #'dropout_rate' :        [0.1, 0.2, 0.3],
              #'activation' :          ['relu', 'elu']
             }
param_grid_Dt = {'criterion': ['gini', 'entropy'],  
              'max_depth' : range(3,14),
              'min_samples_leaf': range(3,4) } 

models = {'MLP': {'build_fn': m.build_MLP((24,)),'params': param_grid_MLP},
          'Decision_tree' : { 'build_fn':m.DecisionTreeModel( train = False),'params': param_grid_Dt}         
         }

if finetune :
    
    model = models[model_name]['build_fn']
    param_grid = models[model_name]['params']
    gs, fitted_model, pred = search_pipeline(X_train, X_test, y_train,  model, param_grid ,scoring_fit = 'accuracy' )
    best_parameters = get_best_parameters(gs )
    y_pred = m.predict(fitted_model, X_test)
    accuracy(y_test, y_pred>0.5)
    results = cross_validation( model, X,y, n_splits=10,  **best_parameters)

             
             

