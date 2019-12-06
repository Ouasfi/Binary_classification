import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K
from keras.optimizers import SGD, Adam

from keras.wrappers.scikit_learn import KerasClassifier
"""
Models :
______
DecisionTree
Multilayer perceptron
RandomForest
SVM
convolutionnal neural network
"""
def DecisionTreeModel(criterion = 'gini', max_depth = 3, min_samples_leaf = 5,**kwargs ):
    """
    Creating the classifier object


    Parameters :
    ___________
    criterion, max_depth, min_samples_leaf : sklearn parameters for  DecisionTreeClassifier

    train : boolean
        Define whether the output is a trained model or not. If it's set to true, "X_train" and 'y_train' 
        arguments should be provided.
   


    Return : 
    _______
    clf : DecisionTreeClassifier model.



     Author : Amine 
    """
    # Creating the classifier object

    clf = DecisionTreeClassifier(criterion = criterion,
            random_state = 100,max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    if kwargs['train']:
    	X_train = kwargs['X_train']
    	y_train = kwargs['y_train']
    	
    	clf.fit(X_train, y_train)
 
    return clf



def build_MLP( input_shape, activation = 'relu', units = [100,100], momentum=0.1, epsilon=1e-05, dropout_rate = 0, optimizer = 'Adam'):
    
    """
    build a neural network with linear blocks fellowed by non linear activations. Every block consists of
     BatchNormalization, Dropout and a  dense layer. The depth of the network can be custumized in parameter "units"


  	Parameters:
    ____________

  	input_shape : tuple of integers.
        Correspond to the shape of the input samples.
    units: list of integers.
        Define the number of units of every linear block of the network. Thus len(units) define the depth of the network. If "units" is an empty list, the built model correspond to a logistic regression preceded by a BatchNormalization.

	Return :

	model : Keras Sequential object
			Compiled model with a binary crossentropy loss and accuracy metric.

    Author : Amine  
    """


    model = Sequential()
    model.add(BatchNormalization(axis=1, momentum=momentum, epsilon=epsilon, input_shape = input_shape))
    for n_units in units: 
      model.add(Dropout(dropout_rate))
      model.add(Dense(units=n_units,  activation=activation))
      model.add(BatchNormalization(axis=1, momentum=momentum, epsilon=epsilon))

    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1, activation='sigmoid'))     
    
    model.compile(loss='binary_crossentropy', 
                      optimizer=optimizer, 
                      metrics=['accuracy'])
    
    

    return model


def build_CNN(input_shape, filters= [3], kernels = [4], activation = 'relu', optimizer= 'Adam'):
    
    """
    build a  convolutionnal neural network . Every block consists of BatchNormalization, Dropout and a  1 dimensionnal   convolution. The depth of the network can be custumized in parameters  "kernels" and "filters". It correspond to max ( len(filters), len(kernels)).


    Parameters:
    __________

    input_shape : tuple of integers
        Correspond to the shape of the input samples.
    filters: list of integers
        Define the number of filters  of every 1 dimensionnal convolution. 
    kernels : list of integers
        Define the kernel size of the filters  of every 1 dimensionnal convolution. 
    Return :
    _______

    model : Keras Sequential object
        Compiled model with a binary crossentropy loss and accuracy metric.
    Author : Amine.  
    """
    model = Sequential()
    model.add(BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05, input_shape = input_shape))
    model.add( Conv1D(filters=3, kernel_size= 4, strides =1, activation = activation))
    for (f, k) in zip(filters, kernels):
        
        model.add(BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05))
        model.add(Dropout(0.1))
        model.add(Conv1D(filters=f, kernel_size= k, strides =1, activation = activation))
        model.add(MaxPooling1D())

    model.add(Flatten())
    model.add(BatchNormalization(axis=1, momentum=0.1, epsilon=1e-05))
    model.add(Dropout(0.1))
    model.add(Dense(units=1, activation='sigmoid')) 



    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    return model 

def train(model,X_train, y_train, X_test, y_test,  batch_size = 70, epochs = 20):
    """
    Definen a function that train a keras or sklearn model on given data.


  Parameters:
  __________
  model :  keras.engine.sequential.Sequential or sklearn model
        the model to train. For keras models, the model should be already compiled. 
  X_train, y_train, X_test, y_test : np array
        data used for training and evaluation. 


  Return :
  _______
  
  history : keras.engine.sequential.Sequential or sklearn model corresponding to input model type.
        a trained model on the given data. 
  Author : Amine
    """
    


  
    is_keras_model = type(model) == keras.engine.sequential.Sequential
    if is_keras_model :

        history = model.fit(X_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=0,
                                 validation_data=(X_test, y_test),
                                 )
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test accuracy:', score[1])

    else:
        history = model.fit(X_train, y_train)
    return history

def RandomForest(n_estimators = 10,max_depth = None, criterion ='gini',**kwargs ):
    """
    Define a function which implements the random forest classifier.
    We can either train this model or not depending if we will search to optimize the hyperparameters
    Made by Louis
    
    """  
    model = RandomForestClassifier(n_estimators = n_estimators,random_state = 100,max_depth = max_depth, criterion =criterion)
    if kwargs['train']:
    	X_train = kwargs['X_train']
    	y_train = kwargs['y_train']
    	
    	model.fit(X_train, y_train)
    return(model)
    
def SVM(C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',**kwargs ):
    """
    Define a function which implements the support vector machines classifier.
    We can either train this model or not depending if we will search to optimize the hyperparameters
    Made by Louis.    
    """
    model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
    if kwargs['train']:
    	X_train = kwargs['X_train']
    	y_train = kwargs['y_train']
    	
    	model.fit(X_train, y_train)
    return(model)
    

def predict(model, X_test):
    """
    Author : Amine
    """
    return model.predict(X_test)


