import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

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

"""
def DecisionTreeModel(criterion = 'gini', max_depth = 3, min_samples_leaf = 5,**kwargs ):
 
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
    build a neural network with blocks of linear blocks fellowed by non linear activations. Every blocks consists of
     BatchNormalization, Dropout and a  dense layer. The depth of the network can be custumized in parameter "units"


  	Parameters:
  	____________

  	input_shape : tuple of integers
  		Correspond to the shape of the input samples.
	units: list of integers
		Define the number of units of every linear block of the network. Thus len(units) define the depth of the network. If "units" is an empty list 
		the built model correspond to a logistic regression preceded by a BatchNormalization.

	Return :

	model : Keras Sequential object
			Compiled model with a binary crossentropy loss and accuracy metric.
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

def train_nn(model,X_train, y_train, X_test, y_test,  batch_size = 70, epochs = 20):
	
	history = model.fit(X_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 validation_data=(X_test, y_test),
                                 )

	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test accuracy:', score[1])
	return history

def predict(model, X_test):

	return model.predict(X_test)


