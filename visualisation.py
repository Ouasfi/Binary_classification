## Import the librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from preprocessing.preprocessing_ckd import Preprocessing
from pandas.plotting import scatter_matrix, parallel_coordinates
from sklearn.metrics import confusion_matrix


def plot_cm(labels, predictions, p=0.5):



    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])
    print('Total : ', np.sum(cm[1]))

def performance(features, labels, model, BATCH_SIZE ):

    predictions_baseline = model.predict(features, batch_size=BATCH_SIZE)
    baseline_results = model.evaluate(features, labels,
                                    batch_size=BATCH_SIZE, verbose=0)
    for name, value in zip(model.metrics_names, baseline_results):
    print(name, ': ', value)
    print()

    plot_cm(labels, predictions_baseline)

def plot_loss(history, label, n):
    
    # Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch,  history.history['loss'],
               color=colors[n], label='Train '+label)
    plt.semilogy(history.epoch,  history.history['val_loss'],
          color=colors[n], label='Val '+label,
          linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()


def scatter_plot_matrix(file, separator=","):
    """
    Show the scatter plot matrix
    Take as input a .csv file
    You can choose the separator,
    default is ","

    Author : Eric N'GUESSAN
    """
    df = pd.read_csv(file, sep= separator)
    scatter_matrix(df, alpha=0.2, figsize=(10, 10))
    plt.show()

def parallel_coordinates_plot(file, target, separator=","):
    """
    Show the parallel coordinate plot
    of the data
    Take as input a .csv file and the
    name of the class columns
    You can choose the separator,
    default is ","

    Authors : Eric N'GUESSAN
    """
    df = pd.read_csv(file, sep= separator)
    parallel_coordinates(df, target, color=('#556270', '#4ECDC4', '#C7F464'))
    plt.title("")
    plt.show()

def kde_plot(file, separator=","):
    """
    Show the density function of each parameters
    Take as input a .csv file
    You can choose the separator,
    default is ","

    Authors : Eric N'GUESSAN
    """
    df = pd.read_csv(file, sep= separator)

def all_visualisation(file, target, separator=","):
    """

    Authors : Eric N'GUESSAN
    """
    print("---- Plotting the scatter plot matrix... ----")
    scatter_plot_matrix(file)
    print("\n------------------------------")
    print("\n---- Plotting the parallel coordinates... ----")
    parallel_coordinates_plot(file, target)
    print("\n------------------------------")
    print("\n---- Plotting the kernel density estimation... ----")
    kde_plot(file)
    print("\n ----- Plots finished! -----")
