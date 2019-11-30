## Import the librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from preprocessing.preprocessing_ckd import Preprocessing

def scatter_plot_matrix(file, separator=","):
    """
    Show the scatter plot matrix
    Take as input a .csv file
    You can choose the separator,
    default is ","
    """
    df = pd.read_csv(file, sep= separator)
    pd.scatter_matrix(df, alpha=0.2, figsize=(10, 10))
    plt.show()

def parallel_coordinates_plot(file, target, separator=","):
    """
    Show the parallel coordinate plot
    of the data
    Take as input a .csv file and the
    name of the class columns
    You can choose the separator,
    default is ","
    """
    df = pd.read_csv(file, sep= separator)
    pd.tools.plotting.parallel_coordinates(df, target, color=('#556270', '#4ECDC4', '#C7F464'))
    plt.title("")
    plt.show()

def kde_plot(file, separator=","):
    """
    Show the density function of each parameters
    Take as input a .csv file
    You can choose the separator,
    default is ","
    """
    df = pd.read_csv(file, sep= separator)

def all_visualisation(file, target, separator=","):
    """
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