import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import argparse
from visualisation import all_visualisation



parser = argparse.ArgumentParser('Parameter for cleaning dataset')
parser.add_argument('--path', type=str, default='none')
parser.add_argument('--target_name', type=str, default='Class')
parser.add_argument('--name', type=str, default='Class')


args = parser.parse_args()

path = args.path
target_name = args.target_name
name = args.name

def Processing(path, name):
    """

    Clean the data by replacing the na_values by the mean
    of the associated column. Take the path and the
    name of the saved file as arguments.

    Authors : Eric N'GUESSAN & Nicolas STUCKI
    """
    data = pd.read_csv(path, na_values='?')
    for i in data.columns:
        for j in data[i]:
            if(isinstance(j,str)):
                l = [x for x in data[i].unique() if str(x) != 'nan']
                le = LabelEncoder()
                le.fit(l)
                y = le.transform(l)
                for k in range(len(l)):
                    data[i].replace(l[k],y[k], inplace = True)

    data.fillna(round(data.mean(), 2), inplace=True)
    data.to_csv("./data/"+name+".csv", sep=',', index=False)

Processing(path,name)
all_visualisation("./data/"+name+".csv",target_name)
