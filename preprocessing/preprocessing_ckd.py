import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
class Preprocessing:

    def __init__(self, dataset):
        self.dataset = pd.read_csv(dataset, na_values="?");
        self.string_columns = {"Rbc":     {"normal": 1, "abnormal": 0},
           "Pc": {"normal": 1, "abnormal": 0},
           "Pcc": {"present": 1, "notpresent": 0},
           "Ba": {"present": 1, "notpresent": 0},
           "Htn": {"yes": 1, "no": 0},
           "Dm": {"yes": 1, "no": 0},
           "Cad": {"yes": 1, "no": 0},
           "Appet": {"good": 1, "poor": 0},
           "pe": {"yes": 1, "no": 0},
           "Ane": {"yes": 1, "no": 0}}

    def cleanup(self):
        # Remplacement des valeurs nominales par les valeurs binaires
        self.dataset.replace(self.string_columns, inplace=True)

        # Combler les valeurs nulles par la moyenne de la colonne en
        # question
        self.dataset.fillna(round(self.dataset.mean(),2), inplace=True)

    def save(self, nom):
        # Sauvegarde du tableau traites
        if (format == "csv"):
            self.dataset.to_csv(nom+".csv", sep=',', index=False)
