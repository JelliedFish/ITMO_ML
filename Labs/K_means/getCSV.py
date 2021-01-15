# get dataset from csv
import pandas as pd


def getCSV():
    filename = "Data/dataset_191_wine.csv"

    dataset = pd.read_csv(filename)
    dataset = dataset[
        ["Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium", "Total_phenols", "Flavanoids",
         "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280%2FOD315_of_diluted_wines",
         "Proline",
         "class"]]

    return dataset

# --------#
