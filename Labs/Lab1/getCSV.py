# get dataset from csv
import pandas as pd


def getCSV():
    filename = "Data/sey1.csv"

    dataset = pd.read_csv(filename)
    dataset = dataset[

        ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "Class"]]
    return dataset

# --------#
