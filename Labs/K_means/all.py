import numpy as np
import matplotlib.pyplot as plt

from Labs.K_means.Normalization import Normalization
from Labs.K_means.PrepareDataset import getNaive
from Labs.K_means.getCSV import getCSV


class K_means:

    def __init__(self, X, K, n):
        self.X = X
        self.K = K
        self.n = n
        self.mi = np.zeros((K, n - 1))
        self.A = np.zeros(len(X))
        self.A_old = np.zeros(len(X))
        self.clusters = {}
        self.changed = True

    def init(self):
        for i in range(self.K):
            self.mi[i] = np.random.sample(len(self.X[0]) - 1)
        for i in range(len(self.X)):
            self.A[i] = -1
        self.changed = True

    def dist(self, xi, mia):
        return np.linalg.norm(xi[:-1] - mia) ** 2

    def new_centers(self):
        self.prepareClusters()

        for i in range(self.K):
            for s in range(len(self.X[0]) - 1):
                for j in range(len(self.clusters[i])):
                    self.mi[i][s] += self.clusters[i][j][s]
                self.mi[i][s] /= len(self.clusters[i])

    def clusterization(self):
        self.init()  # init the first step

        while self.changed:
            self.changed = False
            for i in range(len(self.X)):
                self.A_old[i] = self.A[i]

                min = self.dist(self.X[i], self.mi[0])
                min_a = 0
                for a in range(len(self.mi)):
                    if self.dist(self.X[i], self.mi[a]) < min:
                        min = self.dist(self.X[i], self.mi[a])
                        min_a = a

                self.A[i] = min_a

                if self.A[i] != self.A_old[i]:
                    self.changed = True

            self.new_centers()

        for i in range(len(self.A)):
            self.A[i] += 1

        return self.A, self.clusters

    def rand_Index(self):
        TP = 0
        TN = 0
        FN = 0
        FP = 0

        for i in range(len(self.X)):
            for j in range(len(self.X)):

                if self.X[j][self.n - 1] == self.X[i][self.n - 1]:
                    if self.A[j] == self.A[i]:
                        TP += 1

                elif self.X[j][self.n - 1] != self.X[i][self.n - 1]:
                    if self.A[j] == self.A[i]:
                        TN += 1

                elif self.X[j][self.n - 1] == self.X[i][self.n - 1]:
                    if self.A[j] != self.A[i]:
                        FP += 1

                elif self.X[j][self.n - 1] != self.X[i][self.n - 1]:
                    if self.A[j] != self.A[i]:
                        FN += 1

        return (TP + FN) / (TP + TN + FP + FN)

    def prepareClusters(self):
        self.clusters = {}

        for j in range(self.K):
            self.clusters[j] = []
            for i in range(len(self.A)):
                if self.A[i] == j:
                    self.clusters[j].append(np.array(self.X[i]))

    def silhouette(self):
        N = len(self.X)
        s = []
        self.prepareClusters()

        for i in range(self.K):

            for j in range(len(self.clusters[i])):
                if len(self.clusters[i]) == 0:
                    continue
                else:
                    for z in  range(len(self.clusters[i])):
                     A_s = np.linalg.norm(self.clusters[i][j] - self.clusters[i][z])
                     ans_A = sum(A_s) / (len(A_s) - 1)
                     B_s = []

# functions for normalization

def minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        if i == len(dataset[0]) - 1:
            continue
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max()
        minmax.append([value_min, value_max])
    return minmax


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            if i == len(row) - 1:  # exclude labels
                continue
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    return dataset


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


# -------#


def Normalization(dataset):
    mx = minmax(dataset.values)
    normalized_dataset = normalize(dataset.values, mx)

    return normalized_dataset


dataset = getCSV()
normalized_dataset = Normalization(dataset)
pdataset = getNaive(normalized_dataset)
for k in range(2,22):
    k_means = K_means(pdataset, k, len(pdataset[0]))

A, clusters = k_means.clusterization()
print(k_means.rand_Index())
print(k_means.silhouette())