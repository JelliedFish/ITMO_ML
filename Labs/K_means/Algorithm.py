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

    def PCA(self, X, n):
        CM = np.cov(X.T) / X.shape[0]

        values, vectors = np.linalg.eig(CM)

        idx = values.argsort()[::-1]
        vectors = vectors[:, idx]
        Xnew = X.dot(vectors[:, :n])
        return Xnew

    def silhouette(self):
        N = len(self.X)
        s = []
        self.prepareClusters()

        for i in range(self.K):
            sil_clusters = []
            for j in range(len(self.clusters[i])):
                if len(self.clusters[i]) == 0:
                    continue
                else:
                    A_s = [np.linalg.norm(self.clusters[i][j] - self.clusters[i][z]) for z in
                           range(len(self.clusters[i]))]
                    ans_A = sum(A_s) / (len(A_s) - 1)
                    B_s = []

                    for k in range(len(self.clusters)):
                        if k != i:
                            B_s_set = [np.linalg.norm(self.clusters[i][j] - self.clusters[k][f]) for f in
                                       range(len(self.clusters[k]))]

                            if len(B_s_set) == 0:
                                continue
                            else:
                                B_s.append(sum(B_s_set) / len(B_s_set))

                    if len(self.clusters[i]) == 1:
                        sil_clusters.append(0)
                    else:
                        ans_B = min(B_s)
                        sil_clusters.append((ans_B - ans_A) / max([ans_B, ans_A]))
            s.append(sum(sil_clusters))
        ans = sum(s) / N

        return ans


dataset = getCSV()
normalized_dataset = Normalization(dataset)
pdataset = getNaive(normalized_dataset)

k_means = K_means(pdataset, 3, len(pdataset[0]))

X = dataset.drop(['class'], axis=1).values
Y = dataset['class'].values
Xnew = k_means.PCA(np.array([pdataset[i][:-1] for i in range(len(pdataset))]), 2)

plt.scatter(Xnew[:, 0], Xnew[:, 1], c=Y)
plt.title('Распределение класcов')
plt.xlabel('первая компонента')
plt.ylabel('вторая компонента')
plt.show()

A, clusters = k_means.clusterization()
print(k_means.rand_Index())
print(k_means.silhouette())

plt.scatter(Xnew[:, 0], Xnew[:, 1], c=A)
plt.title('Распределение кластеров')
plt.xlabel('первая компонента')
plt.ylabel('вторая компонента')
plt.show()
