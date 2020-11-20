
def F_score(K, CF):

    Sum = 0
    FC = 0

    C = [0] * K
    Precisions = [0.0] * K
    Recalls = [0.0] * K

    for i in range(K):
        Ci = 0
        for j in range(K):
            Ci += CF[i][j]
            Sum += CF[i][j]
        C[i] = Ci

        if Ci == 0:
            Precisions[i] = 0
        else:
            Precisions[i] = CF[i][i] / Ci

    for j in range(K):
        Cj = 0
        for i in range(K):
            Cj += CF[i][j]

        if Cj == 0:
            Recalls[j] = 0
        else:
            Recalls[j] = CF[j][j] / Cj

    for i in range(K):
        if (Precisions[i] == 0) & (Recalls[i] == 0):
            FC += 0
        else:
            FC += 2 * ((Precisions[i] * Recalls[i]) / (Precisions[i] + Recalls[i])) * C[i]

    F_micro = FC / Sum

    Precision_avg = 0
    Recall_avg = 0
    for i in range(K):
        Precision_avg += Precisions[i] * C[i]
        Recall_avg += Recalls[i] * C[i]

    Precision_avg = Precision_avg / Sum
    Recall_avg = Recall_avg / Sum

    if (Precision_avg == 0) & (Recall_avg == 0):
        F_macro = 0
    else:
        F_macro = 2 * ((Precision_avg * Recall_avg) / (Precision_avg + Recall_avg))

    return F_macro



# get dataset from csv
import pandas as pd

def getCSV():
    filename = "Data/cars.csv"

    dataset = pd.read_csv(filename)
    dataset = dataset[
        ["MPG", "cylinders", "cubicInches", "horsepower", "weightLbs", "time-to-sixty", "year", "class"]]
    return dataset

# --------#



import math

# Core_START
import numpy as np


def core(c, u):
    return {
        c == 'uniform': uniform(u),
        c == 'triangular': triangular(u),
        c == 'epanechnikov': epanechnikov(u),
        c == 'quartic': quartic(u),
        c == 'triweight': triweight(u),
        c == 'tricube': tricube(u),
        c == 'gaussian': gaussian(u),
        c == 'logistic': logistic(u),
        c == 'sigmoid': sigmoid(u),
        c == 'cosine': cosine(u),
    }[True]


def uniform(u):
    if u < 1:
        return 0.5
    else:
        return 0


def triangular(u):
    if u < 1:
        return round(1 - math.fabs(u), 8)
    else:
        return 0


def epanechnikov(u):
    if u < 1:
        return round(3 / 4 * (1 - u ** 2), 8)
    else:
        return 0


def quartic(u):
    if u < 1:
        return round(15 / 16 * (1 - u ** 2) ** 2, 8)
    else:
        return 0


def triweight(u):
    if u < 1:
        return round(35 / 32 * (1 - u ** 2) ** 3, 8)
    else:
        return 0


def tricube(u):
    if u < 1:
        return round(70 / 81 * (1 - math.fabs(u) ** 3) ** 3, 8)
    else:
        return 0


def gaussian(u):
    return round(1 / (math.sqrt(2 * math.pi)) * math.e ** ((-1 / 2) * (u ** 2)), 8)


def logistic(u):
    return round(1 / (math.exp(u) + 2 + math.exp(-u)), 8)


def sigmoid(u):
    return round((2 / math.pi) * (1 / (math.exp(u) + math.exp(-u))), 8)


def cosine(u):
    if u < 1:
        return round(math.pi / 4 * math.cos(math.pi / 2 * u), 8)
    else:
        return 0


# Core_END


# Distance_START


def distance(d, x1, x2):
    return {
        d == 'euclidean': euclidean(x1, x2),
        d == 'manhattan': manhattan(x1, x2),
        d == 'chebyshev': chebyshev(x1, x2)
    }[True]


def euclidean(x1, x2):
    d = 0

    for i in range(len(x1)):
        d += (x1[i] - x2[i]) ** 2



    return math.sqrt(d)


def manhattan(x1, x2):
    d = 0
    for i in range(len(x1)):
        d += math.fabs(x1[i] - x2[i])



    return d


def chebyshev(x1, x2):
    d = []
    for i in range(len(x1)):
        d.append(math.fabs(x1[i] - x2[i]))
    d.sort()


    return d[len(d) - 1]


# Distance_END


def kNN_Naive(N, M, matrix, main_x, dis, c, state, score):
    y = [0] * N
    x = []
    matches = []

    for i in range(N):
        x.append([])

    for i in range(N):
        y[i] = matrix[i][M]
        for j in range(M):
            x[i].append(matrix[i][j])

    sum_num = 0
    sum_enum = 0
    if state == 'fixed':

        for i in range(len(x)):

            if distance(dis, x[i], main_x) == 0:
                matches.append(y[i])

            if score == 0:
                u = 0
            else:
                u = distance(dis, x[i], main_x) / score

            sum_num += y[i] * core(c, u)
            sum_enum += core(c, u)
    else:

        u_set = []
        for i in range(len(x)):
            u_set.append(distance(dis, x[i], main_x))

        u_set.sort()

        k = u_set[score]

        for i in range(len(x)):

            if distance(dis, x[i], main_x) == 0:
                matches.append(y[i])

            if k == 0:

                u = 0

            else:
                u = distance(dis, x[i], main_x) / k

            sum_num += y[i] * core(c, u)

            sum_enum += core(c, u)

    if len(matches) == 0:

        if sum_enum == 0:

            for i in range(len(x)):
                sum_enum += y[i]

            return sum_enum / len(x)
        else:
            return sum_num / sum_enum
    else:
        ans = 0
        for i in range(len(matches)):
            ans += matches[i]

        return ans / len(matches)


def kNN_One_Hot(N, M, matrix, main_x, dis, c, state, score):
    y = []
    x = []
    matches = []

    for i in range(N):
        x.append([])
        y.append([])

    for i in range(N):
        y[i].append(matrix[i][M])
        y[i].append(matrix[i][M + 1])
        y[i].append(matrix[i][M + 2])

        for j in range(M):
            x[i].append(matrix[i][j])

    sum_num = [0.0, 0.0, 0.0]
    sum_enum = 0
    if state == 'fixed':

        for i in range(len(x)):

            if distance(dis, x[i], main_x) == 0:
                matches.append(y[i])

            if score == 0:
                u = 0
            else:
                u = distance(dis, x[i], main_x) / score

            sum_num += np.dot(y[i], core(c, u))
            sum_enum += core(c, u)
    else:

        u_set = []
        for i in range(len(x)):
            u_set.append(distance(dis, x[i], main_x))

        u_set.sort()

        k = u_set[score]

        for i in range(len(x)):

            if distance(dis, x[i], main_x) == 0:
                matches.append(y[i])

            if k == 0:

                u = 0

            else:
                u = distance(dis, x[i], main_x) / k

            sum_num += np.dot(y[i], core(c, u))

            sum_enum += core(c, u)

    if len(matches) == 0:

        if sum_enum == 0:

            ans = [0.0, 0.0, 0.0]
            for i in range(len(x)):
                ans += y[i]

            ans = np.dot(ans, 1 / len(x))

            return np.argmax(ans)

        else:
            sum_num = np.dot(sum_num, 1/sum_enum)
            return np.argmax(sum_num)
    else:
        ans = [0.0, 0.0, 0.0]
        for i in range(len(matches)):
            ans += matches[i]
        ans = np.dot(ans, 1 / len(matches))
        return np.argmax(ans)


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


# -------#


def Normalization(dataset):
    mx = minmax(dataset.values)
    normalized_dataset = normalize(dataset.values, mx)

    return normalized_dataset




def getNaive(dataset):
    return dataset.tolist()



def getOne_Hot(dataset):

    dataset_for_one_hot = dataset.tolist()

    for i in range(len(dataset)):
        score = dataset[i][7]

        dataset_for_one_hot[i] = dataset_for_one_hot[i][0:len(dataset[i])-1]

        if score == 0:
            dataset_for_one_hot[i].append(1)
            dataset_for_one_hot[i].append(0)
            dataset_for_one_hot[i].append(0)

        elif score == 1:
            dataset_for_one_hot[i].append(0)
            dataset_for_one_hot[i].append(1)
            dataset_for_one_hot[i].append(0)

        elif score == 2:
            dataset_for_one_hot[i].append(0)
            dataset_for_one_hot[i].append(0)
            dataset_for_one_hot[i].append(1)

    return dataset_for_one_hot
