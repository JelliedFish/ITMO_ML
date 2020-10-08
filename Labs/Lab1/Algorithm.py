# the algorithm for finding of hyper parameters
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from Labs.Lab1.F_score import F_score
from Labs.Lab1.NonparametricRegression import kNN_Naive, kNN_One_Hot
from Labs.Lab1.Normalization import Normalization
from Labs.Lab1.PrepareDataset import getNaive, getOne_Hot
from Labs.Lab1.getCSV import getCSV


def hyper(dataset, dataset_for_naive, dataset_for_one_hot, N, M):
    max_F_naive = 0
    max_F_one_hot = 0

    max_h = 50

    hyper_naive = []
    hyper_one_hot = []

    distances_t = ["manhattan", "euclidean", "chebyshev"]
    cores_t = ["uniform", "triangular", "epanechnikov", "quartic", "triweight", "tricube", "gaussian", "cosine",
               "logistic", "sigmoid"]
    state_t = ["fixed", "variable"]

    for distance in distances_t:
        print(distance)

        for core in cores_t:
            print(core)
            for state in state_t:

                for hi in range(1, max_h, 1):

                    CM_naive = np.zeros((N, N), dtype=int)
                    CM_one_hot = np.zeros((N, N), dtype=int)

                    for i in range(len(dataset)):

                        true_naive = dataset_for_naive[i][7]

                        true_one_hot = 0
                        for j in range(7, 10):
                            if dataset_for_one_hot[i][j] != 0:
                                true_one_hot = j - 7
                                break

                        predict_naive = kNN_Naive(N - 1, M, dataset_for_naive[0:i] + dataset_for_naive[i + 1:]
                                                  , dataset_for_naive[i][:M], distance, core, state, hi)

                        predict_one_hot = kNN_One_Hot(N - 1, M, dataset_for_one_hot[0:i] + dataset_for_one_hot[i + 1:]
                                                      , dataset_for_one_hot[i][:M], distance, core, state, hi)

                        CM_naive[int(round(true_naive))-1][int(round(predict_naive))-1] += 1
                        CM_one_hot[int(round(true_one_hot))][int(round(predict_one_hot))] += 1

                    F_score_naive = F_score(N, CM_naive)
                    F_score_one_hot = F_score(N, CM_one_hot)

                    if F_score_naive > max_F_naive:
                        hyper_naive = [distance, core, state, hi]
                        max_F_naive = F_score_naive

                    if F_score_one_hot > max_F_one_hot:
                        hyper_one_hot = [distance, core, state, hi]
                        max_F_one_hot = F_score_one_hot

    print(max_F_naive, hyper_naive)
    print(max_F_one_hot, hyper_one_hot)

    return [hyper_naive, hyper_one_hot]


# --------#


def draw(dataset, dataset_for_naive, dataset_for_one_hot, N, M, max_h, hyper_naive, hyper_one_hot):
    f_naive_F = []
    f_naive_h = []
    f_one_hot_F = []
    f_one_hot_h = []

    for hi in range(1, max_h, 1):

        hi_naive = hi
        hi_one_hot = hi

        CM_naive = np.zeros((N, N), dtype=int)
        CM_one_hot = np.zeros((N, N), dtype=int)

        for i in range(len(dataset)):

            true_naive = dataset_for_naive[i][7]

            true_one_hot = 0
            for j in range(7, 10):
                if dataset_for_one_hot[i][j] != 0:
                    true_one_hot = j - 7
                    break

            test_naive_N_1 = dataset_for_naive[0:i] + dataset_for_naive[i + 1:]
            result_naive = dataset_for_naive[i][:M]

            predict_naive = kNN_Naive(N - 1, M, test_naive_N_1
                                      , result_naive, hyper_naive[0], hyper_naive[1], hyper_naive[2], hi_naive)

            test_one_hot_N_1 = dataset_for_one_hot[0:i] + dataset_for_one_hot[i + 1:]
            result_one_hot = dataset_for_one_hot[i][:M]
            predict_one_hot = kNN_One_Hot(N - 1, M, test_one_hot_N_1
                                          , result_one_hot, hyper_one_hot[0], hyper_one_hot[1],
                                          hyper_one_hot[2], hi_one_hot)

            CM_naive[int(round(true_naive))-1][int(round(predict_naive))-1] += 1
            CM_one_hot[int(round(true_one_hot))][int(round(predict_one_hot))] += 1

        F_score_naive = F_score(N, CM_naive)
        F_score_one_hot = F_score(N, CM_one_hot)

        f_naive_F.append(F_score_naive)

        if hi_naive < 1:
            f_naive_h.append(hi_naive * 10)
        else:
            f_naive_h.append(hi_naive)

        f_one_hot_F.append(F_score_one_hot)

        if hi_one_hot < 1:
            f_one_hot_h.append(hi_one_hot * 10)
        else:
            f_one_hot_h.append(hi_one_hot)

    plt.plot(np.array(f_naive_h), np.array(f_naive_F), color="b")
    plt.plot(np.array(f_one_hot_h), np.array(f_one_hot_F), color="r")
    plt.show()


# get data from CSV-file
dataset = getCSV()

# normalize it

normalized_dataset = Normalization(dataset)

# The dataset for Naive (classes {1,2,3}):
dataset_for_naive = getNaive(normalized_dataset)

# The dataset for One-hot:
dataset_for_one_hot = getOne_Hot(normalized_dataset)

# Start of the algorithm:
hypers = hyper(normalized_dataset, dataset_for_naive, dataset_for_one_hot, 210, 7)
draw(dataset, dataset_for_naive, dataset_for_one_hot, 210, 7, 50, hypers[0], hypers[1])
# --------#
