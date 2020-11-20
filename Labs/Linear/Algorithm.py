import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt

from Labs.Linear.Normalization import Normalization
from Labs.Linear.etc.ETC import bestT


def draw(ans, ans_x):
    plt.plot(ans_x, ans, label='test')
    plt.legend(loc='upper right')
    plt.title('SMAPE on epochs')
    plt.savefig('smape.png')
    plt.show()


def getWeights(num_features):
    weights = []

    for x in range(num_features):
        weights.append(random.uniform(-1 / (2 * num_features), 1 / (2 * num_features)))

    return weights


def MSE(s, y):
    return (s - y) ** 2


def diffMSE(s, y):
    return 2 * (s - y)


def SMAPE(X, Y, w):
    result = 0
    for i in range(len(X)):
        y_predict = np.dot(X[i], w)
        y_real = Y[i]
        result += abs(y_predict - y_real) / (abs(y_predict) + abs(y_real))
    return result * 200 / len(X)


def main():
    batch_size = 10
    eps = 0.0001
    alpha = 0.05
    x_train = []
    x_test = []
    max_size = 50
    best_tau = bestT()
    best_smape = 201

    def getObjectsFunctionsTest(X_test):
        for i in range(len(X_test)):
            for j in range(num_features + 1):
                if j < num_features:
                    X[i][j] = X_test[i][j]
                else:
                    Y[i] = X_test[i][j]

    # Get the data from out current file:
    ##
    with open('Data/1.txt', 'r') as f:
        num_features = int(f.readline().rstrip())
        num_train = int(f.readline().rstrip())

        for i in range(num_train):
            item = [int(xki) for xki in f.readline().split()]
            x_train.append(item)

        num_test = int(f.readline().rstrip())

        for i in range(num_test):
            item = [int(xki) for xki in f.readline().split()]
            x_test.append(item)

    X_test = np.array(Normalization(x_test))
    X_train = np.array(Normalization(x_train))

    X = np.zeros((len(X_test), num_features))
    Y = np.zeros(len(X_test))
    getObjectsFunctionsTest(X_test)

    ##

    # Set our weights like random (-1/2n;1/2n)
    ##
    weights = getWeights(num_features)
    ##

    smape_test = 0
    iterations = 0
    anss = []
    ans = np.zeros(max_size)
    ans_x = np.zeros(max_size)
    # The Gradient descent
    for t in range(0, 10):  # tau
        tau = 10 ** -t

        for b in range(max_size):# iterations
            if iterations > 2000:
                break

            max_iterations = iterations + 1
            iterations = 0

            while iterations < max_iterations:
                Lk = 0

                for k in range(0, (num_train - 1) // batch_size + 1):  # Calculate the amount of butches in the Dt
                    cur_batch_size = min(batch_size, num_train - k * batch_size)  # Choose the butch or remaining number
                    L_diff = np.zeros(num_features)
                    L_butch = 0

                    gradient_rate = 0.0006 / (k + 1)

                    iterations += 1
                    for i in range(cur_batch_size):
                        s = np.dot(weights, X_train[k * batch_size + i][:num_features])

                        L_diff += diffMSE(s, X_train[k * batch_size + i][-1]) * gradient_rate * X_train[
                                                                                                    k * batch_size + i][
                                                                                                :num_features]
                        L_butch += alpha * MSE(s, X_train[k * batch_size + i][-1]) + tau * np.linalg.norm(weights)

                    wk = np.array(weights)
                    weights = wk * (1 - gradient_rate * tau) - L_diff

                    Lk_previous = Lk
                    Lk = (1 - alpha) * Lk + alpha * L_butch

                    a = Lk - Lk_previous
                    if a < 0.00001:
                        break

                smape_test = SMAPE(X, Y, weights)

                if smape_test < best_smape:
                    best_smape = smape_test

                ans[b] = smape_test
                ans_x[b] = iterations
        anss.append([[ans[i] for i in range(len(ans))], [ans_x[i] for i in range(len(ans_x))]])

    number = int(math.fabs(int(math.log(best_tau, 10))))
    ans = [x for x in filter(lambda i: i != 0, anss[number][0])]
    ans_x = [x for x in filter(lambda i: i != 0, anss[number][1])]
    draw(ans, ans_x)
    print(best_tau)

    best_smape = 201

    # LSM:

    def LSM(X, Y, tau):
        return np.linalg.pinv(np.add(X.T.dot(X), np.cov(X.T).dot(tau))).dot(X.T).dot(Y)

    for t in range(0, 10):
        tau = 10 ** -t

        w = LSM(X, Y, tau)

        smape_test = SMAPE(X, Y, w)
        if smape_test < best_smape:
            best_smape = smape_test
            best_tau = tau
    print(best_smape)
    print(best_tau)


main()
