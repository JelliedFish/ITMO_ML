import numpy as np


def initData():
    data_train = []
    data_test = [[]]
    x_test = [[[]]]
    y_test = [[]]
    x_train = []
    y_train = []

    for i in range(21):

        with open('../Data/' + str(i + 1) + '_train.txt', 'r') as f:
            num_features = int(f.readline().split()[0])
            num_train = int(f.readline().rstrip())
            data_train.append([])

            for j in range(num_train):
                item = [int(xi) for xi in f.readline().split()]
                data_train[i].append(item)

            x_train.append([])
            for j in range(num_train):
                arr = np.zeros(num_features)
                for k in range(num_features):
                    arr[k] = data_train[i][j][k]
                x_train[i].append(arr)

            y_train.append([])
            for j in range(num_train):
                y_train[i].append(data_train[i][j][num_features])

        with open('../Data/' + str(i + 1) + '_test.txt', 'r') as f:
            num_features = int(f.readline().split()[0])
            num_test = int(f.readline().rstrip())
            data_test.append([])

            for j in range(num_test):
                item = [int(xi) for xi in f.readline().split()]
                data_test[i].append(item)

            x_test.append([])
            for j in range(num_test):
                arr1 = np.zeros(num_features)
                for k in range(num_features):
                    arr1[k] = data_test[i][j][k]
                x_test[i].append(arr1)

            y_test.append([])
            for j in range(num_test):
                y_test[i].append(data_test[i][j][num_features])

    return x_train, x_test, y_train, y_test
