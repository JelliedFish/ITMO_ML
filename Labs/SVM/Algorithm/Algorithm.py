import math
import pandas as pd

from pandas import np
from sklearn.metrics import accuracy_score


def linear(x, y):
    a = np.dot(x, y)
    return a


def pol(x, y, gamma=1, k0=1, degree=3):
    return np.power(gamma * np.dot(x, y) + k0, degree)


def gaussian(x, y, gamma=0.5):
    a = math.exp(-gamma * (np.linalg.norm(x - y) ** 2))
    return a


def Kernel(kernel, x, y, param):
    return {
        kernel == 'linear-kernel': linear(x, y),
        kernel == 'pol-kernel': pol(x, y, param),
        kernel == 'gaussian-kernel': gaussian(x, y, param)
    }[True]


class CustomSVM(object):

    def __init__(self, epsilon=0.01, iterations=2000):
        self._alphas = []
        self._iterations = iterations
        self._epsilon = epsilon
        self._C = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
        self._k = 10
        self._w = []
        self._B = [1, 2, 3, 4, 5]
        self._D = [2, 3, 4, 5]
        self._kernels = ["linear-kernel", "pol-kernel", "gaussian-kernel"]
        self.best_accuracy = 0
        self.best_kernel = "empty"
        self.best_C = 0.05
        self.best_param = 1

    def KNN(self, x_train, y_train, kernel, param, C):
        accuracy = 0
        for i in range(1, self._k):
            n = int(len(x_train) / self._k)
            X_test = x_train[n * (i - 1):n * i][:]
            Y_test = y_train[n * (i - 1):n * i]

            X_train = x_train[n * i + 1:][:]
            Y_train = y_train[n * i + 1:]

            self.fit(X_train, Y_train, kernel, param, C)
            accuracy += self.predict(X_test, Y_test, X_train, Y_train)

        accuracy = accuracy / (self._k - 1)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_kernel = kernel
            self.best_C = C
            self.best_param = param
        print(accuracy, " ")
        print(kernel, " ")
        print(C, " ")
        print(param, " \n")

    def Algorithm(self, x_train, y_train):

        for c in self._C:
            for k in self._kernels:

                if k == "pol-kernel":
                    for b in self._B:
                        self.KNN(x_train, y_train, "pol-kernel", b, c)

                if k == "gaussian-kernel":
                    for d in self._D:
                        self.KNN(x_train, y_train, "gaussian-kernel", d, c)

                if k == "linear-kernel":
                    self.KNN(x_train, y_train, "linear-kernel", 0, c)

    def diffF(self, diffX_train, diffY_train, diffX, diffY, kernel, param, C, index):

        new_alphas = np.zeros(len(self._alphas))
        sum1 = 0
        sum2 = 0

        sum1 += 1
        for j in range(len(self._alphas)):
            sum2 += self._alphas[index] * diffY * diffY_train[j] * Kernel(kernel, diffX, diffX_train[j], param)

        new_alphas[index] = sum2 + sum1

        return new_alphas

    def F(self, FX_train, FY_train, FX, FY, kernel, param, index):

        new_alphas = np.zeros(len(self._alphas))
        sum1 = 0
        sum2 = 0
        sum1 += -self._alphas[index]

        for j in range(len(self._alphas)):
            sum2 += self._alphas[index] * self._alphas[j] * FY * FY_train[j] * Kernel(kernel, FX, FX_train[j], param)

        sum2 = sum2 * (1 / 2)

        new_alphas[index] = sum1 + sum2

        return new_alphas

    def fit(self, X_train, Y_train, kernel, param, C):  # arrays: X; Y =-1,1
        Lk = 0
        self._alphas = np.zeros(len(Y_train))
        self._alphas.fill(C / 1000)

        print("We are before iteration:")
        for iteration in range(self._iterations):

            batch_size = 10
            for k in range(0, (len(X_train) - 1) // batch_size + 1):  # Calculate the amount of butches in the Dt
                cur_batch_size = min(batch_size,
                                     len(X_train) - 1 - k * batch_size)  # Choose the butch or remaining number
                L_diff = np.zeros(len(self._alphas))
                L_butch = 0

                gradient_rate = 0.0006 / (k + 1)
                alpha = 0.05
                for i in range(cur_batch_size):
                    L_diff += self.diffF(X_train, Y_train, X_train[k * batch_size + i][:], Y_train[k * batch_size + i],
                                         kernel, param, C, k * batch_size + i) * gradient_rate
                    L_butch += self.F(X_train, Y_train, X_train[k * batch_size + i], Y_train[k * batch_size + i],
                                      kernel, param, k * batch_size + i)

                sumai = 0
                for i in range(len(self._alphas) - 1):
                    sumai += Y_train[i] * self._alphas[i]

                self._alphas[len(self._alphas) - 1] = -sumai / Y_train[len(self._alphas) - 1]

                alpha_k = self._alphas

                self._alphas = alpha_k - L_diff / batch_size

                for i in range(len(self._alphas)):
                    if self._alphas[i] > C:
                        self._alphas = alpha_k
                        break

                Lk_previous = Lk
                Lk = (1 - alpha) * Lk + alpha * L_butch

                if (np.linalg.norm(alpha_k - self._alphas < 0.00001)) and (np.linalg.norm(Lk - Lk_previous < 0.00001)):
                    break

        w = np.zeros(len(X_train[0]))

        X_extended = X_train
        for i in range(len(X_extended)):
            w += self._alphas[i] * X_train[i]
        self._w = w

    def predict(self, X, Y, X_train, Y_train):
        y_pred = np.zeros(len(Y))

        for j in range(len(X)):
            pr = 0
            for i in range(len(X_train)):
                pr += self._alphas[i] * np.dot(X[j], X_train[i]) * Y_train[i] - (
                        np.dot(self._w, X_train[i]) - Y_train[i])

            y_pred[j] = np.sign(pr)

        accuracy = accuracy_score(Y, y_pred, normalize=True)
        return accuracy


df = pd.read_csv('../Data/chips.csv')
X = df.values[:, :-1]
Y = df.values[:, -1]
svm = CustomSVM()
svm.Algorithm(X, Y)
print(svm.best_C, svm.best_kernel, svm.best_param, svm.best_accuracy)
