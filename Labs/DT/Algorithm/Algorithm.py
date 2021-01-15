from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from Labs.DT.Algorithm.readData import initData
from Labs.DT.Algorithm.all import check


def draw(i_min, i_max, c_mini, s_mini, c_maxi, s_maxi):
    c_mini = "gini" if c_mini == 0 else "entropy"
    c_maxi = "gini" if c_maxi == 0 else "entropy"
    s_maxi = "best" if s_maxi == 0 else "random"
    s_mini = "best" if s_mini == 0 else "random"
    i_min = 2
    i_max = 20

    arr_accur_train = []
    arr_depth_train = []
    arr_accur_test = []
    arr_depth_test = []
    for i in range(1, 21):
        d_train = DT(c_mini, s_mini, i, x_train[i_min], y_train[i_min], x_test[i_min], y_test[i_min])
        arr_accur_train.append(d_train.algorithm())
        arr_depth_train.append(i)

        d_test = DT(c_mini, s_mini, i, x_test[i_min], y_test[i_min], x_test[i_min], y_test[i_min])
        arr_accur_test.append(d_test.algorithm())
        arr_depth_test.append(i)

    arr_accur_test = check(arr_accur_test)

    plt.plot(arr_depth_train, arr_accur_train)
    plt.plot(arr_depth_test,arr_accur_test, color="red")
    plt.title("Dataset-" + str(i_min))
    plt.show()

    arr_accur_train = []
    arr_depth_train = []
    arr_accur_test = []
    arr_depth_test = []
    for i in range(2, 21):
        d = DT(c_maxi, s_maxi, i, x_train[i_max], y_train[i_max], x_test[i_max], y_test[i_max])
        arr_accur_train.append(d.algorithm())
        arr_depth_train.append(i)

        d_test = DT(c_maxi, s_maxi, i, x_test[i_max], y_test[i_max], x_test[i_max], y_test[i_max])
        arr_accur_test.append(d_test.algorithm())
        arr_depth_test.append(i)




    plt.plot(arr_depth_train, arr_accur_train)
    plt.plot(arr_depth_test,arr_accur_test, color="red")
    plt.title("Dataset-" + str(i_max))
    plt.show()


class DT:

    def __init__(self, criterion, splitter, max_depth, X_train, Y_train, X_test, Y_test):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.Y_pred = []

    def algorithm(self):
        clf_tree = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth, splitter=self.splitter)
        clf_tree.fit(self.X_train, self.Y_train)
        self.Y_pred = clf_tree.predict(X=self.X_test)
        return accuracy_score(self.Y_pred, self.Y_test)


x_train, x_test, y_train, y_test = initData()
# d_test = DT("gini", "random", 5, x_train[1], y_train[1], x_test[1], y_test[1])
# print(d_test.algorithm())
criterion = {"gini", "entropy"}
splitter = {"best", "random"}
maxi_ans = 0
mini_ans = 0
ac = []
ac_best = []
for i in range(1, 21):
    for c in criterion:
        for s in splitter:
            for md in range(1, 21):
                d = DT(c, s, md, x_train[i], y_train[i], x_test[i], y_test[i])
                ac_ = [d.algorithm(),
                       0 if c == "gini" else 1,
                       0 if s == "best" else 1,
                       md,
                       i]
                ac.append(ac_)

    maxi = 0
    for j in range(len(ac)):
        if ac[j][0] > ac[maxi][0]:
            maxi = j

    print(ac[maxi][0], "gini" if ac[maxi][1] == 0
    else "entropy", "best" if ac[maxi][2] == 0
    else "random", ac[maxi][3])
    ac_best.append(ac[maxi])
    ac = []

for j in range(len(ac_best)):

    if ac_best[j][3] > ac_best[maxi_ans][3]:
        maxi_ans = j

    if ac_best[j][3] < ac_best[mini_ans][3]:
        mini_ans = j

print("The biggest max_depth:")
print(ac_best[maxi_ans][0], "gini" if ac_best[maxi_ans][1] == 0
else "entropy", "best" if ac_best[maxi_ans][2] == 0
      else "random", ac_best[maxi_ans][3], ac_best[maxi_ans][4] )

print("The smallest max_depth:")
print(ac_best[mini_ans][0], "gini" if ac_best[mini_ans][1] == 0
else "entropy", "best" if ac_best[mini_ans][2] == 0
      else "random", ac_best[mini_ans][3], ac_best[mini_ans][4])

draw(ac_best[mini_ans][4], ac_best[maxi_ans][4], ac_best[mini_ans][1], ac_best[mini_ans][2], ac_best[maxi_ans][1], ac_best[maxi_ans][2])


