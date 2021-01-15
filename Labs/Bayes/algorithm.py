import collections
import math
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
import numpy as np

from Labs.Bayes.read_data import get_dataset

from Labs.Bayes.draw import drawRoc


class Bayes:

    def __init__(self, alpha, lambdas):
        self.alpha = alpha
        self.lambdas = lambdas
        self.set = {}
        self.D = 0
        self.Dc = {"legit": 0, "spam": 0}
        self.Wic = {}
        self.Lc = {}
        self.V = 0
        self.Q = []

    def fit(self, train):
        messages = train[0]
        labels = train[1]

        self.D = len(messages)
        cWic = set()

        for i in range(len(messages)):
            # Go through all messages
            if labels[i] in self.set:
                self.set[labels[i]] += messages[i]
            else:  # Add the messages by labels in one list
                self.set[labels[i]] = messages[i]

            self.Dc[labels[i]] += 1  # Calculate the amount of messages for each label

            cWic = cWic.union(set(messages[i]))

        cWic = collections.Counter(cWic)
        self.Wic = cWic

        self.V = cWic["spam"] + cWic["legit"]

        self.Q = messages

        for k, v in self.set.items():
            self.set[k] = collections.Counter(v)

    def predict(self, test):
        messages = test[0]
        predicted = []
        predicted_roc = []

        for i in range(len(messages)):
            ans = []

            for state in {"spam", "legit"}:
                s = 0.0

                for n_gramm in messages[i]:
                    s += math.log(self.set[state][n_gramm] + self.alpha) - math.log(
                        self.Dc[state] + self.alpha)
                ans.append(math.log(self.lambdas[state] * (self.Dc[state] / sum(self.Dc.values()))) + s)

            ans_state = "spam" if ans[0] > ans[1] else "legit"
            predicted.append(ans_state)

            ans_n = np.zeros(len(ans))
            for j in range(len(ans)):
                ans_n[j] = ans[j] / sum(ans)
            predicted_roc.append((1 - ans_n[1]))

        return predicted, predicted_roc


def split_dataset(messages, labels, i, N):
    messages_train = messages[:N * (i - 1)] + messages[N * i:]
    messages_test = messages[N * (i - 1):N * i]
    labels_train = labels[:N * (i - 1)] + labels[N * i:]
    labels_test = labels[N * (i - 1):N * i]
    return messages_train, labels_train, messages_test, labels_test


def get_fp(labels_predict, labels_true):
    count = 0
    for i in range(len(labels_predict)):
        if labels_predict[i] == 'spam' and labels_true[i] == 'legit':
            count += 1
    return count


def algorithm():
    messages, labels = get_dataset(1)
    accuracies = []
    for i in range(1, 11):
        messages_train, labels_train, messages_test, labels_test = split_dataset(messages, labels, i, 109)
        classifier = Bayes(0.01, {'legit': 1, 'spam': 1})
        classifier.fit((messages_train, labels_train))
        labels_predicted, labels_roc = classifier.predict((messages_test, labels_test))
        accuracies.append(accuracy_score(labels_test, labels_predicted))
    #    drawRoc(labels_roc, labels_test)
    print(f'accuracy: {sum(accuracies) / len(accuracies)}')



def drawL():
    messages, labels = get_dataset(1)
    fp = -1
    accuracies = []
    powers = []
    power = 0
    messages_train, labels_train, messages_test, labels_test = split_dataset(messages, labels, 2,109)
    while fp != 0:
        classifier = Bayes(0.01, {'legit': 10 ** power, 'spam': 1})
        classifier.fit((messages_train, labels_train))
        labels_predict, labels_roc = classifier.predict((messages_test, labels_test))
        accuracies.append(accuracy_score(labels_predict, labels_test))
        powers.append(power)
        fp = get_fp(labels_predict, labels_test)
        power += 5

    plt.plot(powers, accuracies)
    plt.xlabel('lambda_legit)')
    plt.ylabel('Accuracy')
    plt.title('График зависимости точности от параметра log(lambda_legit)')
    plt.show()

algorithm()
drawL()

