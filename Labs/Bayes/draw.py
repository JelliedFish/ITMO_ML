import numpy as np
import matplotlib.pyplot as plt


def drawRoc(labels_roc, labels_test):
    label_prob = list(zip(labels_roc, labels_test))
    label_prob.sort()

    labels = [object[1] for object in label_prob]
    legit_count = labels.count('legit')
    spam_count = labels.count('spam')


    xs = [0]
    ys = [0]
    for l in labels:
        if l == 'legit':
            ys.append(ys[-1])
            xs.append(1 / legit_count + xs[-1])
        else:
            ys.append(1 / spam_count + ys[-1])
            xs.append(xs[-1])
    x_const = np.linspace(0, 1, 30)
    y_const = np.linspace(0, 1, 30)

    plt.plot(xs, ys, label='Обученный классификатор')
    plt.plot(x_const, y_const, linestyle='--', label='Константный классификатор')
    plt.title('ROC-кривая')
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

