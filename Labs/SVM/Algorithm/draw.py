from pandas import np
import matplotlib.pyplot as plt


def draw(name, p, clf, X, y, step, ):
    stepx = step
    stepy = step
    x_min, y_min = np.amin(X, 0)
    x_max, y_max = np.amax(X, 0)
    x_min -= stepx
    x_max += stepx
    y_min -= stepy
    y_max += stepy
    xx, yy = np.meshgrid(np.arange(x_min, x_max, stepx),
                         np.arange(y_min, y_max, stepy))

    mesh_dots = np.c_[xx.ravel(), yy.ravel()]
    zz = np.apply_along_axis(lambda t: clf.predict(t), 1, mesh_dots)
    zz = np.array(zz).reshape(xx.shape)

    plt.figure(figsize=(10, 10))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    x0, y0 = X[y == -1].T
    x1, y1 = X[y == 1].T

    plt.pcolormesh(xx, yy, zz, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(x0, y0, color='red', s=100)
    plt.scatter(x1, y1, color='blue', s=100)

    sup_ind = clf.get_non_bound_indices()
    X_sup = X[sup_ind]
    x_sup, y_sup = X_sup.T

    plt.scatter(x_sup, y_sup, color='white', marker='x', s=60)
    plt.suptitle(p)
    plt.savefig(name + '_' + p['name'] + '.png')
    plt.show()