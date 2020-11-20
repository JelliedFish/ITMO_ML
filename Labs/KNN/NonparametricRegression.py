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


