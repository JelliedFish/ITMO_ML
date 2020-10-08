import math


# Core_START

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
        return 1 - math.fabs(u)
    else:
        return 0


def epanechnikov(u):
    if u < 1:
        return 3 / 4 * (1 - u ** 2)
    else:
        return 0


def quartic(u):
    if u < 1:
        return 15 / 16 * (1 - u ** 2) ** 2
    else:
        return 0


def triweight(u):
    if u < 1:
        return 35 / 32 * (1 - u ** 2) ** 3
    else:
        return 0


def tricube(u):
    if u < 1:
        return 70 / 81 * (1 - math.fabs(u) ** 3) ** 3
    else:
        return 0


def gaussian(u):
    return 1 / (math.sqrt(2 * math.pi)) * math.e ** ((-1 / 2) * (u ** 2))


def logistic(u):
    return 1 / (math.exp(u) + 2 + math.exp(-u))


def sigmoid(u):
    return (2 / math.pi) * (1 / (math.exp(u) + math.exp(-u)))


def cosine(u):
    if u < 1:
        return math.pi / 4 * math.cos(math.pi / 2 * u)
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


def main():
    N, M = map(int, input().split())

    matrix = []
    y = [0] * N
    x = []
    main_x = [0] * 2
    matches = []

    for i in range(N):
        x.append([])

    for i in range(N):
        set = [int(var) for var in input().split()]
        matrix.append(set)

    main_x = [int(var) for var in input().split()]

    dis = str(input())

    c = str(input())

    state = str(input())

    score = int(input())

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

            print(sum_enum/len(x))
        else:
            print(sum_num / sum_enum)
    else:
        ans = 0
        for i in range(len(matches)):
            ans += matches[i]

        print(ans / len(matches))


    return 0


main()
