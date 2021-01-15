import numpy as np


def X():
    K1, K2 = [int(var) for var in input().split()]
    N = int(input())

    P = {}

    P_r_sum = np.zeros(K1)
    P_c_sum = np.zeros(K2)
    for i in range(N):
        set = [int(var) for var in input().split()]

        if set[0] - 1 in P:

            if set[1] - 1 in P[set[0] - 1]:
                P[set[0] - 1][set[1] - 1] += 1
            else:
                P[set[0] - 1][set[1] - 1] = 1
        else:
            P[set[0] - 1] = {set[1] - 1: 1}

        P_r_sum[set[0] - 1] += 1
        P_c_sum[set[1] - 1] += 1

    xi = N
    for i in P:
        for j in P[i]:
            xi += (P[i][j] ** 2 * N) / (P_r_sum[i] * P_c_sum[j]) - 2 * P[i][j]


    print(xi)


X()
