import math


def X():
    K1, K2 = [int(var) for var in input().split()]
    N = int(input())

    P = {}

    for i in range(N):
        set = [int(var) for var in input().split()]

        if set[0] in P:

            if set[1] in P[set[0]]:
                P[set[0]][set[1]] += 1
            else:
                P[set[0]][set[1]] = 1
        else:
            P[set[0]] = {set[1]: 1}

    ans = 0
    for i in P:
        s = 0

        for k in P[i]:
            s += P[i][k]

        e = 0
        for j in P[i]:
            x = P[i][j] / s
            e += -1 * x * math.log(x)

        ans += (s / N) * e
    print(ans)


X()
