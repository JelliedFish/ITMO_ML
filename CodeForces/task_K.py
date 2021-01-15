def D():
    K = int(input())
    N = int(input())

    P = {}

    for i in range(N):
        set = [int(var) for var in input().split()]

        if set[0] in P:
            P[set[0]].append(set[1])
        else:
            P[set[0]] = [set[1]]

    md = 0
    for i in P:
        if len(P[i]) != 0:
            m = sum(P[i]) / len(P[i])

            for j in range(len(P[i])):
                md += (m - P[i][j]) ** 2
    ans = md / N
    print(ans)


D()
