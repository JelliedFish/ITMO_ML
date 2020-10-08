def main():
    Sum = 0
    FC = 0

    N = int(input())
    matrix = []
    C = [0] * N
    Precisions = [0.0] * N
    Recalls = [0.0] * N

    for i in range(N):
        set = [int(var) for var in input().split()]
        matrix.append(set)

    for i in range(N):
        Ci = 0
        for j in range(N):
            Ci += matrix[i][j]
            Sum += matrix[i][j]
        C[i] = Ci

        if Ci == 0:
            Precisions[i] = 0
        else:
            Precisions[i] = matrix[i][i] / Ci

    for j in range(N):
        Cj = 0
        for i in range(N):
            Cj += matrix[i][j]

        if Cj == 0:
            Recalls[j] = 0
        else:
            Recalls[j] = matrix[j][j] / Cj

    for i in range(N):
        if (Precisions[i] == 0) & (Recalls[i] == 0):
            FC += 0
        else:
            FC += 2 * ((Precisions[i] * Recalls[i]) / (Precisions[i] + Recalls[i])) * C[i]

    F_micro = FC / Sum

    Precision_avg = 0
    Recall_avg = 0
    for i in range(N):
        Precision_avg += Precisions[i] * C[i]
        Recall_avg += Recalls[i] * C[i]

    Precision_avg = Precision_avg / Sum
    Recall_avg = Recall_avg / Sum

    if (Precision_avg == 0) & (Recall_avg == 0):
        F_macro = 0
    else:
        F_macro = 2 * ((Precision_avg * Recall_avg) / (Precision_avg + Recall_avg))

    print(F_macro)
    print(F_micro)

    return 0


main()
