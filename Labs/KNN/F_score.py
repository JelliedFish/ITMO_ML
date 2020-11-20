
def F_score(K, CF):

    Sum = 0
    FC = 0

    C = [0] * K
    Precisions = [0.0] * K
    Recalls = [0.0] * K

    for i in range(K):
        Ci = 0
        for j in range(K):
            Ci += CF[i][j]
            Sum += CF[i][j]
        C[i] = Ci

        if Ci == 0:
            Precisions[i] = 0
        else:
            Precisions[i] = CF[i][i] / Ci

    for j in range(K):
        Cj = 0
        for i in range(K):
            Cj += CF[i][j]

        if Cj == 0:
            Recalls[j] = 0
        else:
            Recalls[j] = CF[j][j] / Cj

    for i in range(K):
        if (Precisions[i] == 0) & (Recalls[i] == 0):
            FC += 0
        else:
            FC += 2 * ((Precisions[i] * Recalls[i]) / (Precisions[i] + Recalls[i])) * C[i]

    F_micro = FC / Sum

    Precision_avg = 0
    Recall_avg = 0
    for i in range(K):
        Precision_avg += Precisions[i] * C[i]
        Recall_avg += Recalls[i] * C[i]

    Precision_avg = Precision_avg / Sum
    Recall_avg = Recall_avg / Sum

    if (Precision_avg == 0) & (Recall_avg == 0):
        F_macro = 0
    else:
        F_macro = 2 * ((Precision_avg * Recall_avg) / (Precision_avg + Recall_avg))

    return F_macro
