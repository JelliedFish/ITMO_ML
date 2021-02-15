import numpy as np
def Distances():
    K = int(input())
    N = int(input())

    d_outs = 0
    d_outs_i = 0
    ds = np.zeros(K)
    ds_i = {a: [] for a in range(K)}

    matrix = []

    for i in range(N):
        set = [int(var) for var in input().split()]
        matrix.append(set)
        d_outs += matrix[i][0]

    matrix = sorted(matrix, key=lambda a: a[0])

    d = {a: [] for a in range(K)}
    for i in range(N):
        d[matrix[i][1] - 1].append(matrix[i][0])
        ds[matrix[i][1] - 1] += matrix[i][0]

        if len(ds_i[matrix[i][1] - 1]) > 0:
            ds_i[matrix[i][1] - 1].append(ds_i[matrix[i][1] - 1][-1]+matrix[i][0])
        else:
            ds_i[matrix[i][1] - 1].append(matrix[i][0])

    d_in = 0
    d_out = 0


    for i in range(K):
        for j in range(len(d[i])):
            d_in += d[i][j] * (2 * (j+1) - len(d[i])) + ds[i] - 2 * ds_i[i][j]

    for i in range(N):
        d_outs_i += matrix[i][0]
        d_out += matrix[i][0] * (2 * (i+1) - N) + d_outs - 2 * d_outs_i

    return d_in, d_out - d_in

d_ans_in, d_ans_out = Distances()

print(int(d_ans_in))
print(int(d_ans_out))
