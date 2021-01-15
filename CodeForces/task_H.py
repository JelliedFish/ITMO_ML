import math


def Pirson():
    N = int(input())
    matrix = []
    avg_x = 0
    avg_y = 0

    for i in range(N):
        set = [int(var) for var in input().split()]
        matrix.append(set)

    avg_x, avg_y = avg(N ,matrix)

    E = 0

    for i in range(N):
        E += (matrix[i][0] - avg_x)*(matrix[i][1]-avg_y)

    d1 = 0
    d2 = 0
    for i in range(N):
        d1 += (matrix[i][0] - avg_x)**2
        d2 += (matrix[i][1] - avg_y)**2

    D = math.sqrt(d1*d2)

    if D == 0:
        return 0
    else:
        return E/D

def avg(N, matrix):

    avg_x = 0
    avg_y = 0
    for i in range(N):
        avg_x += matrix[i][0]
    avg_x /= N

    for i in range(N):
        avg_y += matrix[i][1]
    avg_y /= N

    return avg_x ,avg_y

print(Pirson())