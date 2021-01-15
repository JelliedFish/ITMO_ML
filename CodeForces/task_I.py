def Spirman():
    N = int(input())
    matrix = []
    matrix_var = []

    for i in range(N):
        matrix_var.append([0, 0])

    for i in range(N):
        set = [int(var) for var in input().split()]
        matrix.append(set)

    cur1 = []
    cur2 = []

    for i in range(N):
        cur1.append(matrix[i][0])
        cur2.append(matrix[i][1])

    cur1.sort()
    cur2.sort()

    for i in range(N):
        matrix_var[i][0] = [matrix[i][0], binSearch(cur1, matrix[i][0])]
        matrix_var[i][1] = [matrix[i][1], binSearch(cur2, matrix[i][1])]

    matrix_var.append([[], []])
    hasBundles, matrix_var = findBundles(matrix_var, N)

    # hasBundles = False

    sum = 0

    if hasBundles == False:
        ans = 1
        for i in range(N):
            sum += (matrix_var[i][0][1] - matrix_var[i][1][1]) ** 2

        sum *= 6
        sum /= (N * (N - 1) * (N + 1))

        ans = ans - sum

        return ans

    else:
        for i in range(N):
            sum += (matrix_var[i][0][1] - (N + 1) / 2) * (matrix_var[i][1][1] - (N + 1) / 2)

        delta1 = 0
        delta2 = 0
        for i in range(matrix_var[N][0][len(matrix_var[N][0]) - 1]):
            delta1 += matrix_var[N][0][i] * (matrix_var[N][0][i] ** 2 - 1)
        delta1 *= 1 / 2

        for i in range(matrix_var[N][1][len(matrix_var[N][1]) - 1]):
            delta2 += matrix_var[N][1][i] * (matrix_var[N][1][i] ** 2 - 1)
        delta2 *= 1 / 2

        delta = delta1 + delta2

        d = (N * (N - 1) * (N + 1) - delta)
        ans = sum / d

        return ans


def findBundles(matrix, N):
    i = 0
    hasBundle = False

    while i < N - 1:
        k = i
        while matrix[i][0][0] == matrix[i + 1][0][0]:
            i += 1
            hasBundle = True

        for j in range(k, i + 1):
            if i - k >= 1:
                matrix[j][0][1] = ((i + k) / 2)

        if i - k >= 1:
            matrix[N][0].append(i - k + 1)
        i += 1
    if len(matrix[N][0]) > 0:
        matrix[N][0].append(len(matrix[N][0]))
    else:
        matrix[N][0].append(0)
        matrix[N][0].append(0)

    i = 0
    while i < N - 1:
        k = i
        while matrix[i][1][0] == matrix[i + 1][1][0]:
            i += 1
            hasBundle = True

        for j in range(k, i + 1):
            if i - k >= 1:
                matrix[j][1][1] = ((i + k) / 2)

        if i - k >= 1:
            matrix[N][1].append(i - k + 1)

        i += 1

    if len(matrix[N][1]) > 0:
        matrix[N][1].append(len(matrix[N][1]))
    else:
        matrix[N][1].append(0)
        matrix[N][1].append(0)

    return hasBundle, matrix


def binSearch(matrix, value):
    mid = len(matrix) // 2
    low = 0
    high = len(matrix) - 1

    while matrix[mid] != value and low <= high:
        if value > matrix[mid]:
            low = mid + 1
        else:
            high = mid - 1
        mid = (low + high) // 2

    return mid


print(Spirman())
