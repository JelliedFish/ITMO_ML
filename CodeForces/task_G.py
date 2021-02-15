def G():
    N = int(input())

    ones_amount = 0
    matrix = []
    hasOne = False
    for i in range(2 ** N):
        set = int(input())
        matrix.append(set)

    ones = []
    for i in range(len(matrix)):
        if N <= 9:
            if matrix[i] == 1:
                ones.append(i)
                hasOne = True
                ones_amount += 1
        else:
            if matrix[i] == 0:
                ones.append(i)
                ones_amount += 1

    for i in range(len(ones)):
        ones[i] = bin(ones[i], N)

        new_ones = ""
        for j in reversed(ones[i]):
            new_ones += j

        ones[i] = new_ones
    first_layer_result = []
    for i in range(len(ones)):
        first_layer = ""
        sum = 0
        for j in ones[i]:
            if j == "0":
                first_layer += "-1.0 "
            else:
                first_layer += "1.0 "
                sum += 1

        sum = -1 * (sum - 0.5)
        first_layer_result.append(first_layer + str(sum))

    second_layer_result = []
    second_layer = ""
    for i in range(ones_amount):
        if N <= 9:
            second_layer += "1 "
        else:
            second_layer += "-1 "
    if N <= 9:
        second_layer_result.append(second_layer + "-0.5")
    else:
        second_layer_result.append(second_layer + "0.5")

    if N > 9 and ones_amount == 0:
        print(1)
        print(1)
        print("-0.5")

    if hasOne:
        print(2)
        print(len(first_layer_result), len(second_layer_result))

        for i in range(len(first_layer_result)):
            print(first_layer_result[i])

        for i in range(len(second_layer_result)):
            print(second_layer_result[i])
    else:
        print(1)
        print(1)
        first_layer = ""
        for i in range(N):
            first_layer += "0.0 "
        first_layer += "-0.5"
        print(first_layer)
#

def bin(n, l):
    b = ""
    while n > 0:
        b = str(n % 2) + b
        n = n // 2

    ans = ""

    cnt = len(b)
    while cnt < l:
        ans += "0"
        cnt += 1

    ans += b

    return ans


G()
