def main():
    N, M, K = map(int, input().split())

    set = [int(var) for var in input().split()]
    classes = []
    groups = []

    for i in range(K):
        groups.append([])

    for i in range(M):
        classes.append([])

    for i in range(N):
        classes[set[i] - 1].append(i + 1)

    cur = 0
    for i in range(M):

        for k in range(len(classes[i])):
            groups[cur % K].append(classes[i][k])
            cur += 1

    for i in groups:
        print(len(i), end=" ")

        for j in i:
            print(j, end=" ")
        print()

    return 0


main()
