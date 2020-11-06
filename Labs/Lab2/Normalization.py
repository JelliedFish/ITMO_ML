# functions for normalization

def minmax(dataset):
    minmax = list()
    size = len(dataset[0])

    for i in range(size):

        arr = []

        for j in range(len(dataset)):
            arr.append(dataset[j][i])

        value_min = min(arr)
        value_max = max(arr)

        if value_max == value_min:
            value_min = 0

        minmax.append([value_min, value_max])
    return minmax


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    return dataset


# -------#


def Normalization(dataset):
    mx = minmax(dataset)
    normalized_dataset = normalize(dataset, mx)

    return normalized_dataset



