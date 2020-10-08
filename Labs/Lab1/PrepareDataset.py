
def getNaive(dataset):
    return dataset.tolist()



def getOne_Hot(dataset):

    dataset_for_one_hot = dataset.tolist()

    for i in range(len(dataset)):
        score = dataset[i][7]

        dataset_for_one_hot[i] = dataset_for_one_hot[i][0:len(dataset[i])-1]

        if score == 1:
            dataset_for_one_hot[i].append(1)
            dataset_for_one_hot[i].append(0)
            dataset_for_one_hot[i].append(0)

        elif score == 2:
            dataset_for_one_hot[i].append(0)
            dataset_for_one_hot[i].append(1)
            dataset_for_one_hot[i].append(0)

        elif score == 3:
            dataset_for_one_hot[i].append(0)
            dataset_for_one_hot[i].append(0)
            dataset_for_one_hot[i].append(1)

    return dataset_for_one_hot
