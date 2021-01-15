import os


def get_dataset(n): # n - amount of gramms in the N-gramm
    messages = []
    labels = []

    for folder in range(10):
        files = os.listdir("messages/part{}".format(folder + 1))  # Go through our datasets
        for file in files:
            f = open("messages/part{}/{}".format(folder + 1, file), "r")  # Open each file to read
            message = []
            words = []
            lines = f.readlines()
            for line in lines:
                current_words = line.rstrip().split(' ')  # Get each line
                for word in current_words:
                    if word.find('Subject:') == -1:  # If current word is no Subject
                        if word != '':  # If current word is no empty
                            words.append(word)  # Then it's just word

            if file.find('spmsg') != -1:  # If current file is spam -> add to label like spam
                labels.append('spam')
            elif file.find('legit') != -1:  # If current file is legit -> add to label like legit
                labels.append('legit')

            for i in range(len(words) - n + 1):
                n_gramm = ''

                if n > 1:
                    for j in range(i, i + n - 1):
                        n_gramm += words[j]
                        if j != i + n - 2:
                            n_gramm += ' '
                else:
                    n_gramm = words[i]
                message.append(n_gramm)
                if n > 1:
                    message.append(n_gramm + ' ' + words[i + n - 1])
            messages.append(message)
    return messages, labels
