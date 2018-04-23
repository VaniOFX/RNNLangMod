from torch.utils.data import Dataset, DataLoader
import hyperparameters as hp
from sklearn.model_selection import train_test_split, ShuffleSplit
import torch

from preprocess import clean_str


word2idx = {'pad': 0}
idx2word = {0: 'pad'}


class MySet(Dataset):

    def __init__(self, *files):
        X = []
        y_pred = []
        data = self.readData(files)
        self.createDictionaries(data)
        data = [word2idx[x] for x in data]

        #create feature vectors
        index = hp.sequence_len
        data_len = len(data)
        while index < data_len:
            if hp.m2m:
                tempX = []
                for i in range(hp.sequence_len, 0, -1):
                    tempX.append(data[index-i])
                X.append(tempX)
                y_pred.append(data[index])
                index += 1

            else:
                for i in range(hp.sequence_len - 1, -1, -1):
                    tempX = []
                    for m in range(index - hp.sequence_len, index-i):
                        tempX.append(data[m])
                    for n in range(index-i, index):
                        tempX.append(word2idx['pad'])
                    X.append(tempX)
                    y_pred.append(data[index - i])
                index += hp.sequence_len

        print(data)
        print(X)
        print(y_pred)
        self.x_data = torch.LongTensor(X)
        self.y_data = torch.LongTensor(y_pred)
        self.len = len(X)

    def readData(self, files):
        data = []
        for file in files:
            f = open(file, 'r')
            text = f.read()
            f.close()
            text = clean_str(text)
            data += text.split()
        return data

    def createDictionaries(self, data):
        for d in set(data):
            idx2word[len(word2idx)] = d
            word2idx[d] = len(word2idx)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


data = MySet('Data/Train/1ARGN10.TXT')

sss = ShuffleSplit(n_splits=1, test_size=0.2)
train_data = []
test_data = []
for train, test in sss.split(data.x_data, data.y_data):
    for i in train:
        train_data.append(data[i])
    for i in test:
        test_data.append(data[i])


train_loader = DataLoader(dataset=train_data, batch_size=hp.bsz, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=hp.bsz)



# X_train, X_dev, Y_train, Y_dev = train_test_split(data.x_data, data.y_data, test_size=0.2, random_state=0)


# data.x_data = torch.LongTensor(X_train)
# data.y_data = torch.LongTensor(Y_train)
# data.len = len(X_train)
# train_loader = DataLoader(dataset=data, batch_size=hp.bsz, shuffle=True)
#
#
# data.x_data = torch.LongTensor(X_dev)
# data.y_data = torch.LongTensor(Y_dev)
# data.len = len(X_dev)
# test_loader = DataLoader(dataset=data, batch_size=hp.bsz)



