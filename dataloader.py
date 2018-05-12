from torch.utils.data import Dataset, DataLoader
import torch
import hyperparameters as hp
from sklearn.model_selection import ShuffleSplit
from preprocess import clean_str


word2idx = {'pad': 0}
idx2word = {0: 'pad'}


class MySet(Dataset):

    def __init__(self, files):
        X = []
        y_pred = []
        data = self.readData(files)
        self.createDictionaries(data)
        data = [word2idx[x] for x in data]

        #create feature vectors
        data_len = len(data)
        for i in range(data_len - hp.sequence_len):
            X.append(data[i:i + hp.sequence_len])
            if hp.m2m:
                y_pred.append((data[i+1:i+hp.sequence_len+1]))
            else:
                y_pred.append(data[i+hp.sequence_len])

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
            f.close()
        return data

    def createDictionaries(self, data):
        for d in set(data):
            idx2word[len(word2idx)] = d
            word2idx[d] = len(word2idx)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


data = MySet(hp.files)

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