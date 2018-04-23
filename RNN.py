import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import hyperparameters as hp
from dataloader import word2idx

vocab_size = len(word2idx)
print(vocab_size, "is the vocabulary size")

class RNN_M2M(torch.nn.Module):

    def __init__(self, v_size):
        super(RNN_M2M, self).__init__()
        self.embedding = nn.Embedding(v_size, hp.embedding_dim)
        self.rnn = nn.LSTM(input_size=hp.input_size, hidden_size=hp.hidden_size, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hp.hidden_size, v_size)

    def forward(self, inputs):
        pass



class RNN_M2O(torch.nn.Module):

    def __init__(self, v_size):
        super(RNN_M2O, self).__init__()
        self.embedding = nn.Embedding(v_size, hp.embedding_dim)
        self.rnn = nn.LSTM(input_size=hp.input_size, hidden_size=hp.hidden_size, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hp.hidden_size, v_size)

    def forward(self, inputs):
        emb = self.embedding(inputs)
        out, h_n = self.rnn(emb)
        print(out.size(), "out")
        print(h_n[0].size(), "h_n")
        out = self.fc(h_n[0][0])
        return out



if hp.m2m:
    model = RNN_M2M(vocab_size).cuda()
else:
    model = RNN_M2O(vocab_size).cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate)