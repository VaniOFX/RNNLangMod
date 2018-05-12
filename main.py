from dataloader import train_loader, test_loader
from RNN import model, optimizer, loss_function
from torch.autograd import Variable
import hyperparameters as hp
import math
import matplotlib.pyplot as plt


def train():
    model.train()
    total_loss = 0
    data_length = len(train_loader.dataset)
    for idx, (data, label) in enumerate(train_loader):
        data, label = Variable(data).cuda(), Variable(label).cuda()
        outputs = model(data)
        optimizer.zero_grad()
        if hp.m2m:
            loss = loss_function(outputs.view(-1, outputs.size(-1)), label.view(-1))
        else:
            loss = loss_function(outputs, label)
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        if idx % 40 == 0:
            print("The loss is", loss.data[0] / hp.bsz)

    return total_loss / data_length

def test():
    model.eval()
    train_data_size = len(test_loader.dataset)
    total_loss = 0
    for data, label in test_loader:
        data, label = Variable(data, volatile=True).cuda(), Variable(label).cuda()
        outputs = model(data)
        if hp.m2m:
            loss = loss_function(outputs.view(-1, outputs.size(-1)), label.view(-1))
        else:
            loss = loss_function(outputs, label)
        total_loss += loss

    return total_loss / train_data_size



def calc_perplexity(loss):
    return math.exp(loss)


if __name__ == "__main__":
    losses = []
    perplexities = []

    for epoch in range(hp.epochs):
        print("Epoch [{}] starting.. get ready\n".format(epoch+1))
        train_loss = train()
        test_loss = test()
        print("The perplexity score for this epoch is {}\n".format(calc_perplexity(test_loss)))
        losses.append(train_loss)
        perplexities.append(calc_perplexity(test_loss))

    plt.plot(list(range(1, hp.epochs+1)), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.show()

    plt.plot(list(range(1, hp.epochs+1)), perplexities)
    plt.xlabel("Epochs")
    plt.ylabel("Perplexities")
    plt.show()
