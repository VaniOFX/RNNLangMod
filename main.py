from dataloader import train_loader, test_loader
from RNN import model, optimizer, loss_function
from torch.autograd import Variable
import hyperparameters as hp
import math
import matplotlib.pyplot as plt


def train():
    model.train()
    total_loss = 0
    train_data_size = len(train_loader.dataset)
    for data, label in train_loader:
        data, label = Variable(data).cuda(), Variable(label).cuda()
        outputs = model(data)
        optimizer.zero_grad()
        print(outputs.size())
        print(label.size())
        loss = loss_function(outputs, label)
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    return total_loss / train_data_size


def test():
    model.eval()
    correct = 0
    train_data_size = len(test_loader.dataset)
    for data, label in test_loader:
        data, label = Variable(data, volatile=True).cuda(), Variable(label).cuda()
        outputs = model(data)
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, train_data_size, 100. * correct / train_data_size))


def calc_perplexity(loss):
    return math.exp(loss)


if __name__ == "__main__":
    losses = []
    perplexities = []

    for epoch in range(hp.epochs):
        print("Epoch [{}] starting.. get ready\n".format(epoch))
        loss_data = train()
        test()
        print("The perplexity score for this epoch is {}".format(calc_perplexity(loss_data)))
        losses.append(loss_data)
        perplexities.append(calc_perplexity(loss_data))

    plt.plot(list(range(1, hp.epochs+1)), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.show()

    plt.plot(list(range(1, hp.epochs+1)), perplexities)
    plt.xlabel("Epochs")
    plt.ylabel("Perplexities")
    plt.show()
