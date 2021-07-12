import torch
from sklearn import metrics
from torch.autograd import Variable
from torch.nn import MSELoss

from utils.constants import LOG_INTERVAL
from pathlib import Path


def train(epoch, model, optimizer, train_loader, device):
    Path("results/custom").mkdir(parents=True, exist_ok=True)

    train_losses = []
    train_counter = []

    criterion = MSELoss()
    model.train()

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        # normalize X_batch
        X_batch, y_batch = Variable(X_batch).float().to(device), Variable(y_batch).float().to(device)

        # train model
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.6f}'.format(
                    epoch, batch_idx * len(X_batch), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * len(X_batch)) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(model.state_dict(), 'results/custom/model.pth')

    return [train_losses, train_counter]


def test(model, train_loader, device, batch_size):
    model.eval()

    test_loss = 0
    correct = 0

    criterion = MSELoss()
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            # normalize batches
            X_batch, y_batch = Variable(X_batch).float().to(device), Variable(y_batch).float().to(device)

            y_pred = model(X_batch)
            test_loss += criterion(y_pred, y_batch)
            correct += y_pred.eq(y_batch.data.view_as(y_pred)).sum()

    test_loss /= round(len(train_loader.dataset)/batch_size)
    print(
        '\nTest set: | Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))

    return test_loss


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def get_accuracy(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred.reshape(-1).detach().numpy().round())
    return accuracy
