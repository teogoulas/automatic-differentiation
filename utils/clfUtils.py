import torch
from sklearn import metrics
from torch.autograd import Variable
from torch.nn import BCELoss

from utils.batchUtils import generate_batches
from utils.constants import LOG_INTERVAL
from pathlib import Path


def train(epoch, models, optimizers, train_loader, device):
    Path("results/custom").mkdir(parents=True, exist_ok=True)

    train_losses_circle = []
    train_losses_curve = []
    train_losses_line = []
    train_counter = []

    binary_criterion = BCELoss()
    optimizer_circle = optimizers['optimizer_circle']
    optimizer_curve = optimizers['optimizer_curve']
    optimizer_line = optimizers['optimizer_line']

    circle_model = models['circle_model']
    curve_model = models['curve_model']
    line_model = models['line_model']
    circle_model.train()
    curve_model.train()
    line_model.train()

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        # normalize X_batch
        X_batch = Variable(X_batch).float()
        # generate y_batch per desired pattern
        y_batch_circle, y_batch_curve, y_batch_line = generate_batches(y_batch)
        X_batch, y_batch_circle, y_batch_curve, y_batch_line = X_batch.to(device), y_batch_circle.to(
            device), y_batch_curve.to(device), y_batch_line.to(device)

        # train circle model
        optimizer_circle.zero_grad()
        y_pred_circle = circle_model(X_batch)
        loss_circle_bin = binary_criterion(y_pred_circle.reshape(-1), y_batch_circle)
        loss_circle_bin.backward()
        optimizer_circle.step()

        # train curve model
        optimizer_curve.zero_grad()
        y_pred_curve = curve_model(X_batch)
        loss_curve_bin = binary_criterion(y_pred_curve.reshape(-1), y_batch_curve)
        loss_curve_bin.backward()
        optimizer_curve.step()

        # train line model
        optimizer_line.zero_grad()
        y_pred_line = line_model(X_batch)
        loss_line_bin = binary_criterion(y_pred_line.reshape(-1), y_batch_line)
        loss_line_bin.backward()
        optimizer_line.step()

        if batch_idx % LOG_INTERVAL == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Circle -> Loss: {:.6f} | Curve -> Loss: {:.6f} | Line -> Loss: {:.6f}'.format(
                    epoch, batch_idx * len(X_batch), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss_circle_bin.item(), loss_curve_bin.item(),
                    loss_line_bin.item()))
            train_losses_circle.append(loss_circle_bin.item())
            train_losses_curve.append(loss_curve_bin.item())
            train_losses_line.append(loss_line_bin.item())
            train_counter.append((batch_idx * len(X_batch)) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(circle_model.state_dict(), 'results/custom/circle_model.pth')
            # torch.save(optimizer_circle.state_dict(), 'results/custom/optimizer_circle.pth')
            torch.save(curve_model.state_dict(), 'results/custom/curve_model.pth')
            # torch.save(optimizer_curve.state_dict(), 'results/custom/optimizer_curve.pth')
            torch.save(line_model.state_dict(), 'results/custom/line_model.pth')
            # torch.save(optimizer_line.state_dict(), 'results/custom/optimizer_line.pth')

    return [train_losses_circle, train_losses_curve, train_losses_line, train_counter]


def test(models, train_loader, device):
    circle_model = models['circle_model']
    curve_model = models['curve_model']
    line_model = models['line_model']
    circle_model.eval()
    curve_model.eval()
    line_model.eval()

    test_loss_circle = 0
    test_loss_curve = 0
    test_loss_line = 0

    correct_circle = 0
    correct_curve = 0
    correct_line = 0

    binary_criterion = BCELoss()
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            # normalize X_batch
            X_batch = Variable(X_batch).float()
            # generate y_batch per desired pattern
            y_batch_circle, y_batch_curve, y_batch_line = generate_batches(y_batch)
            X_batch, y_batch_circle, y_batch_curve, y_batch_line = X_batch.to(device), y_batch_circle.to(
                device), y_batch_curve.to(device), y_batch_line.to(device)

            y_pred_circle = circle_model(X_batch)
            test_loss_circle += binary_criterion(y_pred_circle.reshape(-1), y_batch_circle)
            pred_circle = y_pred_circle.round()
            correct_circle += pred_circle.eq(y_batch_circle.data.view_as(pred_circle)).sum()

            y_pred_curve = curve_model(X_batch)
            test_loss_curve += binary_criterion(y_pred_curve.reshape(-1), y_batch_curve)
            pred_curve = y_pred_curve.round()
            correct_curve += pred_curve.eq(y_batch_curve.data.view_as(pred_curve)).sum()

            y_pred_line = line_model(X_batch)
            test_loss_line += binary_criterion(y_pred_line.reshape(-1), y_batch_line)
            pred_line = y_pred_line.round()
            correct_line += pred_line.eq(y_batch_line.data.view_as(pred_line)).sum()

    test_loss_circle /= len(train_loader.dataset)
    test_loss_curve /= len(train_loader.dataset)
    test_loss_line /= len(train_loader.dataset)
    print(
        '\nTest set: | Circle Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) | Curve Avg. loss: {:.4f}, Accuracy: {}/{} '
        '({:.0f}%) | Line Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss_circle, correct_circle, len(train_loader.dataset),
            100. * correct_circle / len(train_loader.dataset),
            test_loss_curve, correct_curve, len(train_loader.dataset), 100. * correct_curve / len(train_loader.dataset),
            test_loss_line, correct_line, len(train_loader.dataset), 100. * correct_line / len(train_loader.dataset)))

    return [test_loss_circle, test_loss_curve, test_loss_line]


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def get_accuracy(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred.reshape(-1).detach().numpy().round())
    return accuracy
