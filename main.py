# importing the libraries
import pandas as pd
import numpy as np
## from tqdm import tqdm

# for reading and displaying images
## from skimage.io import imread
## from skimage.transform import resize
## import matplotlib.pyplot as plt


# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout, BatchNorm1d, BCELoss, Sigmoid, Flatten
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader

# torchvision for pre-trained models
from torchvision import models, datasets, transforms
from models.NnModel import NnModel
from models.CnnModel import CnnModel
from utils.batchUtils import generate_batches
from utils.clfUtils import binary_acc, view_classify


def main():
    # import MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    trainset = datasets.MNIST('./', download=True, train=True, transform=transform)
    valset = datasets.MNIST('./', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    base_model = CnnModel()

    base_model.train()

    criterion = CrossEntropyLoss()
    optimizer = Adam(base_model.parameters(), lr=0.001)
    for e in range(1, 11):
        epoch_loss = 0
        epoch_loss_curve = 0
        epoch_loss_line = 0

        epoch_acc = 0
        epoch_acc_curve = 0
        epoch_acc_line = 0
        for X_batch, y_batch in trainloader:
            # y_batch_circle, y_batch_curve, y_batch_line = generate_batches(y_batch)
            X_batch = Variable(X_batch).float()
            y_batch = Variable(y_batch)

            optimizer.zero_grad()
            # curve_optimizer.zero_grad()
            # line_optimizer.zero_grad()

            y_pred, _, _, _ = base_model(X_batch)
            # y_pred_curve = curve_model(X_batch)
            # y_pred_line = line_model(X_batch)

            loss = criterion(y_pred, y_batch)
            # loss_curve = criterion(y_pred_curve, y_batch_curve.unsqueeze(1))
            # loss_line = criterion(y_pred_line, y_batch_line.unsqueeze(1))

            # acc_curve = binary_acc(y_pred_curve, y_batch_curve.unsqueeze(1))
            # acc_line = binary_acc(y_pred_line, y_batch_line.unsqueeze(1))

            loss.backward()
            # loss_curve.backward()
            # loss_line.backward()

            optimizer.step()
            # curve_optimizer.step()
            # line_optimizer.step()

            epoch_loss += loss.item()
            # epoch_loss_curve += loss_curve.item()
            # epoch_loss_line += loss_line.item()

            epoch_acc += (torch.max(y_pred.data, 1)[1] == y_batch).sum()
            # poch_acc_curve += acc_curve.item()
            # poch_acc_line += acc_line.item()

        print(
            f'Circle: Epoch {e + 0:03}: | Loss: {epoch_loss / len(trainloader):.5f} | Acc: {epoch_acc / len(trainloader):.3f}')
        # print(
        #     f'Curve: Epoch {e + 0:03}: | Loss: {epoch_loss_curve / len(trainloader):.5f} | Acc: {epoch_acc_curve / len(trainloader):.3f}')
        # print(
        #     f'Line: Epoch {e + 0:03}: | Loss: {epoch_loss_line / len(trainloader):.5f} | Acc: {epoch_acc_line / len(trainloader):.3f}')

    images, labels = next(iter(valloader))

    img = images[0][None, :, :, :]
    with torch.no_grad():
        logps, logps_circle, logps_curve, logps_line = base_model(img)

    ps = torch.exp(logps)
    ps_circle = torch.exp(logps_circle)
    ps_curve = torch.exp(logps_curve)
    ps_line = torch.exp(logps_line)
    probab = list(ps.numpy()[0])
    probab_circle = list(ps_circle.numpy()[0])
    probab_curve = list(ps_curve.numpy()[0])
    probab_line = list(ps_line.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))
    print("Is Circle =", probab.index(max(probab_circle)))
    print("Is Curve =", probab.index(max(probab_curve)))
    print("Is Line =", probab.index(max(probab_line)))
    view_classify(img.view(1, 28, 28), ps, 10)
    view_classify(img.view(1, 28, 28), ps_circle, 10)
    view_classify(img.view(1, 28, 28), ps_curve, 10)
    view_classify(img.view(1, 28, 28), ps_line, 10)


if __name__ == '__main__':
    main()
