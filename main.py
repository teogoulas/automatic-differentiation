# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset

# torchvision for pre-trained models
from torchvision import datasets, transforms
from models.CnnModel import CnnModel
from utils.clfUtils import view_classify


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
    criterion = CrossEntropyLoss()
    optimizer = Adam(base_model.parameters(), lr=0.001)

    base_model.train()
    for e in range(1, 11):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in trainloader:
            X_batch = Variable(X_batch).float()
            y_batch = Variable(y_batch)

            optimizer.zero_grad()
            y_pred, _, _, _ = base_model(X_batch)

            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += (torch.max(y_pred.data, 1)[1] == y_batch).sum()

        print(
            f'Circle: Epoch {e + 0:03}: | Loss: {epoch_loss / len(trainloader):.5f} | Acc: {epoch_acc / len(trainloader):.3f}')

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
    view_classify(img.view(1, 28, 28), ps_circle, 2)
    view_classify(img.view(1, 28, 28), ps_curve, 2)
    view_classify(img.view(1, 28, 28), ps_line, 2)


if __name__ == '__main__':
    main()
