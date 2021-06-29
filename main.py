# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import BCELoss
from torch.optim import SGD
from torch.utils.data import Dataset

# torchvision for pre-trained models
from torchvision import datasets, transforms

# custom libraries
from models.CnnModel import BinaryCnnModel
from utils.batchUtils import generate_batches
from utils.clfUtils import view_classify, get_accuracy, test, train
from utils.constants import RANDOM_SEED, N_EPOCHS, LEARNING_RATE, MOMENTUM


def main():
    # import MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    trainset = datasets.MNIST('./', download=True, train=True, transform=transform)
    valset = datasets.MNIST('./', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    # check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # torch.backends.cudnn.enabled = False
    torch.manual_seed(RANDOM_SEED)

    # initialize the model
    circle_model = BinaryCnnModel()
    curve_model = BinaryCnnModel()
    line_model = BinaryCnnModel()
    models = {
        'circle_model': circle_model,
        'curve_model': curve_model,
        'line_model': curve_model
    }

    # set loss & optimizer functions
    binary_criterion = BCELoss()
    optimizers = {
        'optimizer_circle': SGD(circle_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM),
        'optimizer_curve': SGD(curve_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM),
        'optimizer_line': SGD(line_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    }

    # store accuracy and loss per epoch
    train_losses_circle = []
    train_losses_curve = []
    train_losses_line = []
    train_counter = []
    test_losses_circle = []
    test_losses_curve = []
    test_losses_line = []
    test_counter = [i * len(trainloader.dataset) for i in range(N_EPOCHS + 1)]

    test_loss_circle, test_loss_curve, test_loss_line = test(models, valloader)
    test_losses_circle.append(test_loss_circle)
    test_losses_curve.append(test_loss_curve)
    test_losses_line.append(test_loss_line)
    for epoch in range(1, N_EPOCHS + 1):
        epoch_losses_circle, epoch_losses_curve, epoch_losses_line, epoch_counter = train(epoch, models, optimizers, trainloader)
        train_losses_circle += epoch_losses_circle
        train_losses_curve += epoch_losses_curve
        train_losses_line += epoch_losses_line
        train_counter += epoch_counter
        test_loss_circle, test_loss_curve, test_loss_line = test(models, valloader)
        test_losses_circle.append(test_loss_circle)
        test_losses_curve.append(test_loss_curve)
        test_losses_line.append(test_loss_line)

    images, labels = next(iter(valloader))

    img = images[0][None, :, :, :]
    with torch.no_grad():
        logps_circle = circle_model(img)
        logps_curve = curve_model(img)
        logps_line = line_model(img)

    probab_circle = list(logps_circle.numpy()[0])
    probab_curve = list(logps_curve.numpy()[0])
    probab_line = list(logps_line.numpy()[0])
    print("Is Circle = ", probab_circle[0] > 0.5)
    print("Is Curve = ", probab_curve[0] > 0.5)
    print("Is Line = ", probab_line[0] > 0.5)
    view_classify(img.view(1, 28, 28), [round(1 - probab_circle[0]), round(probab_circle[0])], 2)
    view_classify(img.view(1, 28, 28), [round(1 - probab_curve[0]), round(probab_curve[0])], 2)
    view_classify(img.view(1, 28, 28), [round(1 - probab_line[0]), round(probab_line[0])], 2)


if __name__ == '__main__':
    main()
