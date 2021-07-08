# PyTorch libraries and modules
import torch
from torch.optim import SGD
from torch.utils.data import Dataset

# torchvision for pre-trained models
from torchvision import datasets, transforms

# custom libraries
from models.CnnModel import BinaryCnnModel
from utils.clfUtils import test, train
from utils.imageUtils import view_classify, plot_samples, classify_image
from utils.constants import RANDOM_SEED, N_EPOCHS, LEARNING_RATE, MOMENTUM, BATCH_SIZE_TRAIN


def set_parser(subparsers):
    parser = subparsers.add_parser("train", help="Train functions")
    parser.set_defaults(func=run_train)

    parser.add_argument(
        "-e"
        "--epochs",
        type=int,
        default=N_EPOCHS,
        help="Number of training epochs",
    )

    parser.add_argument(
        "-lr",
        type=int,
        default=LEARNING_RATE,
        help="Learning rate",
    )

    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=BATCH_SIZE_TRAIN,
        help="Batch size",
    )

    parser.add_argument(
        "-m",
        "--momentum",
        type=int,
        default=MOMENTUM,
        help="Batch size",
    )


def run_train(args):
    # import MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    train_set = datasets.MNIST('./', download=True, train=True, transform=transform)
    test_set = datasets.MNIST('./', download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch, shuffle=True)

    # check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Available device {}".format(device))

    # torch.backends.cudnn.enabled = False
    torch.manual_seed(RANDOM_SEED)

    plot_samples(train_loader)

    # initialize the model
    circle_model = BinaryCnnModel().to(device)
    curve_model = BinaryCnnModel().to(device)
    line_model = BinaryCnnModel().to(device)
    models = {
        'circle_model': circle_model,
        'curve_model': curve_model,
        'line_model': curve_model
    }

    # set loss & optimizer functions
    optimizers = {
        'optimizer_circle': SGD(circle_model.parameters(), lr=args.lr, momentum=args.momentum),
        'optimizer_curve': SGD(curve_model.parameters(), lr=args.lr, momentum=args.momentum),
        'optimizer_line': SGD(line_model.parameters(), lr=args.lr, momentum=args.momentum)
    }

    # store accuracy and loss per epoch
    train_losses_circle = []
    train_losses_curve = []
    train_losses_line = []
    train_counter = []
    test_losses_circle = []
    test_losses_curve = []
    test_losses_line = []

    test_loss_circle, test_loss_curve, test_loss_line = test(models, test_loader, device)
    test_losses_circle.append(test_loss_circle)
    test_losses_curve.append(test_loss_curve)
    test_losses_line.append(test_loss_line)
    for epoch in range(1, args.e__epochs + 1):
        epoch_losses_circle, epoch_losses_curve, epoch_losses_line, epoch_counter = train(epoch, models, optimizers,
                                                                                          train_loader, device)
        train_losses_circle += epoch_losses_circle
        train_losses_curve += epoch_losses_curve
        train_losses_line += epoch_losses_line
        train_counter += epoch_counter
        test_loss_circle, test_loss_curve, test_loss_line = test(models, test_loader, device)
        test_losses_circle.append(test_loss_circle)
        test_losses_curve.append(test_loss_curve)
        test_losses_line.append(test_loss_line)

    torch.save(circle_model.state_dict(), 'results/custom/circle_model.pth')
    torch.save(curve_model.state_dict(), 'results/custom/curve_model.pth')
    torch.save(line_model.state_dict(), 'results/custom/line_model.pth')

    images, labels = next(iter(test_loader))

    img = images[0][None, :, :, :].to(device)
    classify_image(img, circle_model, curve_model, line_model)
