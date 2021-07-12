# PyTorch libraries and modules
import torch
from torch.optim import Adam
from torch.utils.data import Dataset

# torchvision for pre-trained models
from torchvision import datasets, transforms

# custom libraries
from models.EndToEndModel import EndToEndModel
from utils.clfUtils import test, train
from utils.imageUtils import plot_samples, classify_image, plot_training_curve
from utils.constants import RANDOM_SEED, N_EPOCHS, LEARNING_RATE, BATCH_SIZE_TRAIN


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
        type=float,
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
    model = EndToEndModel().to(device)

    # set loss & optimizer functions
    optimizer = Adam(model.parameters(), lr=args.lr)

    # store accuracy and loss per epoch
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(args.e__epochs + 1)]

    test_loss = test(model, test_loader, device, args.batch)
    test_losses.append(test_loss)
    for epoch in range(1, args.e__epochs + 1):
        epoch_losses, epoch_counter = train(epoch, model, optimizer, train_loader, device)
        train_losses += epoch_losses
        train_counter += epoch_counter

        test_loss = test(model, test_loader, device, args.batch)
        test_losses.append(test_loss)

    torch.save(model.state_dict(), 'results/custom/model.pth')

    plot_training_curve(train_counter, train_losses, test_counter, test_losses, "")

    images, labels = next(iter(test_loader))

    img = images[0][None, :, :, :].to(device)
    classify_image(img, model.circle, model.curve, model.line)
