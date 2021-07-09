import os
import sys

from torch.autograd import Variable

from drawing.Drawing import open_window
from torchvision import transforms, datasets
import torch

from models.CnnModel import BinaryCnnModel
from sklearn.metrics import confusion_matrix

from utils.batchUtils import generate_batches
from utils.imageUtils import plot_confusion_matrix


def set_parser(subparsers):
    parser = subparsers.add_parser("evaluate", help="Gets model's evaluation metrics")
    parser.set_defaults(func=run_eval)

    parser.add_argument(
        "-c",
        "--custom",
        action="store_true",
        default=False,
        help="Custom model",
    )


def run_eval(args):
    if args.custom:
        if not os.path.isfile("results/custom/circle_model.pth") or not os.path.isfile(
                "results/custom/curve_model.pth") or not os.path.isfile("results/custom/line_model.pth"):
            print('No custom model exists! Please execute command `python3 main.py train` to generate a custom model!')
            sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    circle_model = BinaryCnnModel()
    curve_model = BinaryCnnModel()
    line_model = BinaryCnnModel()
    circle_model.load_state_dict(torch.load(
        "results/custom/circle_model.pth" if args.custom else "results/pretrained/circle_model.pth",
        map_location=device))
    curve_model.load_state_dict(torch.load(
        "results/custom/curve_model.pth" if args.custom else "results/pretrained/curve_model.pth",
        map_location=device))
    line_model.load_state_dict(
        torch.load("results/custom/line_model.pth" if args.custom else "results/pretrained/line_model.pth",
                   map_location=device))
    circle_model.to(device)
    curve_model.to(device)
    line_model.to(device)
    circle_model.eval()
    curve_model.eval()
    line_model.eval()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    test_set = datasets.MNIST('./', download=True, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # normalize X_batch
            X_batch = Variable(X_batch).float()
            # generate y_batch per desired pattern
            y_batch_circle, y_batch_curve, y_batch_line = generate_batches(y_batch)
            X_batch, y_batch_circle, y_batch_curve, y_batch_line = X_batch.to(device), y_batch_circle.to(
                device), y_batch_curve.to(device), y_batch_line.to(device)

            y_pred_circle = circle_model(X_batch)
            cm = confusion_matrix(y_batch_circle.cpu(), y_pred_circle.round().cpu().reshape(-1))
            names = ('0', '1')
            plot_confusion_matrix(cm, names, False, "Circle Function Confusion Matrix")

            y_pred_curve = curve_model(X_batch)
            cm = confusion_matrix(y_batch_curve.cpu(), y_pred_curve.round().cpu().reshape(-1))
            names = ('0', '1')
            plot_confusion_matrix(cm, names, False, "Curve Function Confusion Matrix")

            y_pred_line = line_model(X_batch)
            cm = confusion_matrix(y_batch_line.cpu(), y_pred_line.round().cpu().reshape(-1))
            names = ('0', '1')
            plot_confusion_matrix(cm, names, False, "Line Function Confusion Matrix")
