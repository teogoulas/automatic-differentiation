import os
import sys

from torch.autograd import Variable

from drawing.Drawing import open_window
from torchvision import transforms, datasets
import torch

from models.CnnModel import CnnModel
from sklearn.metrics import confusion_matrix, accuracy_score

from models.EndToEndModel import EndToEndModel
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
        if not os.path.isfile("results/custom/model.pth"):
            print('No custom model exists! Please execute command `python3 main.py train` to generate a custom model!')
            sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndToEndModel()
    print(model.circle)
    model.load_state_dict(torch.load(
        "results/custom/model.pth" if args.custom else "results/pretrained/model.pth",
        map_location=device))
    model.to(device)
    model.eval()

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

            names = ('0', '1', '2', '3')
            y_pred_circle = model.circle(X_batch)
            cm = confusion_matrix(y_batch_circle.cpu(), torch.argmax(y_pred_circle, dim=1).cpu().reshape(-1))
            circle_accuracy = accuracy_score(y_batch_circle.cpu(), torch.argmax(y_pred_circle, dim=1).cpu().reshape(-1))
            plot_confusion_matrix(cm, names, False, "Circle Function Confusion Matrix")

            y_pred_curve = model.curve(X_batch)
            cm = confusion_matrix(y_batch_curve.cpu(), torch.argmax(y_pred_curve, dim=1).cpu().reshape(-1))
            curve_accuracy = accuracy_score(y_batch_curve.cpu(), torch.argmax(y_pred_curve, dim=1).cpu().reshape(-1))
            plot_confusion_matrix(cm, names, False, "Curve Function Confusion Matrix")

            y_pred_line = model.line(X_batch)
            cm = confusion_matrix(y_batch_line.cpu(), torch.argmax(y_pred_line, dim=1).cpu().reshape(-1))
            line_accuracy = accuracy_score(y_batch_line.cpu(), torch.argmax(y_pred_line, dim=1).cpu().reshape(-1))
            plot_confusion_matrix(cm, names, False, "Line Function Confusion Matrix")
            print("Circle function Accuracy score: {}".format(circle_accuracy))
            print("Curve function Accuracy score: {}".format(curve_accuracy))
            print("Line function Accuracy score: {}".format(line_accuracy))
