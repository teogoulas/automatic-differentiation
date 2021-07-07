import os
import sys


from drawing.Drawing import open_window
from torchvision import transforms
import torch

from models.CnnModel import BinaryCnnModel
from utils.image_utils import view_classify
from utils.image_utils import convert


def set_parser(subparsers):
    parser = subparsers.add_parser("test", help="Test functions on custom digits")
    parser.set_defaults(func=run_test)

    parser.add_argument(
        "-c",
        "--custom",
        type=bool,
        help="Custom model",
    )


def run_test(args):
    if args.custom is not None:
        if not os.path.isfile("results/pretrained/circle_model.pt") or not os.path.isfile("results/pretrained/curve_model.pt") or os.path.isfile("results/pretrained/line_model.pt"):
            print('No custom model exists! Please execute command `python3 main.py train` to generate a custom model!')
            sys.exit(2)

    open_window()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    circle_model = BinaryCnnModel()
    curve_model = BinaryCnnModel()
    line_model = BinaryCnnModel()
    circle_model.load_state_dict(torch.load("results/custom/circle_model.pth" if args.custom is not None else "results/pretrained/circle_model.pth", map_location=device))
    curve_model.load_state_dict(torch.load("results/custom/curve_model.pth" if args.custom is not None else "results/pretrained/curve_model.pth", map_location=device))
    line_model.load_state_dict(torch.load("results/custom/line_model.pth" if args.custom is not None else "results/pretrained/line_model.pth", map_location=device))
    circle_model.eval()
    curve_model.eval()
    line_model.eval()

    trans = transforms.Compose([transforms.ToTensor()])
    img = convert('drawing/digit.png')
    img = trans(img).unsqueeze(0)
    with torch.no_grad():
        logps_circle = circle_model(img)
        logps_curve = curve_model(img)
        logps_line = line_model(img)

    probab_circle = list(logps_circle.cpu().numpy()[0])
    probab_curve = list(logps_curve.cpu().numpy()[0])
    probab_line = list(logps_line.cpu().numpy()[0])
    print("Is Circle = ", probab_circle[0] > 0.5)
    print("Is Curve = ", probab_curve[0] > 0.5)
    print("Is Line = ", probab_line[0] > 0.5)
    view_classify(img.view(1, 28, 28).cpu(), [round(1 - probab_circle[0]), round(probab_circle[0])], 2, "Circle")
    view_classify(img.view(1, 28, 28).cpu(), [round(1 - probab_curve[0]), round(probab_curve[0])], 2, "Curve")
    view_classify(img.view(1, 28, 28).cpu(), [round(1 - probab_line[0]), round(probab_line[0])], 2, "Line")
    print("123")
