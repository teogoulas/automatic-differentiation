import os
import sys

from drawing.Drawing import open_window
from torchvision import transforms
import torch

from models.CnnModel import BinaryCnnModel
from utils.imageUtils import view_classify, classify_image
from utils.imageUtils import convert


def set_parser(subparsers):
    parser = subparsers.add_parser("test", help="Test functions on custom digits")
    parser.set_defaults(func=run_test)

    parser.add_argument(
        "-c",
        "--custom",
        action="store_true",
        default=False,
        help="Custom model",
    )


def run_test(args):
    if args.custom:
        if not os.path.isfile("results/custom/circle_model.pth") or not os.path.isfile(
                "results/custom/curve_model.pth") or not os.path.isfile("results/custom/line_model.pth"):
            print('No custom model exists! Please execute command `python3 main.py train` to generate a custom model!')
            sys.exit(2)

    open_window()

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

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    img = convert('drawing/digit.png')
    img = trans(img).unsqueeze(0).to(device)
    classify_image(img, circle_model, curve_model, line_model)
