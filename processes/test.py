import os
import sys

from drawing.Drawing import open_window
from torchvision import transforms
import torch

from models.EndToEndModel import EndToEndModel
from utils.imageUtils import classify_image
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
        if not os.path.isfile("results/custom/model.pth"):
            print('No custom model exists! Please execute command `python3 main.py train` to generate a custom model!')
            sys.exit(2)

    open_window()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndToEndModel()
    model.load_state_dict(torch.load(
        "results/custom/model.pth" if args.custom else "results/pretrained/model.pth",
        map_location=device))

    model.to(device)
    model.eval()

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    img = convert('drawing/digit.png')
    img = trans(img).unsqueeze(0).to(device)
    classify_image(img, model.circle, model.curve, model.line)
