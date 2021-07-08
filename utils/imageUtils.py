import cv2
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import torch


def convert(filepath):
    newImage = Image.open(filepath).resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN).convert('LA')
    newImage.save("drawing/mnist_digit.png")
    ii = cv2.imread("drawing/mnist_digit.png")
    gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    return gray_image


def plot_samples(test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure()
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Digit: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig.show()


def view_classify(img, ps, classes, model):
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze(), cmap='gray', interpolation='none')
    ax1.axis('off')
    ax2.barh(np.arange(classes), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(classes))
    ax2.set_yticklabels(np.arange(classes))
    ax2.set_title('{} Class Probability'.format(model))
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    fig.show()


def classify_image(img, circle_model, curve_model, line_model):
    with torch.no_grad():
        logps_circle = circle_model(img).cpu().numpy()[0]
        print("logps_circle: {}".format(logps_circle[0]))
        logps_curve = curve_model(img).cpu().numpy()[0]
        print("logps_curve: {}".format(logps_curve[0]))
        logps_line = line_model(img).cpu().numpy()[0]
        print("logps_line: {}".format(logps_line[0]))

    probab_circle = list(logps_circle)
    probab_curve = list(logps_curve)
    probab_line = list(logps_line)
    print("Is Circle = ", probab_circle[0] > 0.5)
    print("Is Curve = ", probab_curve[0] > 0.5)
    print("Is Line = ", probab_line[0] > 0.5)
    view_classify(img.view(1, 28, 28).cpu(), [round(1 - probab_circle[0]), round(probab_circle[0])], 2, "Circle")
    view_classify(img.view(1, 28, 28).cpu(), [round(1 - probab_curve[0]), round(probab_curve[0])], 2, "Curve")
    view_classify(img.view(1, 28, 28).cpu(), [round(1 - probab_line[0]), round(probab_line[0])], 2, "Line")
