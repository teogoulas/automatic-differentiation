import itertools

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
        print("logps_circle: {}".format(logps_circle))
        logps_curve = curve_model(img).cpu().numpy()[0]
        print("logps_curve: {}".format(logps_curve))
        logps_line = line_model(img).cpu().numpy()[0]
        print("logps_line: {}".format(logps_line))

    print("Number of Circles: ", np.argmax(logps_circle))
    print("Number of Curves: ", np.argmax(logps_curve))
    print("Number of Lines: ", np.argmax(logps_line))
    view_classify(img.view(1, 28, 28).cpu(), logps_circle, 4, "Circle")
    view_classify(img.view(1, 28, 28).cpu(), logps_curve, 4, "Curve")
    view_classify(img.view(1, 28, 28).cpu(), logps_curve, 4, "Line")


def plot_training_curve(train_counter, train_losses, test_counter, test_losses, model):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.plot(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.title("{} function Learning Curve".format(model))
    fig.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.show()
