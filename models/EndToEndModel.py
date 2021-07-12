import torch
from torch.nn import Module

from random import choice, randint

from models.CnnModel import CnnModel


class EndToEndModel(Module):
    def __init__(self):
        super(EndToEndModel, self).__init__()
        self.circle = CnnModel()
        self.curve = CnnModel()
        self.line = CnnModel()

    def forward(self, x):
        y_pred = []
        circles = self.circle(x)
        curves = self.curve(x)
        lines = self.line(x)
        for i in range(0, len(x)):
            if torch.argmax(circles[i]) == 1 and torch.argmax(curves[i]) == 0 and torch.argmax(lines[i]) == 0:
                y_pred.append(float(0))
            if torch.argmax(circles[i]) == 0 and torch.argmax(curves[i]) == 0 and torch.argmax(lines[i]) == 1:
                y_pred.append(float(1))
            elif torch.argmax(circles[i]) == 1 and torch.argmax(curves[i]) == 0 and torch.argmax(lines[i]) == 2:
                y_pred.append(float(2))
            elif torch.argmax(circles[i]) == 0 and torch.argmax(curves[i]) == 2 and torch.argmax(lines[i]) == 0:
                y_pred.append(float(3))
            elif torch.argmax(circles[i]) == 0 and torch.argmax(curves[i]) == 0 and torch.argmax(lines[i]) == 3:
                y_pred.append(float(4))
            elif torch.argmax(circles[i]) == 0 and torch.argmax(curves[i]) == 1 and torch.argmax(lines[i]) == 2:
                y_pred.append(float(5))
            elif torch.argmax(circles[i]) == 1 and torch.argmax(curves[i]) == 1 and torch.argmax(lines[i]) == 0:
                y_pred.append(float(6))
            elif torch.argmax(circles[i]) == 0 and torch.argmax(curves[i]) == 0 and torch.argmax(lines[i]) == 2:
                y_pred.append(float(7))
            elif torch.argmax(circles[i]) == 2 and torch.argmax(curves[i]) == 0 and torch.argmax(lines[i]) == 0:
                y_pred.append(float(8))
            elif torch.argmax(circles[i]) == 1 and torch.argmax(curves[i]) == 0 and torch.argmax(lines[i]) == 1:
                y_pred.append(float(9))
            else:
                y_pred.append(float(randint(0, 9)))

        return torch.tensor(y_pred, requires_grad=True)
