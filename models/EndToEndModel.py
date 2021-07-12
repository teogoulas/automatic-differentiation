import torch
from torch.nn import Module

from random import choice, randint

from models.CnnModel import BinaryCnnModel


class EndToEndModel(Module):
    def __init__(self):
        super(EndToEndModel, self).__init__()
        self.circle = BinaryCnnModel()
        self.curve = BinaryCnnModel()
        self.line = BinaryCnnModel()

    def forward(self, x):
        y_pred = []
        circles = self.circle(x)
        curves = self.curve(x)
        lines = self.line(x)
        for i in range(0, len(x)):
            if circles[i].round() == 1 and curves[i].round() == 0 and lines[i].round() == 0:
                y_pred.append(float(choice([0, 8])))
            elif circles[i].round() == 1 and curves[i].round() == 0 and lines[i].round() == 1:
                y_pred.append(float(choice([2, 6, 9])))
            elif circles[i].round() == 1 and curves[i].round() == 1 and lines[i].round() == 0:
                y_pred.append(float(choice([6, 9])))
            elif circles[i].round() == 0 and curves[i].round() == 1 and lines[i].round() == 0:
                y_pred.append(float(3))
            elif circles[i].round() == 0 and curves[i].round() == 1 and lines[i].round() == 1:
                y_pred.append(float(choice([2, 5])))
            elif circles[i].round() == 0 and curves[i].round() == 0 and lines[i].round() == 1:
                y_pred.append(float(choice([1, 4, 7])))
            else:
                y_pred.append(float(randint(0, 9)))

        return torch.tensor(y_pred, requires_grad=True)
