from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Sigmoid, Flatten, Softmax

from utils.batchUtils import split_batches

base_model = Sequential(
    Conv2d(
        in_channels=1,
        out_channels=16,
        kernel_size=(5, 5),
        stride=1,
        padding=2,
    ),
    ReLU(),
    MaxPool2d(kernel_size=2),
    Conv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=(5, 5),
        stride=1,
        padding=2
    ),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(32 * 7 * 7, 10),
    Softmax()
)

binary_model = Sequential(
    Conv2d(
        in_channels=1,
        out_channels=16,
        kernel_size=(5, 5),
        stride=1,
        padding=2,
    ),
    ReLU(),
    MaxPool2d(kernel_size=2),
    Conv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=(5, 5),
        stride=1,
        padding=2
    ),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(32 * 7 * 7, 1),
    Sigmoid()
)


class CnnModel(Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.base_model = base_model
        self.circle_model = binary_model
        self.curve_model = binary_model
        self.line_model = binary_model

    def forward(self, x):
        output = self.base_model(x)
        x_circle, x_curve, x_line = split_batches(x, output)
        circle_output = self.circle_model(x_circle) if x_circle is not None else None
        curve_output = self.curve_model(x_curve) if x_curve is not None else None
        line_output = self.line_model(x_line) if x_line is not None else None
        return output, circle_output, curve_output, line_output
