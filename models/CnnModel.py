from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, Flatten


class CnnModel(Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.model = Sequential(
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
            Linear(32 * 7 * 7, 4),
            Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
