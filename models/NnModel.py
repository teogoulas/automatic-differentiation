from torch.nn import Sequential, Flatten, Linear, ReLU, Dropout, Sigmoid, Module


class NnModel(Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            Flatten(),
            Linear(784, 128), ReLU(),
            # BatchNorm1d(128),
            Dropout(0.2),
            Linear(128, 64), ReLU(),
            # BatchNorm1d(64),
            Dropout(0.2),
            Linear(64, 1), Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def string(self):
        return f'y = {self.model.item()}'

