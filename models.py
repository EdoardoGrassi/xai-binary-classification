import torch as tc
from torch import nn

class ConvNet(nn.Module):
    def __init__(self, shape: tuple[int, int, int]) -> None:
        assert shape[0] in (1, 3, 4)

        super().__init__()
        self.name = type(self).__name__.lower()

        channels, _, _ = shape
        self.conv0 = nn.Conv2d(channels, 64, kernel_size=(7), stride=2, padding=4)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=(5), stride=2, padding=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3), stride=1, padding=1)
        self.features = nn.Sequential(
            self.conv0,
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Dropout2d(p=0.25),

            self.conv1,
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Dropout2d(p=0.25),

            self.conv2,
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Dropout2d(p=0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 3 * 3, out_features=640),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=640, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, x: tc.Tensor):
        x = self.features.forward(x)
        x = self.classifier.forward(x)
        return x


from kymatio.torch import Scattering2D

class ScatNet(nn.Module):
    def __init__(self, shape: tuple[int, int, int]):
        assert all(x > 0 for x in shape)
        assert shape[0] in (1, 3, 4),\
            "Expected either GREYSCALE, RGB or RGBA image"

        super().__init__()
        self.name = type(self).__name__.lower()

        # TODO: justify values
        J = 5 # wavelet invariant scales
        L = 8 # wavelet invariant angles
        channels, w, h = shape

        # see https://www.kymat.io/userguide.html#output-size
        # only for order m = 2
        K = 1 + J * L + (L ** 2 * J * (J - 1)) // 2
        coefficients = K * (w // (2 ** J)) * (h // (2 ** J)) * channels

        self.features = Scattering2D(J=J, shape=(w, h), L=L, max_order=2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(in_features=coefficients, out_features=1024),
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.Linear(in_features=1024, out_features=512),
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.Linear(in_features=512, out_features=128),
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.Linear(in_features=128, out_features=1),
            nn.Linear(in_features=coefficients, out_features=640),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=640, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, x: tc.Tensor):
        x = self.features.forward(x)
        x = self.classifier.forward(x)
        return x