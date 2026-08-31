import torch
import torch.nn as nn
from kymatio.torch import Scattering2D

# from models.classifiers import MLP


# class ScatNet2D(nn.Module):
#     def __init__(self, shape: tuple[int, int, int], classes: int, pca: torch.Tensor):
#         assert all(x > 0 for x in shape)
#         assert classes > 0

#         super().__init__()

#         # TODO: justify values
#         J = 4 # wavelet invariant scales
#         L = 8 # wavelet invariant angles
#         image_color_channels, w, h = shape

#         # see https://www.kymat.io/userguide.html#output-size
#         # only for order m = 2
#         K = 1 + J * L + (L ** 2 * J * (J - 1)) // 2
#         coefficients = K * (w // (2 ** J)) * (h // (2 ** J)) * image_color_channels

#         self.features = Scattering2D(J=J, shape=(w, h), L=L, max_order=2)
#         self.pca = pca
#         self.classifier = MLP(inputs=pca.shape[-1], outputs=classes, hidden=[1024, 1024])

#     def forward(self, x: torch.Tensor):
#         assert x is not None
        
#         x = self.features.forward(x)
#         x = x.flatten(start_dim=1) # flatten but keep batches shape
#         x = x.matmul(self.pca)
#         x = self.classifier.forward(x)
#         return x


class ScatNet(nn.Module):
    def __init__(self, shape: tuple[int, int, int]):
        assert all(x > 0 for x in shape)
        assert shape[0] in (1, 3, 4),\
            "Expected either GREYSCALE, RGB or RGBA image"

        super().__init__()
        self.name = type(self).__name__.lower()

        # TODO: justify values
        J = 6 # wavelet invariant scales
        # J = 3 # wavelet invariant scales
        L = 8 # wavelet invariant angles
        channels, w, h = shape

        # see https://www.kymat.io/userguide.html#output-size
        # only for order m = 2
        K = 1 + J * L + (L ** 2 * J * (J - 1)) // 2
        coefficients = K * (w // (2 ** J)) * (h // (2 ** J)) * channels

        self.features = Scattering2D(J=J, shape=(w, h), L=L, max_order=2)
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     # nn.Dropout(p=0.2),
        #     nn.Linear(in_features=coefficients, out_features=classes),
        # )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(p=0.2),
            nn.Linear(in_features=coefficients, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2), # binary classifier
        )

    def forward(self, x: torch.Tensor):
        x = self.features.forward(x)
        x = self.classifier.forward(x)
        return x