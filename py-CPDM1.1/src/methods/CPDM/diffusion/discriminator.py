import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embed = nn.Linear(768, 256, True)
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
        )
        self.end_linear = nn.Linear(512, 1, True)

    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze(self):
        for parameter in self.parameters():
            parameter.requires_grad = True

    def forward(self, x, y):
        y = self.label_embed(y)
        x = self.layers(x)
        z = torch.cat((x.squeeze(), y.squeeze()), dim=1)
        z = self.end_linear(z)
        return z
