import torch

import torch.nn as nn

import torchvision.models as models

class VGG16(nn.Module):

    def __init__(self):

        super(VGG16, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.classifier = nn.Sequential(

            nn.Linear(512 * 7 * 7, 4096),

            nn.ReLU(True),

            nn.Dropout(),

            nn.Linear(4096, 4096),

            nn.ReLU(True),

            nn.Dropout(),

            nn.Linear(4096, 1000),

        )

    def load_weights(self, path):

        state_dict = torch.load(path)

        self.load_state_dict(state_dict)

    def get_features(self, x, layer_name):

        for name, module in self.features.named_children():

            x = module(x)

            if name == layer_name:

                return x

        return None

