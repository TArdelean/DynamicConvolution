import torch
import torch.nn.functional as F
from torch import nn

# Simple example for testing from https://github.com/pytorch/examples/blob/master/mnist/main.py
from models.common import BaseModel


class SimpleConvNet(BaseModel):
    def __init__(self, ConvLayer, c_in=1, out_dim=10):
        super().__init__(ConvLayer)
        self.conv1 = nn.Conv2d(c_in, 32, 3, 1)  # First convolution is always non-dynamic
        self.conv2 = self.ConvLayer(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x, temperature):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x, temperature)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
