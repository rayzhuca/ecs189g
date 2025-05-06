from torch.utils.data import DataLoader, TensorDataset
from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy

from local_code.stage_3_code.Method_CNN_Base import Method_CNN_Base

device = "mps"

class Method_CNN_ORL(Method_CNN_Base):
    max_epoch = 10
    learning_rate = 1e-3
    l1_lambda_reg = 0
    l2_lambda_reg = 0

    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 14 * 11, 128)
        self.act_1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 40)

        self.pool = nn.MaxPool2d(2, 2)
        self.to(device)

    def forward(self, x):
        x = self.pool(self.bn1(self.conv1(x)))
        x = self.pool(self.bn2(self.conv2(x)))
        x = self.pool(self.bn3(self.conv3(x)))

        x = torch.flatten(x, 1)

        x = self.dropout(x)

        x = self.act_1(self.fc1(x))
        x = self.fc2(x)
        return x