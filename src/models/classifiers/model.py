import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentClassifier(nn.Module):
    def __init__(self, in_n, out_n):
        super(LatentClassifier, self).__init__()
        self.mlp = nn.Linear(in_n, out_n)

    def forward(self, x):
        logits = self.mlp(x)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


# Note: the following two classifiers for SVHN and MNIST are from MMVAE repository.
class SVHNClassifier(nn.Module):
    def __init__(self):
        super(SVHNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def extract_features(self, x, dims=500):
        assert dims in [500, 50]
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)  # (B, 500)
        if dims == 50:
            x = F.relu(self.fc1(x))  # (B, 50)
        return x


class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def extract_features(self, x, dims=320):
        assert dims in [320, 50]
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)  # (B, 320)
        if dims == 50:
            x = F.relu(self.fc1(x))  # (B, 50)
        return x


# Reference:
# https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x):
        flag = 0
        if len(x.shape)==5:
            flag = 1
            K, B = x.shape[:2]
            x = x.view(-1, *x.shape[2:])
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        if flag:
            logits = logits.view(K, B, -1)
            probs = probs.view(K, B, -1)
        return logits, probs

    def extract_features(self, batch, dims=120):
        assert dims in [120, 84]
        h = self.feature_extractor(batch).view(batch.shape[0], -1)  # (B, 120)
        if dims == 84:
            h = self.classifier[1](self.classifier[0](h))  # (B, 84)
        return h
