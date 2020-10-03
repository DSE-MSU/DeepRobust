"""
Model for YOPO.

Reference
---------
..[1]https://github.com/a1600012888/YOPO-You-Only-Propagate-Once

"""

from collections import OrderedDict
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, drop=0.5):
        super(Net, self).__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)
        self.conv1 = nn.Conv2d(self.num_channels, 32, 3)
        self.layer_one = nn.Sequential(OrderedDict([
            ('conv1', self.conv1),
            ('relu1', activ),]))

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))
        self.other_layers = nn.ModuleList()
        self.other_layers.append(self.feature_extractor)
        self.other_layers.append(self.classifier)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        y = self.layer_one(input)
        self.layer_one_out = y
        self.layer_one_out.requires_grad_()
        self.layer_one_out.retain_grad()
        features = self.feature_extractor(y)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits


