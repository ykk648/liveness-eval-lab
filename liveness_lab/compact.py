"""Compact MobileFaceNet-style classifiers for 80 by 80 RGB crops."""

import torch
from torch import nn


def channel_config(preset):
    transition_widths = {'compact-a': (180, 52), 'compact-b': (154, 103)}
    middle, late = transition_widths[preset]
    return (13, 13, 77, 77, 64, *((13, 13, 64) * 4), middle, middle, 128,
            *((26, 26, 128) * 6), late, late, 128, *((26, 26, 128) * 2), 512, 512)


def conv_unit(inputs, outputs, kernel, stride=1, padding=0, groups=1, activate=True):
    layers = [nn.Conv2d(inputs, outputs, kernel, stride, padding, groups=groups, bias=False),
              nn.BatchNorm2d(outputs)]
    if activate:
        layers.append(nn.PReLU(outputs))
    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    def __init__(self, inputs, expansion, depth, outputs, stride):
        super().__init__()
        self.branch = nn.Sequential(conv_unit(inputs, expansion, 1),
                                    conv_unit(expansion, depth, 3, stride, 1, expansion),
                                    conv_unit(depth, outputs, 1, activate=False))
        self.skip = stride == 1

    def forward(self, inputs):
        branch = self.branch(inputs)
        return inputs + branch if self.skip else branch


class CompactNet(nn.Module):
    def __init__(self, preset='compact-a', classes=3):
        super().__init__()
        widths = channel_config(preset)
        self.preset = preset
        self.stem = nn.Sequential(conv_unit(3, widths[0], 3, 2, 1),
                                  conv_unit(widths[0], widths[1], 3, 1, 1, widths[0]))
        blocks = []
        for index in range(2, 47, 3):
            stride = 2 if index in (2, 17, 38) else 1
            blocks.append(Bottleneck(*widths[index - 1:index + 3], stride))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.Sequential(conv_unit(widths[46], widths[47], 1),
                                  conv_unit(widths[47], widths[48], 5, groups=widths[47], activate=False))
        self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(widths[48], 128, bias=False),
                                       nn.BatchNorm1d(128), nn.Dropout(0.75))
        self.logits = nn.Linear(128, classes, bias=False)

    def forward(self, inputs):
        features = self.pool(self.blocks(self.stem(inputs)))
        return self.logits(self.embedding(features))


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = CompactNet(checkpoint['preset'], checkpoint['classes'])
    model.load_state_dict(checkpoint['state_dict'])
    return model
