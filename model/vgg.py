import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, vgg_name, in_channels=1, out_channels=7, p_dropout=0.5):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, out_channels)
        self.dropout = nn.Dropout2d(p=p_dropout, inplace=True)

    def forward(self, x):
        out = self.features(x)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=4, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11', 1, 7)
    print(net)
    x = torch.randn(2, 1, 128, 128)
    y = net(x)
    print(y)
    print(y.size())


if __name__ == "__main__":
    test()
