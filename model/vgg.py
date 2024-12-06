from model.layer import *

feature_cfg = {
    'VGG5': [64, 'A', 128, 128, 'A', 'AA'],
    'VGG9': [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512, 'AA'],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512, 'AA'],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512],
    'CIFAR': [128, 256, 'A', 512, 'A', 1024, 512],
    'VGGSNN_CIFAR': [64, 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512, 'AA3'],
    'VGGSNN_DVS': [64, 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512, 'A'],
}

clasifier_cfg = {
    'VGG5': [128, 10],
    'VGG11': [512, 10],
    'VGG13': [512, 10],
    'VGG16': [2048, 4096, 4096, 10],
    'VGG19': [2048, 4096, 4096, 10],
    'VGGSNN_CIFAR': [4608, 100],
    'VGGSNN_DVS': [4608, 10]
}


class VGG(nn.Module):
    def __init__(self, architecture='VGG16', kernel_size=3, in_channel=3, use_bias=True,
                 num_class=10, **kwargs_spikes):
        super(VGG, self).__init__()
        self.kwargs_spikes = kwargs_spikes
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.use_bias = use_bias
        self.num_class = num_class
        clasifier_cfg[architecture][-1] = num_class
        self.feature = self._make_feature(feature_cfg[architecture])
        self.classifier = self._make_classifier(clasifier_cfg[architecture])
        self.readout = ReadOut()
        self._initialize_weights()

    def _make_feature(self, config):
        layers = []
        channel = self.in_channel
        for x in config:
            if x == 'A':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            elif x == 'AA':
                layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            elif x == 'AA3':
                layers.append(nn.AdaptiveAvgPool2d((3, 3)))
            else:
                layers.append(nn.Conv2d(in_channels=channel, out_channels=x, kernel_size=self.kernel_size,
                                        stride=1, padding=self.kernel_size // 2, bias=self.use_bias))

                layers.append(nn.BatchNorm2d(x))
                layers.append(LIFLayer(**self.kwargs_spikes))
                channel = x
        return nn.Sequential(*layers)

    def _make_classifier(self, config):
        layers = []
        for i in range(len(config) - 1):
            layers.append(nn.Linear(config[i], config[i + 1], bias=self.use_bias))
            layers.append(LIFLayer(**self.kwargs_spikes))
        layers.pop()
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.5)
                # m.weight.data.normal_(0, 0.5)
                # n = m.weight.size(1)
                # m.weight.data.normal_(0, 1.0 / float(n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        x = self.readout(x)
        return x


def vggsnn_cifar(num_classes=10, in_channel=3, **kwargs):
    return VGG(architecture="VGGSNN_CIFAR", in_channel=in_channel, num_class=num_classes, **kwargs)

def vggsnn_dvs(num_classes=10, in_channel=3, **kwargs):
    return VGG(architecture="VGGSNN_DVS", in_channel=in_channel, num_class=num_classes, **kwargs)

def vgg11(num_classes=10, in_channel=3, **kwargs):
    return VGG(architecture="VGG11", in_channel=in_channel, num_class=num_classes, **kwargs)


def vgg13(num_classes=10, in_channel=3, **kwargs):
    return VGG(architecture="VGG13", in_channel=in_channel, num_class=num_classes, **kwargs)
