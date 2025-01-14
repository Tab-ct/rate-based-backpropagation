from torch import Tensor
from typing import Callable, List, Optional, Type, Union

from .layer import *

__all__ = ['sew_resnet34', ]


# modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=True,
                     dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


def _merge_synapse_neuron(synapse: Type[Union[nn.Conv2d, nn.Linear]] = None, neuron: callable = None,
                          bn_dim: int = None, **kwargs):
    bn = nn.BatchNorm2d(bn_dim)
    layers = [synapse, bn, neuron(**kwargs)]

    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            neuron: callable = None,
            **kwargs,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        conv1 = conv3x3(inplanes, planes, stride)
        self.convNeuron1 = _merge_synapse_neuron(conv1, neuron, planes, **kwargs)
        conv2 = conv3x3(planes, planes)
        self.convNeuron2 = _merge_synapse_neuron(conv2, neuron, planes, **kwargs)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor):
        identity = x

        out = self.convNeuron1(x)
        out = self.convNeuron2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out


class SewSpikingResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock,]],
            layers: List[int],
            num_classes: int = 1000,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            neuron: callable = None,
            **kwargs
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.convNeuron1 = _merge_synapse_neuron(conv1, neuron, self.inplanes, **kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], neuron=neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       neuron=neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       neuron=neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       neuron=neuron, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.readout = ReadOut()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
            self,
            block: Type[Union[BasicBlock,]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            neuron: callable = None,
            **kwargs
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv_down = conv1x1(self.inplanes, planes * block.expansion, stride)
            downsample = _merge_synapse_neuron(conv_down, neuron, planes * block.expansion, **kwargs)

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,
                neuron, **kwargs
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    neuron=neuron,
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:

        out = self.convNeuron1(x)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = self.readout(out)
        return out


def spiking_resnet(arch, block, layers, pretrained, progress, single_step_neuron, **kwargs):
    model = SewSpikingResNet(block, layers, neuron=single_step_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def sew_resnet34(pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs):
    return spiking_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                          single_step_neuron=LIFLayer, **kwargs)
