import torch
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SearchableResNet(nn.Module):
    def __init__(self, num_blocks, num_classes, width_multipliers, early_exit_location=None):
        super(SearchableResNet, self).__init__()

        self.inplanes = int(16 * width_multipliers[0])
        self.num_blocks = num_blocks
        self.width_multipliers = width_multipliers
        self.early_exit_location = early_exit_location

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, int(16 * width_multipliers[0]), num_blocks[0])
        self.layer2 = self._make_layer(BasicBlock, int(32 * width_multipliers[1]), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, int(64 * width_multipliers[2]), num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(int(64 * width_multipliers[2]) * BasicBlock.expansion, num_classes)
        
        self.early_exit_classifier = None
        if self.early_exit_location is not None:
            # Determine the number of output channels for the early exit
            exit_in_planes = self._get_inplanes_at_location(early_exit_location)
            self.early_exit_classifier = nn.Linear(exit_in_planes, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _get_inplanes_at_location(self, location):
        if location < self.num_blocks[0]:
            return int(16 * self.width_multipliers[0])
        elif location < self.num_blocks[0] + self.num_blocks[1]:
            return int(32 * self.width_multipliers[1])
        else:
            return int(64 * self.width_multipliers[2])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, manual_early_exit_index=None):
        x = self.relu(self.bn1(self.conv1(x)))

        block_count = 0
        
        # Layer 1
        for i, l in enumerate(self.layer1):
            x = l(x)
            if self.early_exit_location is not None and block_count == self.early_exit_location:
                pooled = self.avgpool(x)
                flat = pooled.view(pooled.size(0), -1)
                return [self.early_exit_classifier(flat)] # Return as list to match op_counter
            block_count += 1

        # Layer 2
        for i, l in enumerate(self.layer2):
            x = l(x)
            if self.early_exit_location is not None and block_count == self.early_exit_location:
                pooled = self.avgpool(x)
                flat = pooled.view(pooled.size(0), -1)
                return [self.early_exit_classifier(flat)]
            block_count += 1

        # Layer 3
        for i, l in enumerate(self.layer3):
            x = l(x)
            if self.early_exit_location is not None and block_count == self.early_exit_location:
                pooled = self.avgpool(x)
                flat = pooled.view(pooled.size(0), -1)
                return [self.early_exit_classifier(flat)]
            block_count += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        final_output = self.classifier(x)

        return final_output