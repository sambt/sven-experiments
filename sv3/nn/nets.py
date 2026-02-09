import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .batchnorm import BatchNorm2d as customBatchNorm2D


class MultiLinear(nn.Module):
    """Batched linear layer: num_models independent linear transforms in parallel.

    Parameters have shape (num_models, out_features, in_features) for weight
    and (num_models, out_features) for bias, so that all models are computed
    via a single bmm.

    Input:  (num_models, batch, in_features)
    Output: (num_models, batch, out_features)
    """
    def __init__(self, num_models, in_features, out_features, bias=True):
        super().__init__()
        self.num_models = num_models
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_models, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_models, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Match nn.Linear init: kaiming_uniform per model slice
        for i in range(self.num_models):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in = self.in_features
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x):
        # x: (num_models, batch, in_features)
        out = torch.bmm(x, self.weight.transpose(1, 2))
        if self.bias is not None:
            out = out + self.bias.unsqueeze(1)
        return out


class MultiMLP(nn.Module):
    """Ensemble of num_models independent MLPs, batched into a single forward pass.

    Input:  (batch, input_dim)
    Output: (num_models, batch, output_dim)
    """
    def __init__(self, num_models, input_dim, hidden_dims, output_dim, activation=nn.GELU):
        super().__init__()
        self.num_models = num_models
        if isinstance(activation, str):
            activation = getattr(nn, activation.upper())
        layers = [MultiLinear(num_models, input_dim, hidden_dims[0]), activation()]
        for i in range(1, len(hidden_dims)):
            layers.append(MultiLinear(num_models, hidden_dims[i-1], hidden_dims[i]))
            layers.append(activation())
        layers.append(MultiLinear(num_models, hidden_dims[-1], output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # x: (batch, input_dim) -> (num_models, batch, input_dim)
        x = x.unsqueeze(0).expand(self.num_models, -1, -1)
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.GELU):
        super().__init__()
        # Handle string activation names
        if isinstance(activation, str):
            activation = getattr(nn, activation.upper())
        layers = [nn.Linear(input_dim, hidden_dims[0]), activation()]
        for i in range(1,len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = customBatchNorm2D(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = customBatchNorm2D(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                customBatchNorm2D(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SmallResNet(nn.Module):
    def __init__(self, num_classes=10, width=16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = customBatchNorm2D(width)
        
        self.layer1 = self._make_layer(width, width, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(width, width*2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(width*2, width*4, num_blocks=2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width*4, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class SmallCNN(nn.Module):
    """
    Compact CNN for CIFAR-10
    """
    def __init__(self):
        super(SmallCNN, self).__init__()
        
        # First conv block: 3 -> 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second conv block: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third conv block: 64 -> 64 channels
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Block 2: Conv -> BN -> ReLU -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Block 3: Conv -> BN -> ReLU -> Pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Flatten
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        
        # FC layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x