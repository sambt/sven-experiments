import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .batchnorm import BatchNorm2d as customBatchNorm2D, replace_batchnorm
from torchvision.models import resnet18
from torchvision.models import resnet18 as _tv_resnet18, ResNet18_Weights


def resnet18_pretrained_functional(num_classes: int = 10, freeze_backbone: bool = False,
                                   **kwargs) -> nn.Module:
    """ImageNet-pretrained torchvision ResNet18, torch.func-compatible, new head.

    For the fine-tuning experiment (genuine P >> N): ~11M pretrained parameters
    adapted to a small labelled set. BatchNorm is swapped for the transform-safe
    variant (frozen to its ImageNet running stats by the Gram wrapper). The final
    fc is replaced to match ``num_classes`` (fresh init). ``freeze_backbone`` is
    accepted for experiments that only train the head, but by default everything
    trains.
    """
    model = _tv_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    replace_batchnorm(model)
    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad_(False)
    return model


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with explicit q/k/v/proj nn.Linear layers.

    Kept as separate nn.Linear modules (not a packed qkv) so the Gram hooks
    capture sees standard Linear layers; the attention softmax/matmuls are
    parameter-free and per-sample (positions couple within a sequence, never
    across the batch), so they need no capture.
    """
    def __init__(self, n_embd, n_head, block_size, bias=True):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.q = nn.Linear(n_embd, n_embd, bias=bias)
        self.k = nn.Linear(n_embd, n_embd, bias=bias)
        self.v = nn.Linear(n_embd, n_embd, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        h = self.n_head
        q = self.q(x).view(B, T, h, C // h).transpose(1, 2)  # (B, h, T, d)
        k = self.k(x).view(B, T, h, C // h).transpose(1, 2)
        v = self.v(x).view(B, T, h, C // h).transpose(1, 2)
        # Fused causal attention (memory-efficient at long context). Parameter-free,
        # so the Gram hooks (which capture the q/k/v/proj Linear grad-outputs) are
        # unaffected; mathematically identical to explicit softmax attention.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class _Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    """Minimal GPT kept on the Gram fast (hooks) path.

    Design constraints for hooks capture (exact per-sample Gram without the
    (B, P) Jacobian): (1) NO weight tying between token embedding and lm_head
    (tying forces the slow chunked capture); (2) dropout = 0 (dropout resamples
    between the capture and update passes); (3) positional lookup uses an
    nn.Embedding fed (B, T) indices, not a bare nn.Parameter table (invisible to
    hooks) or 1-D arange indices (rejected). All parameters therefore live in
    captured nn.Linear / nn.LayerNorm / nn.Embedding modules.
    """
    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4,
                 n_embd=128, tie_weights=False):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[_Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        if tie_weights:  # only usable with capture="chunked"
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)  # (B, T), hooks-safe
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)  # (B, T, vocab)


def resnet18_functional(num_classes: int = 10, **kwargs) -> nn.Module:
    """torchvision ResNet18 with torch.func-compatible BatchNorm (required for Sven).

    Standard nn.BatchNorm2d updates running_mean/running_var in-place during the
    forward pass, which crashes inside torch.func transforms (jacrev, grad, vmap…).
    This wrapper replaces every BatchNorm2d with the drop-in version from
    experiments.nn.batchnorm that skips those in-place writes when inside a transform.
    """
    model = resnet18(num_classes=num_classes, **kwargs)
    replace_batchnorm(model)
    return model


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
    def __init__(self, num_classes=10, width=16, num_blocks=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = customBatchNorm2D(width)

        self.layer1 = self._make_layer(width, width, num_blocks=num_blocks, stride=1)
        self.layer2 = self._make_layer(width, width*2, num_blocks=num_blocks, stride=2)
        self.layer3 = self._make_layer(width*2, width*4, num_blocks=num_blocks, stride=2)
        
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