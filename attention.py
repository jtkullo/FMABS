import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Implementation of Channel Attention (CA) as described in Figure 2(a)[cite: 163].
    Input: Feature Inconsistency Matrix
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Global Pooling [cite: 129]
        y = self.avg_pool(x).view(b, c)
        # MLP + Sigmoid [cite: 130]
        y = self.mlp(y).view(b, c, 1, 1)
        return y

class SpatialAttention(nn.Module):
    """
    Implementation of Spatial Attention (SA) as described in Figure 2(b)[cite: 164].
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Concatenating Avg and Max pooling results results in 2 channels [cite: 132]
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # AvgPool & MaxPool along channel axis [cite: 132]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # Conv + Sigmoid [cite: 133]
        out = self.conv(x_cat)
        return self.sigmoid(out)

class SpatioChannelAttention(nn.Module):
    """
    Implementation of Spatio-Channel Attention (SCA) as described in Figure 2(c).
    This is the core novelty taking feature inconsistency directly as input[cite: 165].
    Formula: M = sigma(D(delta(E(X)))) [cite: 144]
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super(SpatioChannelAttention, self).__init__()
        # Small-scale auto-encoder structure [cite: 165]
        # Encoder E(.) [cite: 146]
        self.encoder = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU() # delta(.) [cite: 146]
        # Decoder D(.) [cite: 146]
        self.decoder = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid() # sigma(.) [cite: 146]

    def forward(self, x):
        # x is the Feature Inconsistency Matrix directly [cite: 142]
        out = self.encoder(x)
        out = self.relu(out)
        out = self.decoder(out)
        return self.sigmoid(out)
