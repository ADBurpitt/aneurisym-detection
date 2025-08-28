from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet34, ResNet34_Weights
except Exception:
    resnet34 = None
    ResNet34_Weights = None


class _GAP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


def _adapt_first_conv(conv: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    if conv.in_channels == in_ch:
        return conv
    new = nn.Conv2d(
        in_ch,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None),
    )
    with torch.no_grad():
        if conv.weight.shape[1] == 3:  # ImageNet weights
            # average RGB to single and replicate to in_ch
            w = conv.weight.mean(dim=1, keepdim=True)  # [out,1,kh,kw]
            new.weight.copy_(w.repeat(1, in_ch, 1, 1) / in_ch)
        else:
            # repeat existing weights
            rep = int((in_ch + conv.weight.shape[1] - 1) // conv.weight.shape[1])
            w = conv.weight.repeat(1, rep, 1, 1)[:, :in_ch]
            new.weight.copy_(w * (conv.weight.shape[1] / in_ch))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class AttnPoolSimple(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # H: [N, C]
        A = self.fc2(self.tanh(self.fc1(H)))  # [N,1]
        A = torch.softmax(A.transpose(0, 1), dim=1)  # [1,N]
        M = torch.mm(A, H)  # [1,C]
        return M, A.squeeze(0)  # pooled, weights


class AttnPoolGated(nn.Module):
    # Gated attention (adds a sigmoid gate)
    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        temperature: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(), nn.Dropout(dropout)
        )
        self.u = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Sigmoid(), nn.Dropout(dropout)
        )
        self.w = nn.Linear(hidden, 1, bias=False)
        self.temperature = temperature

    def forward(self, H: torch.Tensor):
        s = self.w(self.v(H) * self.u(H)) / max(self.temperature, 1e-6)  # [N,1]
        A = torch.softmax(s.T, dim=1)  # [1,N]
        M = A @ H  # [1,C]
        return M, A.squeeze(0)


class ResNet25D(nn.Module):
    """2.5D MIL baseline using ResNet-34.

    - Input: a bag of instances [N, k, H, W]
    - Per-instance embed -> logits (14)
    - Bag pooling: mean or max
    """

    def __init__(
        self,
        num_classes: int = 14,
        k: int = 5,
        pool: str = "mean",
        pretrained: bool = True,
        attn_hidden: int = 128,
        attn_temp: float = 1.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        assert pool in {"mean", "max", "attn"}

        if resnet34 is None:
            raise ImportError(
                "torchvision not available; install torchvision to use ResNet25D"
            )

        weights = (
            ResNet34_Weights.DEFAULT
            if (pretrained and ResNet34_Weights is not None)
            else None
        )

        base = resnet34(weights=weights)
        base.conv1 = _adapt_first_conv(base.conv1, in_ch=k)

        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )

        self.gap = _GAP()

        self.pool = pool
        if pool == "attn_simple":
            self.attn = AttnPoolSimple(
                base.fc.in_features, attn_hidden, attn_temp, attn_dropout
            )
        elif pool == "attn_gated":
            self.attn = AttnPoolGated(
                base.fc.in_features, attn_hidden, attn_temp, attn_dropout
            )

        self.head = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(bag)  # [N,C,h,w]
        emb = self.gap(feats)  # [N,C]
        logits_inst = self.head(emb)  # [N,C]
        if self.pool == "mean":
            pooled = emb.mean(dim=0, keepdim=True)
        elif self.pool == "max":
            pooled, _ = emb.max(dim=0, keepdim=True)
        else:  # attn
            pooled, _ = self.attn(emb)  # [1,C]
        logits_bag = self.head(pooled)  # [1,C]
        return logits_inst, logits_bag
