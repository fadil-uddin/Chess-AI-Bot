"""
model.py

Defines the neural network architecture for the Chess AI, consisting of:
- Residual blocks (for deep feature extraction)
- Policy head (predicts move probabilities)
- Value head (predicts game outcome)

Author: [Syed Fadil Uddin]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    """
    Residual block with two convolutional layers and batch normalization.

    Args:
        c (int): Number of input/output channels.
    """
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        """
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    """
    Combined Policy-Value Network.

    - Policy head: predicts move probabilities over the action space.
    - Value head: predicts the expected outcome of the game.

    Args:
        planes (int): Number of input planes (default: 14).
        channels (int): Number of channels in convolution layers (default: 96).
        nblocks (int): Number of residual blocks (default: 8).
        action_size (int): Number of possible actions/moves (default: 4672).
    """
    def __init__(self, planes: int = 14, channels: int = 96,
                 nblocks: int = 8, action_size: int = 4672):
        super().__init__()

        # Initial convolution ("stem")
        self.stem = nn.Sequential(
            nn.Conv2d(planes, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Residual tower
        self.body = nn.Sequential(*[Residual(channels) for _ in range(nblocks)])

        # Policy head (move probabilities)
        self.p_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.p_fc = nn.Linear(32 * 8 * 8, action_size)

        # Value head (game outcome)
        self.v_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.GroupNorm(1, 1),
            nn.ReLU(inplace=True)
        )
        self.v_fc1 = nn.Linear(8 * 8, 128)
        self.v_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 8, 8, planes).

        Returns:
            p_out (torch.Tensor): Policy output (move probabilities).
            v_out (torch.Tensor): Value output (predicted outcome in [-1, 1]).
        """
        # Reorder dimensions to match Conv2D expectations: (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        # Shared trunk (stem + residual tower)
        h = self.body(self.stem(x))

        # Policy head
        p = self.p_head(h).reshape(h.size(0), -1)
        p_out = self.p_fc(p)

        # Value head
        v = self.v_head(h).reshape(h.size(0), -1)
        v_out = torch.tanh(self.v_fc2(F.relu(self.v_fc1(v)))).squeeze(1)

        return p_out, v_out
