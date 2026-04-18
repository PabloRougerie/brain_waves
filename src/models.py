"""
Model definitions for EEG spectrogram classification.

All models take input of shape (batch, n_channels=4, n_freq=100, n_time=25)
and return log-softmax probabilities over n_classes=6 brain activity patterns.

Sections:
  - Baseline:   ConvBlock, BaselineModel (VGG-style CNN)
  - Hybrid:     CNNHybrid, LSTMHybrid, HybridModel (CNN + 4 parallel LSTMs)
  - Sequential: CNNSequential, LSTMSequential, SequentialModel (CNN → LSTM)
  - Optuna:     OptunaModel (configurable VGG-style CNN, used for hyperparameter search)
"""

import torch
from torch import nn


# ─────────────────────────────────────────────────────────────────────────────
# Baseline
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU block, with optional Dropout2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel: int = 3,
        padding: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                kernel_size=kernel,
                bias=False,  # bias is redundant with BatchNorm
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineModel(nn.Module):
    """Hardcoded 2-block VGG-style CNN for EEG spectrogram classification.

    Two convolutional blocks (32→64, 128→256 channels) with MaxPool and Dropout,
    followed by an adaptive average pool and a two-layer MLP head.

    Input shape:  (batch, n_channels, n_freq, n_time)
    Output:       log-softmax over n_classes
    """

    def __init__(self, n_classes: int, n_channels: int):
        super().__init__()
        self.n_classes  = n_classes
        self.n_channels = n_channels

        self.block1 = nn.Sequential(
            ConvBlock(in_channels=self.n_channels, out_channels=32),
            ConvBlock(in_channels=32,              out_channels=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.3),
        )

        self.block2 = nn.Sequential(
            ConvBlock(in_channels=64,  out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.3),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.0),
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.Linear(64, self.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return nn.functional.log_softmax(x, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid (CNN + 4 parallel LSTMs)
# ─────────────────────────────────────────────────────────────────────────────

class CNNHybrid(nn.Module):
    """Configurable CNN backbone for the parallel hybrid model.

    Processes the full spectrogram (all channels) through a stack of
    ConvBlock + MaxPool stages to produce a global spatial embedding.

    Input shape:  (batch, n_channels, n_freq, n_time)
    Output shape: (batch, hidden_dims[-1])
    """

    def __init__(
        self,
        n_channels: int,
        hidden_dims: list = [16, 32, 64],
        dropout: float = 0.0,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.n_channels = n_channels

        dims = [n_channels] + hidden_dims
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                ConvBlock(dims[i], dims[i + 1], kernel=kernel_size, padding=padding, dropout=dropout),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            for i in range(len(dims) - 1)
        ])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.head(x)
        return x  # (batch, hidden_dims[-1])


class LSTMHybrid(nn.Module):
    """Bidirectional LSTM encoder for a single spectrogram channel.

    Takes a time sequence and returns the last hidden state,
    concatenating forward and backward directions when bidirectional.

    Input shape:  (batch, seq_len, n_features)
    Output shape: (batch, hidden_size * num_directions)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        # LSTM dropout only applies between layers — no effect with num_layers=1
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            # concatenate last forward and last backward hidden states
            return torch.cat([h_n[-2], h_n[-1]], dim=1)
        return h_n[-1]


class HybridModel(nn.Module):
    """CNN + 4 parallel LSTMs hybrid model for EEG spectrogram classification.

    Architecture:
      - One CNNHybrid processes the full spectrogram (all channels) → spatial embedding
      - 4 LSTMHybrid process each EEG region (LL, RL, LP, RP) independently along
        the time axis → 4 temporal embeddings
      - All embeddings are concatenated and passed to a two-layer MLP head

    Input shape:  (batch, n_channels=4, n_freq, n_time)
    Output:       log-softmax over n_classes
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        cnn_hidden_dims: list = [16, 32, 64],
        n_freq: int = 100,
        lstm_hidden_size: int = 32,
        n_layers: int = 1,
        dropout: float = 0,
        bidirectional: bool = True,
    ):
        """
        Args:
            n_channels:       Number of EEG regions (default 4).
            n_classes:        Number of output classes.
            cnn_hidden_dims:  Channel progression for CNNHybrid.
            n_freq:           Frequency bins — used as LSTM input size.
            lstm_hidden_size: Hidden size per LSTM.
            n_layers:         Number of LSTM layers.
            dropout:          Dropout rate (shared across CNN and LSTMs).
            bidirectional:    Whether LSTMs are bidirectional.
        """
        super().__init__()

        self.cnn = CNNHybrid(n_channels=n_channels, hidden_dims=cnn_hidden_dims, dropout=dropout)

        # one LSTM per spectrogram channel, each reading (time, freq) sequences
        lstm_kwargs = dict(n_features=n_freq, hidden_size=lstm_hidden_size,
                           dropout=dropout, bidirectional=bidirectional, num_layers=n_layers)
        self.lstm1 = LSTMHybrid(**lstm_kwargs)
        self.lstm2 = LSTMHybrid(**lstm_kwargs)
        self.lstm3 = LSTMHybrid(**lstm_kwargs)
        self.lstm4 = LSTMHybrid(**lstm_kwargs)

        bidir       = 2 if bidirectional else 1
        output_dims = cnn_hidden_dims[-1] + 4 * bidir * lstm_hidden_size
        self.head   = nn.Sequential(
            nn.Linear(output_dims, 64),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(x)

        # each LSTM sees one channel: (batch, n_time, n_freq)
        l1 = self.lstm1(x[:, 0, :, :].permute(0, 2, 1))
        l2 = self.lstm2(x[:, 1, :, :].permute(0, 2, 1))
        l3 = self.lstm3(x[:, 2, :, :].permute(0, 2, 1))
        l4 = self.lstm4(x[:, 3, :, :].permute(0, 2, 1))

        embedding = torch.cat([cnn_out, l1, l2, l3, l4], dim=1)
        logits    = self.head(embedding)
        return nn.functional.log_softmax(logits, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Sequential (CNN → LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class CNNSequential(nn.Module):
    """CNN backbone for the sequential model.

    Unlike CNNHybrid, there is no MaxPool between conv blocks — spatial dims are
    preserved along the time axis. The frequency dimension is collapsed to 1 via
    AdaptiveAvgPool2d((1, None)), leaving a per-timestep feature vector for the
    downstream LSTM.

    Input shape:  (batch, n_channels, n_freq, n_time)
    Output shape: (batch, hidden_dims[-1], 1, n_time)
    """

    def __init__(
        self,
        n_channels: int,
        hidden_dims: list = [16, 32, 64],
        dropout: float = 0.0,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.n_channels = n_channels

        dims = [n_channels] + hidden_dims
        self.blocks = nn.Sequential(*[
            ConvBlock(dims[i], dims[i + 1], kernel=kernel_size, padding=padding, dropout=dropout)
            for i in range(len(dims) - 1)
        ])

        # collapse frequency dimension to 1, keep time dimension intact
        self.pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)  # (batch, hidden_dims[-1], n_freq, n_time)
        x = self.pool(x)    # (batch, hidden_dims[-1], 1, n_time)
        return x


class LSTMSequential(nn.Module):
    """Bidirectional LSTM encoder for the sequential model.

    Identical in structure to LSTMHybrid — takes a time sequence and returns
    the last hidden state.

    Input shape:  (batch, seq_len, n_features)
    Output shape: (batch, hidden_size * num_directions)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            return torch.cat([h_n[-2], h_n[-1]], dim=1)
        return h_n[-1]


class SequentialModel(nn.Module):
    """Sequential CNN → LSTM model for EEG spectrogram classification.

    The CNNSequential backbone extracts per-timestep feature vectors (collapsing
    the frequency dimension while preserving time), then a single bidirectional
    LSTM reads the temporal sequence and produces a fixed-size embedding for
    classification.

    Input shape:  (batch, n_channels, n_freq, n_time)
    Output:       log-softmax over n_classes
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        cnn_hidden_dims: list = [16, 32, 64],
        lstm_hidden_size: int = 32,
        n_layers: int = 1,
        dropout: float = 0,
        bidirectional: bool = True,
    ):
        """
        Args:
            n_channels:       Number of input channels (EEG regions).
            n_classes:        Number of output classes.
            cnn_hidden_dims:  Channel progression for CNNSequential.
            lstm_hidden_size: LSTM hidden size.
            n_layers:         Number of LSTM layers.
            dropout:          Dropout rate (shared across CNN and LSTM).
            bidirectional:    Whether the LSTM is bidirectional.
        """
        super().__init__()

        self.cnn = CNNSequential(n_channels=n_channels, hidden_dims=cnn_hidden_dims, dropout=dropout)
        self.lstm = LSTMSequential(
            n_features=cnn_hidden_dims[-1], hidden_size=lstm_hidden_size,
            dropout=dropout, bidirectional=bidirectional, num_layers=n_layers,
        )

        bidir       = 2 if bidirectional else 1
        output_dims = bidir * lstm_hidden_size
        self.head   = nn.Sequential(
            nn.Linear(output_dims, 64),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)          # (batch, hidden_dims[-1], 1, n_time)
        x = x.squeeze(2)         # (batch, hidden_dims[-1], n_time)
        x = x.permute(0, 2, 1)  # (batch, n_time, hidden_dims[-1]) — LSTM expects (batch, seq, features)
        x = self.lstm(x)         # (batch, hidden_size * num_directions)
        x = self.head(x)         # (batch, n_classes)
        return nn.functional.log_softmax(x, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Optuna model
# ─────────────────────────────────────────────────────────────────────────────

class OptunaModel(nn.Module):
    """Configurable VGG-style CNN used during hyperparameter search.

    Builds a variable number of two-conv-block stages from a flat list of
    channel widths (`hidden_dims`). Each stage consists of two ConvBlocks
    followed by MaxPool and Dropout2d. A global average pool + two-layer MLP
    head produces the final class logits.

    This is the architecture selected by the Optuna study and used for the
    final trained model.

    Input shape:  (batch, n_channels, n_freq, n_time)
    Output:       log-softmax over n_classes
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden_dims: list,
        kernel_size: int,
        dropout: float,
    ):
        """
        Args:
            n_channels:  Number of input channels (EEG regions, typically 4).
            n_classes:   Number of output classes (typically 6).
            hidden_dims: Flat list of channel widths for all conv layers.
                         Must have an even length; pairs are grouped into blocks.
                         Example: [117, 52, 60, 64, 103, 226] → 3 blocks.
            kernel_size: Convolution kernel size (shared across all blocks).
            dropout:     Dropout probability applied after each block (Dropout2d)
                         and before the final linear layer.
        """
        super().__init__()

        blocks = []
        dims   = [n_channels] + hidden_dims

        # build VGG-like blocks: pairs of ConvBlocks + MaxPool + Dropout2d
        for i in range(0, len(dims) - 2, 2):
            blocks.append(nn.Sequential(
                ConvBlock(
                    in_channels=dims[i],     out_channels=dims[i + 1],
                    dropout=0,               kernel=kernel_size,
                    padding=kernel_size // 2, stride=1,
                ),
                ConvBlock(
                    in_channels=dims[i + 1], out_channels=dims[i + 2],
                    dropout=0,               kernel=kernel_size,
                    padding=kernel_size // 2, stride=1,
                ),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout2d(dropout),
            ))

        self.backbone = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dims[-1], 64),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return nn.functional.log_softmax(x, dim=1)
