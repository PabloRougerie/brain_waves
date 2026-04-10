import torch
from torch import nn


#=================
# BASELINE ELEMENTS
#=================


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU block, with optional Dropout2d."""

    def __init__(self, in_channels,
                 out_channels,
                 stride= 1,
                 kernel= 3,
                 padding= 1,
                 dropout = 0.0):

        super().__init__()

        layers = [
            nn.Conv2d(in_channels= in_channels,
                      out_channels= out_channels,
                      stride= stride,
                      padding= padding,
                      kernel_size= kernel,
                      bias= False),  # bias redundant with BatchNorm
            nn.BatchNorm2d(num_features= out_channels),
            nn.ReLU()
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self,x):
        return self.block(x)


class BaselineModel(nn.Module):
    """
    Hardcoded 2-block CNN baseline for EEG spectrogram classification.

    Input shape: (batch, n_channels, n_freq, n_time)
    Output: log-softmax over n_classes
    """

    def __init__(self, n_classes, n_channels):

        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.block1 = nn.Sequential(
            ConvBlock(in_channels= self.n_channels,
                      out_channels= 32),
            ConvBlock(in_channels= 32,
                      out_channels= 64),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1),
            nn.Dropout(0.3)
        )

        self.block2 = nn.Sequential(
            ConvBlock(in_channels= 64,
                      out_channels= 128),
            ConvBlock(in_channels= 128,
                      out_channels= 256),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1),
            nn.Dropout(0.3)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.0),
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.Linear(64, self.n_classes)
        )

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)

        return nn.functional.log_softmax(x, dim= 1)


#=================
# HYBRID MODEL
#=================


class CNNHybrid(nn.Module):
    """
    Configurable CNN backbone for the hybrid model.

    Extracts a spatial embedding from the full spectrogram (all channels).
    Output shape: (batch, hidden_dims[-1])
    """

    def __init__(self, n_channels,
                 hidden_dims: list = [16,32,64],
                 dropout= 0.3,
                 kernel_size= 3,
                 stride= 1,
                 padding= 1):

        super().__init__()
        self.n_channels = n_channels

        #build conv blocks dynamically from hidden_dims
        dims = [n_channels] + hidden_dims
        self.blocks = nn.Sequential(*[
            ConvBlock(dims[i], dims[i+1], kernel=kernel_size, padding=padding, dropout=dropout)
            for i in range(len(dims) - 1)
        ])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )

    def forward(self,x):
        x = self.blocks(x)
        x = self.head(x)
        return x  # shape: (batch, hidden_dims[-1])


class LSTMHybrid(nn.Module):
    """
    LSTM encoder for spectrogram.

    Takes a spectrogram and returns the last hidden state,
    concatenating both directions if bidirectional.

    Input shape:  (batch, seq_len, n_features)  e.g. (batch, 25, 100)
    Output shape: (batch, hidden_size * num_directions)
    """

    def __init__(self, n_features, hidden_size, num_layers= 1, dropout= 0.3, bidirectional= True):
        super().__init__()
        self.bidirectional = bidirectional

        # note: LSTM dropout only applies between layers — no effect with num_layers=1
        self.lstm = nn.LSTM(input_size= n_features,
                            hidden_size= hidden_size,
                            num_layers= num_layers,
                            batch_first= True,
                            dropout= dropout if num_layers > 1 else 0.0,
                            bidirectional= bidirectional)

    def forward(self,x):
        # x: (batch, seq_len, n_features)
        # h_n: (num_layers * num_directions, batch, hidden_size) — last hidden state
        # c_n: last cell state (not used)
        _, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            #concat last forward and last backward hidden states
            last = torch.cat([h_n[-2], h_n[-1]], dim= 1)
        else:
            last = h_n[-1]

        return last  # shape: (batch, hidden_size * num_directions)


class HybridModel(nn.Module):
    """
    CNN + 4 parallel LSTMs hybrid model for EEG spectrogram classification.

    Architecture:
    - CNNHybrid processes the full spectrogram (all channels) → spatial embedding
    - 4 LSTMHybrid process each channel independently along the time axis → temporal embeddings
    - All embeddings are concatenated and passed to a classification head

    Input shape:  (batch, n_channels=4, n_freq, n_time)
    Output:       log-softmax over n_classes
    """

    def __init__(self,
                    n_channels,
                    n_classes,
                    cnn_hidden_dims= [16,32,64],
                    n_freq= 100,
                    lstm_hidden_size= 32,
                    dropout= 0,
                    bidirectional = True):
        """
        Args:
            n_channels:       number of EEG regions (default: 4)
            n_classes:        number of output classes
            cnn_hidden_dims:  channel progression for CNNHybrid
            n_freq:           number of frequency bins — used as LSTM input size
            lstm_hidden_size: hidden size per LSTM
            dropout:          dropout rate (shared across CNN and LSTMs)
            bidirectional:    whether LSTMs are bidirectional
        """
        super().__init__()

        self.cnn = CNNHybrid(n_channels= n_channels,
                                hidden_dims= cnn_hidden_dims,
                                dropout= dropout)

        #one LSTM per spectrogram channel, each reading (time, freq) sequences
        self.lstm1 = LSTMHybrid(n_features= n_freq, hidden_size= lstm_hidden_size,
                                dropout= dropout, bidirectional= bidirectional)
        self.lstm2 = LSTMHybrid(n_features= n_freq, hidden_size= lstm_hidden_size,
                                dropout= dropout, bidirectional= bidirectional)
        self.lstm3 = LSTMHybrid(n_features= n_freq, hidden_size= lstm_hidden_size,
                                dropout= dropout, bidirectional= bidirectional)
        self.lstm4 = LSTMHybrid(n_features= n_freq, hidden_size= lstm_hidden_size,
                                dropout= dropout, bidirectional= bidirectional)

        #output dim: cnn embedding + 4 lstm embeddings (each hidden_size * num_directions)
        bidir = 2 if bidirectional else 1
        output_dims = cnn_hidden_dims[-1] + 4 * bidir * lstm_hidden_size
        self.head = nn.Sequential(
            nn.Linear(output_dims, 64),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self,x):
        #x: (batch, channels, n_freq, n_time)

        #CNN sees full spectrogram (all channels)
        cnn_output = self.cnn(x)

        #each LSTM sees one channel, transposed to (batch, n_time, n_freq)
        lstm1_output = self.lstm1(x[:,0,:,:].permute(0,2,1))
        lstm2_output = self.lstm2(x[:,1,:,:].permute(0,2,1))
        lstm3_output = self.lstm3(x[:,2,:,:].permute(0,2,1))
        lstm4_output = self.lstm4(x[:,3,:,:].permute(0,2,1))

        #concatenate all embeddings along feature dimension
        final_embedding = torch.cat([cnn_output, lstm1_output, lstm2_output, lstm3_output, lstm4_output], dim= 1)

        logits = self.head(final_embedding)

        return nn.functional.log_softmax(logits, dim= 1)
