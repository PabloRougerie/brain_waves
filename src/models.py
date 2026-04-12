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
                 dropout= 0.0,
                 kernel_size= 3,
                 stride= 1,
                 padding= 1):

        super().__init__()
        self.n_channels = n_channels

        #build conv blocks dynamically from hidden_dims
        dims = [n_channels] + hidden_dims
        self.blocks = nn.Sequential(*[
        nn.Sequential(
        ConvBlock(dims[i], dims[i+1], kernel=kernel_size, padding=padding, dropout=dropout),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    for i in range(len(dims)-1)
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
                    n_layers= 1,
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
                                dropout= dropout, bidirectional= bidirectional, num_layers= n_layers)
        self.lstm2 = LSTMHybrid(n_features= n_freq, hidden_size= lstm_hidden_size,
                                dropout= dropout, bidirectional= bidirectional, num_layers= n_layers)
        self.lstm3 = LSTMHybrid(n_features= n_freq, hidden_size= lstm_hidden_size,
                                dropout= dropout, bidirectional= bidirectional, num_layers= n_layers)
        self.lstm4 = LSTMHybrid(n_features= n_freq, hidden_size= lstm_hidden_size,
                                dropout= dropout, bidirectional= bidirectional, num_layers= n_layers)

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


#=================
# SEQUENTIAL MODEL
# ================


class CNNSequential(nn.Module):
    """
    CNN backbone for the Sequential model.

    Unlike CNNHybrid, there is no MaxPool between blocks — spatial dims are preserved
    along the time axis. The frequency dimension is collapsed to 1 via AdaptiveAvgPool2d,
    leaving a per-timestep feature vector for the downstream LSTM.

    Input shape:  (batch, n_channels, n_freq, n_time)
    Output shape: (batch, hidden_dims[-1], 1, n_time)
    """

    def __init__(self, n_channels,
                 hidden_dims: list = [16,32,64],
                 dropout= 0.0,
                 kernel_size= 3,
                 stride= 1,
                 padding= 1):

        super().__init__()
        self.n_channels = n_channels

        #build conv blocks dynamically from hidden_dims
        dims = [n_channels] + hidden_dims
        self.blocks = nn.Sequential(*[
            ConvBlock(dims[i], dims[i+1], kernel=kernel_size, padding=padding, dropout=dropout)
            for i in range(len(dims)-1)
        ])

        #collapse frequency dimension to 1, keep time dimension intact
        self.pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        # x: (batch, n_channels, n_freq, n_time)
        x = self.blocks(x)     # → (batch, hidden_dims[-1], n_freq, n_time)
        x = self.pool(x)       # → (batch, hidden_dims[-1], 1, n_time)
        return x


class LSTMSequential(nn.Module):
    """
    LSTM encoder for the Sequential model.

    Identical in structure to LSTMHybrid — takes a sequence and returns
    the last hidden state.

    Input shape:  (batch, seq_len, n_features)
    Output shape: (batch, hidden_size * num_directions)
    """

    def __init__(self, n_features, hidden_size, num_layers= 1, dropout= 0.0, bidirectional= True):

        super().__init__()
        self.bidirectional = bidirectional

        # note: LSTM dropout only applies between layers — no effect with num_layers=1
        self.lstm = nn.LSTM(input_size= n_features,
                            hidden_size= hidden_size,
                            num_layers= num_layers,
                            batch_first= True,
                            dropout= dropout if num_layers > 1 else 0.0,
                            bidirectional= bidirectional)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        # h_n: (num_layers * num_directions, batch, hidden_size) — last hidden state
        # c_n: last cell state (not used)
        _, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            #concat last forward and last backward hidden states
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last = h_n[-1]

        return last  # shape: (batch, hidden_size * num_directions)


class SequentialModel(nn.Module):
    """
    Sequential CNN → LSTM model

    The CNN extracts per-timestep feature vectors (collapsing frequency),
    then a single LSTM reads the sequence and produces a fixed-size embedding
    for classification.

    Input shape:  (batch, n_channels, n_freq, n_time)
    Output:       log-softmax over n_classes
    """

    def __init__(self, n_channels,
                    n_classes,
                    cnn_hidden_dims= [16,32,64],
                    lstm_hidden_size= 32,
                    n_layers= 1,
                    dropout= 0,
                    bidirectional= True):
        """
        Args:
            n_channels:       number of input channels (EEG regions)
            n_classes:        number of output classes
            cnn_hidden_dims:  channel progression for CNNSequential
            lstm_hidden_size: LSTM hidden size
            n_layers:         number of LSTM layers
            dropout:          dropout rate (shared across CNN and LSTM)
            bidirectional:    whether the LSTM is bidirectional
        """
        super().__init__()

        #CNN extracts (batch, hidden_dims[-1], 1, n_time) feature maps
        self.cnn = CNNSequential(n_channels=n_channels,
                                 hidden_dims=cnn_hidden_dims,
                                 dropout=dropout)

        #single LSTM reads the time sequence of CNN feature vectors
        self.lstm = LSTMSequential(n_features=cnn_hidden_dims[-1], hidden_size=lstm_hidden_size,
                                   dropout=dropout, bidirectional=bidirectional, num_layers=n_layers)

        #output dim: lstm last hidden state (hidden_size * num_directions)
        bidir = 2 if bidirectional else 1
        output_dims = bidir * lstm_hidden_size
        self.head = nn.Sequential(
            nn.Linear(output_dims, 64),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (batch, n_channels, n_freq, n_time)
        x = self.cnn(x)              # → (batch, hidden_dims[-1], 1, n_time)
        x = x.squeeze(2)             # → (batch, hidden_dims[-1], n_time)  — squeeze freq dim only
        x = x.permute(0, 2, 1)      # → (batch, n_time, hidden_dims[-1])  — LSTM expects (batch, seq, features)
        x = self.lstm(x)             # → (batch, hidden_size * num_directions)
        x = self.head(x)             # → (batch, n_classes)
        return nn.functional.log_softmax(x, dim=1)



#==============
# Optuna model
#==============

class OptunaModel(nn.Module):

    def __init__(self, n_channels, n_classes, hidden_dims, kernel_size, dropout):

        super().__init__()

        blocks = []

        dims = [n_channels] + hidden_dims
        #construct several VGG-like blocks
        for i in range(0,len(dims) - 2, 2):

            blocks.append(
                nn.Sequential(ConvBlock(in_channels= dims[i], out_channels= dims[i+1],
                                        dropout= 0,
                                        kernel= kernel_size,
                                        padding= kernel_size // 2,
                                        stride= 1),
                              ConvBlock(in_channels= dims[i+1], out_channels= dims[i+2],
                                        dropout= 0,
                                        kernel= kernel_size,
                                        padding= kernel_size // 2,
                                        stride= 1),
                              nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1),
                              nn.Dropout2d(dropout)
                              ) )

        self.backbone = nn.Sequential(*blocks)

        self.head = nn.Sequential(
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten(),
                        nn.Linear(dims[-1], 64),
                        nn.Dropout(dropout),
                        nn.Linear(64, n_classes)
        )


    def forward(self,   x):

        x = self.backbone(x)
        x = self.head(x)
        x = nn.functional.log_softmax(x, dim= 1)
        return x
