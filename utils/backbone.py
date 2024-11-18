import torch
import torch.nn as nn

import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = nn.Linear(
            input_dim, attention_dim
        )  # Linear layer to get attention scores
        self.attention_vector = nn.Linear(
            attention_dim, 1
        )  # Output attention weights as a single value per time step

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]

        # Compute attention scores
        attn_scores = torch.tanh(
            self.attention_fc(x)
        )  # [batch_size, sequence_length, attention_dim]
        attn_weights = self.attention_vector(
            attn_scores
        )  # [batch_size, sequence_length, 1]

        # Apply softmax to get normalized attention weights along sequence dimension
        attn_weights = F.softmax(
            attn_weights, dim=1
        )  # [batch_size, sequence_length, 1]

        # Apply attention weights to input
        context = torch.sum(
            attn_weights * x, dim=1
        )  # Weighted sum over sequence length

        return context, attn_weights.squeeze(-1)


class GRULSTMAttentionModel(nn.Module):
    def __init__(
        self, input_size, gru_size, lstm_size, attention_size, num_layers=2, dropout=0.2
    ):
        super(GRULSTMAttentionModel, self).__init__()
        self.num_layers = num_layers

        # Batch normalization for input
        self.input_bn = nn.BatchNorm1d(input_size)

        # GRU Layer
        self.gru = nn.GRU(
            input_size,
            gru_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # LSTM Layer
        self.lstm = nn.LSTM(
            gru_size,
            lstm_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention Layer (Ensure AttentionLayer outputs context of attention_size)
        self.attention = AttentionLayer(
            lstm_size, attention_size
        )  # Pass lstm_size as input

        # Output Layer with batch normalization
        self.fc = nn.Sequential(
            nn.BatchNorm1d(lstm_size),
            nn.Linear(lstm_size, lstm_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(lstm_size // 2),
            nn.Linear(lstm_size // 2, 1),
        )

        for layer in self.children():
            if hasattr(layer, "weight") and len(layer.weight.shape) > 1:
                torch.nn.init.xavier_uniform_(layer.weight)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        x = x.transpose(1, 2)  # Change to [batch_size, input_size, sequence_length]
        x = self.input_bn(x)
        x = x.transpose(
            1, 2
        )  # Change back to [batch_size, sequence_length, input_size]

        # GRU
        gru_out, _ = self.gru(x)  # [batch_size, sequence_length, gru_size]

        # LSTM
        lstm_out, _ = self.lstm(gru_out)  # [batch_size, sequence_length, lstm_size]

        # Attention
        context, _ = self.attention(lstm_out)  # context: [batch_size, attention_size]
        # Output
        output = self.fc(context)  # [batch_size, 1]
        return output
