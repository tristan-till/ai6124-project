import torch
import torch.nn as nn

import utils.params as params 

class GRULSTMAttentionModel(nn.Module):
    def __init__(self, input_size, 
                 gru_size=params.GRU_SIZE, gru_layers=params.GRU_LAYERS, 
                 lstm_size=params.LSTM_SIZE, lstm_layers=params.LSTM_LAYERS,
                  attention_size=params.ATTENTION_SIZE, 
                  dropout=params.DROPOUT):
        super(GRULSTMAttentionModel, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_size)
        
        self.gru = nn.GRU(input_size, gru_size, num_layers=gru_layers, 
                          batch_first=True, dropout=dropout if gru_layers > 1 else 0)
        
        self.lstm = nn.LSTM(gru_size, lstm_size, num_layers=lstm_layers, 
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        
        self.attention = AttentionLayer(lstm_size, attention_size)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(lstm_size),
            nn.Linear(lstm_size, lstm_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(lstm_size // 2),
            nn.Linear(lstm_size // 2, 1)
        )

    def load(self, path):
         self.load_state_dict(torch.load(path, weights_only=True))
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)
        
        gru_out, _ = self.gru(x)        
        lstm_out, _ = self.lstm(gru_out)
        context, _ = self.attention(lstm_out)

        output = self.fc(context)
        return output
