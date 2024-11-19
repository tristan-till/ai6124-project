import torch.nn as nn

import ai6124_project.utils.params as params

class LSTMModel(nn.Module):
    def __init__(self, input_size=30, hidden_size=params.BASELINE_HIDDEN_SIZE, num_layers=params.BASELINE_HIDDEN_LAYERS):
        super(LSTMModel, self).__init__()        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output