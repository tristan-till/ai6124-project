import torch
import torch.nn as nn
import torch.optim as optim

import utils.params as params

class LSTMModel(nn.Module):
    def __init__(self, input_size=30, hidden_size=params.BASELINE_HIDDEN_SIZE, num_layers=params.BASELINE_HIDDEN_LAYERS):
        super(LSTMModel, self).__init__()
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Define fully connected layer to produce a single output
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        last_hidden_state = lstm_out[:, -1, :]
        
        output = self.fc(last_hidden_state)
        
        return output
    
def get_baseline(num_features, device):
    model = LSTMModel(input_size=num_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params.LEARNING_RATE, weight_decay=params.WEIGHT_DECAY)
    return model, criterion, optimizer


if __name__ == '__main__':
    input_tensor = torch.randn(8, 96, 30)
    model = LSTMModel()
    output = model(input_tensor)
    print(output.shape)
