import torch
import torch.nn as nn
import torch.optim as optim

import utils.params as params
from ai6124_project.classes.lstm import LSTMModel
    
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
