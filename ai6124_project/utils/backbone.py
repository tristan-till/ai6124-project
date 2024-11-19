import torch.optim as optim
import torch.nn as nn

import ai6124_project.utils.params as params  

from ai6124_project.classes.custom_loss import HitRateLoss
from ai6124_project.classes.gru_lstm_attention import GRULSTMAttentionModel
from ai6124_project.classes.lstm import LSTMModel

def get_backbone(num_features, device):
    model = GRULSTMAttentionModel(input_size=num_features).to(device)
    criterion = nn.MSELoss()
    # criterion = HitRateLoss()
    optimizer = optim.Adam(model.parameters(), lr=params.LEARNING_RATE, weight_decay=params.WEIGHT_DECAY)
    return model, criterion, optimizer

def get_baseline(num_features, device):
    model = LSTMModel(input_size=num_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params.LEARNING_RATE, weight_decay=params.WEIGHT_DECAY)
    return model, criterion, optimizer