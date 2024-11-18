import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import utils.params as params

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = nn.Linear(input_dim, attention_dim)
        self.attention_vector = nn.Linear(attention_dim, 1)

    def forward(self, x):
        attn_scores = torch.tanh(self.attention_fc(x))
        attn_weights = self.attention_vector(attn_scores)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * x, dim=1)
        return context, attn_weights.squeeze(-1)  
    
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


from classes.custom_loss import HitRateLoss
def get_backbone(num_features, device):
    model = GRULSTMAttentionModel(input_size=num_features).to(device)
    criterion = nn.MSELoss()
    # criterion = HitRateLoss()
    optimizer = optim.Adam(model.parameters(), lr=params.LEARNING_RATE, weight_decay=params.WEIGHT_DECAY)
    return model, criterion, optimizer