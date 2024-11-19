import torch
import torch.nn as nn
import torch.nn.functional as F

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