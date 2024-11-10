import numpy as np

import torch

import utils.data as data

from utils.backbone import GRULSTMAttentionModel
from utils.controller import EvolutionController

def get_target(stock, start_date, end_date=None, phase=1):
    target_data = data.get_data_with_signal(stock, start_date, end_date)
    close_d = target_data[f'{stock}_close_d']
    future_close_d = close_d.shift(-phase)
    future_close_d = future_close_d.dropna()
    x = future_close_d.values
    return x

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    start_date = "2020-01-01"
    end_date = "2024-11-01"
    target = "XOM"
    phase=30

    df = data.get_data_with_signal(target, start_date, end_date)
    prices = np.array(df['XOM_Close'].values[-100:])

    f1 = df['XOM_close_d'].values[-100:]
    f2 = df['XOM_macd'].values[-100:]
    f3 = df['XOM_Volume'].values[-100:]
    f4 = df['XOM_roc'].values[-100:]
    target_y = get_target(target, start_date, end_date, phase=phase)[-100:]
    X = np.stack([f1, f2, f3, f4, target_y], axis=1)
    

    # num_features = 96 

    # gru_size=512
    # lstm_size=256
    # attention_size=128
    # num_layers = 4
    # predictor = GRULSTMAttentionModel(
    #     input_size=num_features, 
    #     gru_size=gru_size, 
    #     lstm_size=lstm_size, 
    #     attention_size=attention_size,
    #     num_layers=num_layers
    #     ).to(device)
    controller = EvolutionController(10, 10)
    controller.train(X, prices)

if __name__ == '__main__':
    main()