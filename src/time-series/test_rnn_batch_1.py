#ライブラリのインポート
import numpy as np
import torch
import torch.nn as nn
from train_rnn_batch_1 import evaluate, predict, src_len, TimeSeriesDataset, DataLoader, My_rnn_net, My_lstm_net, My_gru_net, exp_dir_path, fix_seed, Transformer, TokenEmbedding, PositionalEncoding

#ランダムシードの設定
# fix_seed = 42
np.random.seed(fix_seed)
torch.manual_seed(fix_seed)

# src_len = 30 * 5
# tgt_len = 1
# batch_size = 13
# exp_dir_path = 'drink_20230909/'

# data_preprocess_padding()

best_model = torch.load(exp_dir_path+'best_model_weight.pth')

data_set = TimeSeriesDataset(src_len, best_model)

train_len = int(len(data_set)*0.9)
valid_len = len(data_set) - train_len

train_dataset, valid_dataset = torch.utils.data.random_split(
    data_set, 
    [train_len, valid_len]
)

# #データをバッチごとに分けて出力できるDataLoaderを使用
train_loader = DataLoader(train_dataset,
                            batch_size=1, 
                            shuffle=True
                        )
valid_loader = DataLoader(valid_dataset,
                            batch_size=1, 
                            shuffle=False
                        )
# test_loader = DataLoader(valid_dataset,
#                             batch_size=1, 
#                             shuffle=False
#                         )

criterion = nn.MSELoss()

predict(model=best_model, data_loader=valid_loader)