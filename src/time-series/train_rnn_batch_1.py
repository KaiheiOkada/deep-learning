# without depth, only person, [kpt,2] to [kpt,node_num] to [512], only kpt is predicted by predicted src
#ライブラリのインポート
import math
import numpy as np
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import cv2
import json
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

import torch
from torch import nn,optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torchvision import transforms
# from torchinfo import summary #torchinfoはニューラルネットの中身を見れるのでおすすめ
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random

#ランダムシードの設定
fix_seed = 1
np.random.seed(fix_seed)
torch.manual_seed(fix_seed)

#デバイスの設定
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

ss = StandardScaler(with_mean=True, with_std=True)

kpt_n = 17
kpt_n_3 = kpt_n * 2
# obj_n = 1
input_n = kpt_n
input_n_3 = input_n * 2
img_width = 640
img_height = 480

d_input = input_n_3
d_output = input_n_3
d_model = 512
nhead = 8
num_encoder_layers = 1
# num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
sampling_interval = 1
Hz = int(30 / sampling_interval)
# batch_size = 13
epochs = 300
best_loss = float('Inf')
best_model = None
node_num1 = 16
predict_kpt_id = 9
data_num = 0
padding_frames = Hz * 20
# predict_kpt_xy_id = 1

original_data = None

exp_dir_path = 'drink_20230909/'
os.makedirs(exp_dir_path,exist_ok=True)

src_len = 30
tgt_len = 30
src_tgt_len = src_len + tgt_len
slide_len = 30

#位置エンコーディングの定義
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

#モデルに入力するために次元を拡張する
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Linear(c_in, d_model) 

    def forward(self, x):
        x = self.tokenConv(x)
        # print(x.shape)
        return x


class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, d_model, d_input, d_output):
        
        super(Transformer, self).__init__()
        

        #エンべディングの定義
        self.token_embedding_src = TokenEmbedding(kpt_n_3, d_model)
        self.token_embedding_tgt = TokenEmbedding(kpt_n_3, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        #エンコーダの定義
        encoder_layer = TransformerEncoderLayer(d_model=d_model, 
                                                nhead=nhead, 
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation='gelu'
                                               )
        encoder_norm = LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, 
                                                      num_layers=num_encoder_layers,
                                                      norm=encoder_norm
                                                     )
        
        #出力層の定義
        self.output = nn.Linear(d_model, kpt_n_3)
        

    def forward(self, src, mask_src):
        #mask_src, mask_tgtはセルフアテンションの際に未来のデータにアテンションを向けないためのマスク
        
        # embedding_src = self.positional_encoding(self.token_embedding_src(src))
        embedding_src = (self.token_embedding_src(src))
        outs = self.transformer_encoder(embedding_src, mask_src)
        
        output = self.output(outs)
        return output

    def encode(self, src, mask_src):
        # return self.transformer_encoder(self.positional_encoding(self.token_embedding_src(src)), mask_src)
        return self.transformer_encoder(self.token_embedding_src(src), mask_src)

def create_mask(src, tgt):
    
    seq_len_src = src.shape[1]
    seq_len_tgt = tgt.shape[1]

    mask_tgt = generate_square_subsequent_mask(seq_len_tgt).to(device)
    mask_src = generate_square_subsequent_mask(seq_len_src).to(device)

    return mask_src, mask_tgt

def generate_square_subsequent_mask(seq_len):
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask

class My_rnn_net(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(My_rnn_net, self).__init__()

        self.input_size = input_size #入力データ(x)
        self.hidden_dim = hidden_dim #隠れ層データ(hidden)
        self.n_layers = n_layers #RNNを「上方向に」何層重ねるか？の設定 ※横方向ではない
        """
        PyTorchのRNNユニット。batch_first=Trueでバッチサイズを最初にくるように設定
        また、先程示した図ではRNN_cellが複数あったがここではRNNが1個しかない。
　　　　 つまりこの「nn.RNN」は複数のRNN_cellをひとまとめにしたものである。
　　　　 ※シーケンシャルデータと初期値のhiddenだけ入れてあげれば内部で勝手に計算してくれる
　　　　 ※出力部は各時刻毎に出力されるが、下で述べているように最後の時刻しか使用しない
        """
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size) #全結合層でhiddenからの出力を1個にする

    def forward(self, x):
        #h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        #y_rnn, h = self.rnn(x, h0)
        # 5次元テンソルを3次元テンソルに変換
        # x = x.view(-1, src_len, d_input)

        y_rnn, h = self.rnn(x, None) #hidden部分はコメントアウトした↑2行と同じ意味になっている。
        y = self.fc(y_rnn[:, -1, :]) #最後の時刻の出力だけを使用するので「-1」としている

        return y

class My_lstm_net(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(My_lstm_net, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 

        return out

class My_gru_net(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(My_gru_net, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, hn = self.gru(x, h0.detach())

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 

        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, src_len, model):
        #学習期間と予測期間の設定
        # self.pred_len = pred_len
        # self.sliding_window_step = slide_len
        # self.batch_size = batch_size
        self.video_paths = glob.glob('../yolov7/drink_20230909/exp_*/keypoints/keypoints.npy')
        self.len = len(self.video_paths)
        self.model = model
        self.src_len = src_len
        self.tgt_len = tgt_len

    def __getitem__(self, index):

        # Load depth video
        keypoints_video = np.load(self.video_paths[index])
        # print(self.video_paths[index])
        # bottle_video = np.load('../yolov7/drink_front/exp_{:03}/bottle.npy'.format(k))
        
        keypoints = [elem for i, elem in enumerate(keypoints_video) if (i+1) % 3 != 0]
        self.fn = int(len(keypoints)/51)
        self.n_sample = self.fn - self.src_len - 1 #学習予測サンプルはt=10~99なので89個

        #シーケンシャルデータの固まり数、シーケンシャルデータの長さ、RNN_cellへの入力次元(1次元)に形を成形
        self.input_data = np.zeros((self.n_sample, self.src_len, d_input)) #シーケンシャルデータを格納する箱を用意(入力)
        self.correct_input_data = np.zeros((self.n_sample, d_input)) #シーケンシャルデータを格納する箱を用意(正解)

        keypoints_x = [elem for i, elem in enumerate(keypoints) if (i+1) % 2 != 0]
        keypoints_y = [elem for i, elem in enumerate(keypoints) if (i) % 2 != 0]
        # bottle_x = [elem for i, elem in enumerate(bottle_video) if (i+1) % 2 != 0]
        # bottle_y = [elem for i, elem in enumerate(bottle_video) if (i) % 2 != 0]

        data = []
        gray_part = 16
        # Process frame_color and frame_depth to obtain keypoints and depth values
        for i in range(self.fn):
            data_xy = []
            # keypoint of human
            i = sampling_interval*i
            for j in range(0,kpt_n):
                x = (keypoints_x[kpt_n * i + j])
                x = max(0, min(x, img_width))
                data_xy.append(x)

                y = (keypoints_y[kpt_n * i + j] - gray_part)
                y = max(0, min(y, img_height))
                data_xy.append(y)
            
            data.append(data_xy)

            # # Bottle
            # x = int(bottle_x[i])
            # if x == 0 and i != 0:
            #     pre_x = data[-input_n_3]
            #     data.append(pre_x)
            # else:
            #     data.append(x)

            # y = int(bottle_y[i])
            # if y == 0 and i != 0:
            #     pre_y = data[-input_n_3]
            #     data.append(pre_y)
            # else:
            #     data.append(y)

        data = np.array(data)
        data = data.reshape(self.fn, kpt_n_3)
        original_data = data
        np.save(exp_dir_path + 'original_data.npy', original_data)

        data = ss.fit_transform(data)

        src_frames = []
        tgt_frames = []
        
        if type(self.model).__name__ == "Transformer":
            for i in range(0, self.fn - self.src_len, slide_len):
                src = data[i:i+self.src_len, :]
                # print(src.shape)
                tgt = data[i+self.src_len : i+self.src_len + self.tgt_len, :]
                src_frames.append(src)
                tgt_frames.append(tgt)

            return src_frames, tgt_frames, data
        else:
            for i in range(self.n_sample):
                self.input_data[i] = data[i:i+self.src_len, :]
                self.correct_input_data[i] = data[i+self.src_len:i+self.src_len+1, :]
            # print(self.input_data.shape)
            # print(self.correct_input_data.shape)
            return self.input_data, self.correct_input_data
    
    def __len__(self):
        return self.len


def train(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = []
    if type(model).__name__ == "Transformer":
        for src_list, tgt_list, data in data_loader:
            data = data.float().to(device)
            input = data[:,:-1,:]
            out = data[:,1:,:]

            mask_src = create_mask(input, out)[0]

            output = model(
                src=input,
                mask_src=mask_src
            )

            optimizer.zero_grad()

            loss_1 = criterion(output[:, :, 10:21], out[:, :, 10:21])
            loss_2 = criterion(output[:, :, 0:10], out[:, :, 0:10])
            loss_3 = criterion(output[:, :, 21:], out[:, :, 21:])
            loss = 0.8 * loss_1 + 0.1 * loss_2 + 0.1 * loss_3

            loss.backward()
            total_loss.append(loss.cpu().detach())
            optimizer.step()
            # scheduler.step()
    else:
        for id, (x, t) in enumerate(data_loader):
            for i in range(len(x)):
                src = x[i].float().to(device)
                tgt = t[i].float().to(device)

                y = model(src)

                optimizer.zero_grad()
                loss = criterion(y, tgt)
                loss.backward()
                optimizer.step()

                total_loss.append(loss.item())

    return np.average(total_loss)


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = []
    if type(model).__name__ == "Transformer":
        for src_list, tgt_list, data in data_loader:
            data = data.float().to(device)
            input = data[:,:-1,:]
            out = data[:,1:,:]

            mask_src = create_mask(input, out)[0]

            output = model(
                src=input,
                mask_src=mask_src
            )

            loss_1 = criterion(output[:, :, 10:21], out[:, :, 10:21])
            loss_2 = criterion(output[:, :, 0:10], out[:, :, 0:10])
            loss_3 = criterion(output[:, :, 21:], out[:, :, 21:])
            loss = 0.8 * loss_1 + 0.1 * loss_2 + 0.1 * loss_3

            total_loss.append(loss.cpu().detach())
    else:
        for id, (x, t) in enumerate(data_loader):
            for i in range(len(x)):
                src = x[i].float().to(device)
                tgt = t[i].float().to(device)

                y = model(src)

                loss = criterion(y, tgt)
                total_loss.append(loss.item())

    return np.average(total_loss)


def predict(model, data_loader):
    model.eval()
    total_loss = []
    if type(model).__name__ == "Transformer":
        for id, (src_list, tgt_list, _) in enumerate(data_loader):
            # idごとにdirを作成
            id_dir_path = os.path.join(exp_dir_path, "{:03}/".format(id))
            os.makedirs(id_dir_path, exist_ok=True)

            src_list = src_list
            tgt_list = tgt_list
            predict_seq = []

            for i in range(len(src_list)):

                src = src_list[i]
                src = src.float().to(device)
                tgt = tgt_list[i]
                tgt = tgt.float().to(device)

                outputs = src[:, -1:, :]
            
                for i in range(tgt_len):
                
                    mask_src = (generate_square_subsequent_mask(src.size(1))).to(device)
                
                    output = model.encode(src, mask_src)
                    output = model.output(output)
                    src = torch.cat([src, output[:, -1:, :]], dim=1)
                    # print(output.shape)

                    outputs = torch.cat([outputs, output[:, -1:, :]], dim=1)
                
                # 最初の値予測値
                predict_seq.append(outputs[:, 1:, :].squeeze().cpu().detach().numpy())

            true = np.load(exp_dir_path + 'original_data.npy')
            pred = np.array(predict_seq)

            for i in range(0, pred.shape[0]):
                pred[i] = ss.inverse_transform(pred[i])

            print("output pdf")
            # plot
            for i in range(0, kpt_n_3):
                plt.clf()
                plt.plot(true[:, i], label='true')
                for j in range(pred.shape[0]):
                    plt.plot(np.arange(src_len + slide_len * j, src_len + slide_len * j + pred.shape[1]), pred[j, :, i])
                plt.legend()
                if i % 2 == 0:
                    plt.ylim(0, img_width)
                    plt.savefig(id_dir_path + 'keypoint_pos_' + str(int(i/2)) + '_x.png')
                elif i % 2 == 1:
                    plt.ylim(0, img_height)
                    plt.savefig(id_dir_path + 'keypoint_pos_' + str(int(i/2)) + '_y.png')

            # calc error
            error_4 = []
            for i in range(0, kpt_n_3):
                error_2 = []
                for j in range(pred.shape[0]):
                    t = true[src_len + slide_len * j:src_len + slide_len * j + pred.shape[1], i]
                    t_len = len(t)
                    p = pred[j, :, i]
                    p_len = len(p)
                    error_1 = 0.
                    if t_len < p_len:
                        error_1 = np.mean(np.abs(t - pred[j, :t_len, i]))
                    else:
                        error_1 = np.mean(np.abs(t[:p_len] - pred[j, :, i]))
                    error_2.append(error_1)
                error_3 = np.mean(error_2)
                error_4.append(error_3)
            error_mean = np.mean(error_4)
            print(error_mean)

            true = np.ravel(true)
            pred = np.ravel(pred)

            # plot_skeleton_3d(true,pred,id_dir_path,"true")
            # plot_skeleton_3d(true,pred,id_dir_path,"pred")
            # plot_skeleton_3d(true,pred,id_dir_path,"true_and_pred")

    else:
        for id, (x, t) in enumerate(data_loader):

            # 新しいフォルダを作成
            id_dir_path = os.path.join(exp_dir_path, "{:03}/".format(id))
            os.makedirs(id_dir_path, exist_ok=True)

            src_seq = []
            predict_seq = []

            for i in range(len(x)):
                src = x[i].float().to(device)
                tgt = t[i].float().to(device)
                    
                for j in range(tgt_len):
                
                    y = model(src)
                    y = y.unsqueeze(1)
                    src = torch.cat([src, y], dim=1)
                    predict_seq.append(y.cpu().detach().numpy())
                    src = src[:, 1:, :]
            
            # すべてのデータを numpy 配列に連結
            predict_seq = np.concatenate(predict_seq, axis=1)


            true = np.load(exp_dir_path + 'original_data.npy')
            pred = predict_seq

            # true = true.squeeze().cpu().detach().numpy()
            # pred = pred.squeeze().cpu().detach().numpy()

            # true = np.array(true)
            pred = np.array(pred)

            # # true = true.reshape(true.shape[1], 102)
            # pred = pred.reshape(pred.shape[0], pred.shape[2], pred.shape[3])

            # true = ss.inverse_transform(true)
            for i in range(0, pred.shape[0]):
                pred[i] = ss.inverse_transform(pred[i])

            print("output pdf")
            for i in range(0, kpt_n_3):
                plt.clf()
                plt.plot(true[:, i], label='true')
                for j in range(src_len, pred.shape[0], slide_len):
                    # 横軸方向にslide_len = 10ずらしながらプロットする。
                    plt.plot(np.arange(j, j+pred.shape[1]), pred[j, :, i])
                    # plt.plot(pred[j, :, i], label='pred')
                plt.legend()
                if i % 2 == 0:
                    plt.ylim(0, img_width)
                    plt.savefig(id_dir_path + 'keypoint_pos_' + str(int(i/2)) + '_x.png')
                elif i % 2 == 1:
                    plt.ylim(0, img_height)
                    plt.savefig(id_dir_path + 'keypoint_pos_' + str(int(i/2)) + '_y.png')

            # calc error
            error_4 = []
            for i in range(10, 21):
                error_2 = []
                for j in range(src_len, pred.shape[0], slide_len):
                    t = true[j : j+pred.shape[1], i]
                    t_len = len(t)
                    p = pred[j, :, i]
                    p_len = len(p)
                    error_1 = 0.
                    if t_len < p_len:
                        error_1 = np.mean(np.abs(t - pred[j, :t_len, i]))
                    else:
                        error_1 = np.mean(np.abs(t[:p_len] - pred[j, :, i]))
                    error_2.append(error_1)
                error_3 = np.mean(error_2)
                error_4.append(error_3)
            error_mean = np.mean(error_4)
            with open(exp_dir_path + "compare_model.txt", mode="a") as f:
                f.write(f"{type(model).__name__}, {id}, {str(error_mean)}\n")

            true = np.ravel(true)
            pred = np.ravel(pred)

            # plot_skeleton_3d(true,pred,id_dir_path,"true")
            # plot_skeleton_3d(true,pred,id_dir_path,"pred")
            # plot_skeleton_3d(true,pred,id_dir_path,"true_and_pred")

def plot_skeleton_3d(true, pred, id_dir_path, flag):

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5]]
    
    # ax = fig.add_subplot(111,projection='3d')
    ims = []

    print(f"output {flag}.gif")
    # print(max_frames)

    for j in range(src_tgt_len-1, int(len(true)/kpt_n_3), tgt_len):
        fig, ax = plt.subplots()
        plt.xlim(0,640)
        plt.ylim(0,480)

        # print(j)
        # print(true.shape)
        # print(pred.shape)
        x_true = [true[i]              for i in range(d_input*j + 0, d_input*(j+1), 2)]
        y_true = [img_height - true[i] for i in range(d_input*j + 1, d_input*(j+1) + 1, 2)]

        x_pred = [pred[i]              for i in range(d_input*j + 0, d_input*(j+1), 2)]
        y_pred = [img_height - pred[i] for i in range(d_input*j + 1, d_input*(j+1) + 1, 2)]

        # 軸ラベルを設定
        # ax.view_init(elev=0, azim=270)
        # ax.set_xlabel("x", size = 14)
        # ax.set_ylabel("z", size = 14)
        # ax.set_zlabel("y", size = 14)
        # 軸目盛を設定
        # ax.set_xticks([0, img_width/2, img_width])
        # ax.set_yticks([0, img_height/2, img_height])

        if flag == "true":
            im_0 = plt.plot(x_true, y_true, "b.",)
            for l, m in skeleton:
                im_0 += ax.plot([x_true[l-1],x_true[m-1]], [y_true[l-1],y_true[m-1]], 'b-')
            im = im_0
        
        elif flag == "pred":
            im_1 = plt.plot(x_pred, y_pred, "r.",)
            for l, m in skeleton:
                im_1 += ax.plot([x_pred[l-1],x_pred[m-1]], [y_pred[l-1],y_pred[m-1]], 'r-')
            im = im_1

        elif flag == "true_and_pred":
            im_0 = plt.plot(x_true, y_true, "b.",)
            im_1 = plt.plot(x_pred, y_pred, "r.",)
            for l, m in skeleton:
                im_0 += ax.plot([x_true[l-1],x_true[m-1]], [y_true[l-1],y_true[m-1]], 'b-')
            for l, m in skeleton:
                im_1 += ax.plot([x_pred[l-1],x_pred[m-1]], [y_pred[l-1],y_pred[m-1]], 'r-')
            im = im_0 + im_1
        
        else:
            print("Flag is not correct.")

        flag_dir_path = os.path.join(id_dir_path, flag)
        os.makedirs(flag_dir_path, exist_ok=True)

        plt.savefig(flag_dir_path + "/" + str(j) + ".png")
        # ims.append(im)

    # ani = animation.ArtistAnimation(fig, ims, interval=333, blit=True)
    # ani.save(id_dir_path + flag + ".gif", writer="imagemagick")

if __name__ == "__main__":

    # data_preprocess_padding()

    #para設定
    n_inputs  = d_input
    n_outputs = d_input
    n_hidden  = 32 #隠れ層(hidden)を64個に設定
    n_layers  = 1

    # model定義
    # model = Transformer(num_encoder_layers, d_model, d_input, d_output)
    # model = My_rnn_net(n_inputs, n_outputs, n_hidden, n_layers)
    # model = My_lstm_net(n_inputs, n_outputs, n_hidden, n_layers)
    model = My_gru_net(n_inputs, n_outputs, n_hidden, n_layers)

    print(model) #作成したRNNの層を簡易表示

    # 重み初期値設定
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # モデルをGPUへ転送
    model = model.to(device)

    #データセットの作成
    data_set = TimeSeriesDataset(src_len=src_len, model=model)

    # 学習と検証用に分割(9:1)
    train_len = int(len(data_set)*0.9)
    valid_len = len(data_set) - train_len
    train_dataset, valid_dataset = torch.utils.data.random_split(
        data_set, 
        [train_len, valid_len]
    )

    #train用のDataLoaderを定義
    train_loader = DataLoader(train_dataset,
                             batch_size=1, 
                             shuffle=True, pin_memory=True
                            )
    #valid用のDataLoaderを定義
    valid_loader = DataLoader(valid_dataset,
                             batch_size=1, 
                             shuffle=False, pin_memory=True
                            )
    # test用のDataLoaderを使用
    # test_loader = DataLoader(valid_dataset,
    #                          batch_size=1, 
    #                          shuffle=True
    #                         )

    #損失関数の定義
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()
    # criterion = torch.nn.SmoothL1Loss()

    #最適化手法の定義
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-03)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-03, momentum=0.9)
    # # スケジューラーの定義
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-08)

    history_train = {"train_loss": []}
    history_valid = {"valid_loss": []}
    for epoch in range(1, epochs + 1):
        
        loss_train = train(
            model=model, data_loader=train_loader, optimizer=optimizer, criterion=criterion
        )

        # if epoch > 1:
        history_train["train_loss"].append(loss_train)

        loss_valid = evaluate(
            model=model, data_loader=valid_loader, criterion=criterion
        )

        # if epoch > 1:
        history_valid["valid_loss"].append(loss_valid)

        print('[{}/{}] train loss: {:.5f}, valid loss: {:.5f}, best loss: {:.5f}'.format(
            epoch, epochs, loss_train, loss_valid, best_loss
        ))
            
        if best_loss > loss_valid:
            best_loss = loss_valid
            best_model = model
            torch.save(best_model, exp_dir_path+'best_model_weight.pth')

        if epoch % 30 == 0:
            torch.save(model, exp_dir_path+'epoch30_model_weight.pth')

        if epoch%10==0:
            plt.clf()
            plt.plot(history_train["train_loss"])
            plt.plot(history_valid['valid_loss'])
            plt.savefig(exp_dir_path+'history')

    torch.save(model, exp_dir_path+'last_model_weight.pth')



