# %%
import os, shutil
import re, gc, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import warnings, random
import cv2

warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import ipdb
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.sampler import BatchSampler
from torch.optim import lr_scheduler

# import torchvision
# from torchvision import transforms
# import torchvision.models as models
from torch.cuda.amp import GradScaler
import timm

# import yaml
from tqdm import tqdm
import time
import copy
# from collections import defaultdict

import albumentations as albu
from albumentations.pytorch import transforms as AT

from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

# %% [markdown]
# ## パラメータの設定

# %%
ARGS = {'DATA_DIR': '',
  'OUT_DIR': './5fold_bin/',
#   'model_name': "tf_efficientnet_b7.ns_jft_in1k",
#   'model_name': "maxvit_base_tf_512.in21k_ft_in1k",
#   'model_name': "	maxvit_large_tf_512.in21k_ft_in1k",
#   'model_name': "	vit_large_patch14_clip_336.laion2b_ft_in12k_in1k",
#   'model_name': "	vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k",
# 'model_name': "	convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384",
# 'model_name': "	maxvit_base_tf_384.in21k_ft_in1k",
  'model_name': "eva_large_patch14_336.in22k_ft_in22k_in1k",
#   'model_name': "eva_large_patch14_196.in22k_ft_in22k_in1k",
#   'model_name': "eva02_large_patch14_448.mim_in22k_ft_in22k_in1k",
# 'model_name': "	vit_large_patch14_clip_336.laion2b_ft_in12k_in1k",
# 'model_name': "	vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k",
# 'model_name': "	convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384",
# 'model_name': "	maxvit_base_tf_384.in21k_ft_in1k",
# 		
  "backbone_only": False,
  "batchNorm": True,
  "num_albu": 4,
  'n_fold': 5,
  'epochs': 20,
  'drop_rate': 0.1,
  "patient": 20,
  'unfreeze_backbone_epoch': 0,

  'criterion': 'CrossEntropy',
#   'image_size': (600, 600),
#   'image_size': (512 ,512),
#   'image_size': (512 ,512),
#   'image_size': (448 ,448),
  'image_size1': (336, 336),
  'image_size2': (224, 224),
#   'image_size': (224, 224),
#   'image_size': (384, 384),
# 'image_size': (384, 384),
# 'image_size': (224,224),
#   'image_size': (336, 336),
#   'image_size': (224, 224),
#   'image_size': (196, 196),

  'train_batch_size': 10,
  'valid_batch_size': 10,
  'seed': 777,
  'optimizer': 'AdamW',
  'learning_rate': 1e-05,
  'scheduler': 'CosineAnnealingLR',
  'min_lr':1e-07,
  'T_max': 500,
  'n_accumulate': 1,
  'clip_grad_norm': 'None',
  'apex': True,
  'num_classes': 2,
  'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  }
ARGS

# %%
def get_logger(filename):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler2)
    return logger

#再現性を出すために必要な関数となります
def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)
    random.seed(worker_id)
    np.random.seed(worker_id)
    torch.cuda.manual_seed(worker_id)
    os.environ['PYTHONHASHSEED'] = str(worker_id)

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


LOGGER = get_logger(ARGS['OUT_DIR']+'train_beit')
set_seed(ARGS["seed"])


# %%
class CustomDataset(Dataset):
    def __init__(self, df, transform, data_type):
        self.df = df
        self.data_type = data_type
        self.images = []

        if self.data_type == "train":
            self.image_paths = df['image_name']
            self.labels = df['label']
        if self.data_type == "test":
            self.image_paths = df[0]

        self.transform= transform

        # Load images into RAM beforehand
        for image_path in self.image_paths:
            if self.data_type == "train":
                image = cv2.imread(f"./train/{image_path}")
            if self.data_type == "test":
                image = cv2.imread(f"./test/{image_path}")
            #     image = cv2.imread(f"./compressed_train/{image_path[:4]}_Swin2SR.png")
            # if self.data_type == "test":
            #     image = cv2.imread(f"./compressed_test/{image_path[:4]}_Swin2SR.png")

            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except :
                ipdb.set_trace()
                
            self.images.append(image)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        image = self.images[index]

        image = self.transform(image=image)["image"]

        if self.data_type == "train":
            label = self.labels[index]
            label = torch.tensor(label, dtype=torch.long)
            return image, label

        if self.data_type == "test":
            return image

# import torch.nn.functional as F
# class CustomModel(nn.Module):
#     def __init__(self, embedding_size=512):
#         super(CustomModel, self).__init__()
#         # バックボーン1としてResNetをロード
#         self.backbone1 = timm.create_model('eva02_base_patch14_224.mim_in22k', drop_rate=ARGS["drop_rate"],pretrained=True, num_classes=int(embedding_size/2))
#         in_features1 = self.backbone1.head.out_features
   
#         # バックボーン2としてDenseNetをロード
#         self.backbone2 = timm.create_model('swinv2_large_window12to16_192to256.ms_in22k_ft_in1k', drop_rate=ARGS["drop_rate"],pretrained=True, num_classes=int(embedding_size/2))
#         # print(self.backbone2.head.in_features)
#         in_features2 = self.backbone2.head.in_features

#         self.head = nn.Sequential(
#             # nn.GELU(),
#             # nn.Dropout(0.5),
#             # nn.Linear(embedding_size, int(embedding_size/4)),
#             nn.GELU(),
#             nn.Dropout(0.5),
#             nn.Linear(int(embedding_size), ARGS["num_classes"])
#         )
#     def forward(self, input_x):
#         # バックボーン1の特徴量抽出
#         x1 = F.interpolate(input_x, size=(ARGS["image_size1"][0], ARGS["image_size1"][1]), mode='bilinear', align_corners=False)
#         x2 = F.interpolate(input_x, size=(ARGS["image_size2"][0], ARGS["image_size2"][1]), mode='bilinear', align_corners=False)
#         # print(x1.shape)
#         x12 = self.backbone1(x1)
#         x22 = self.backbone2(x2)
        
#         x = torch.cat((x12, x22), dim=1)
#         # ArcFaceヘッドを適用
#         logits = self.head(x)
#         return logits

import torch.nn.functional as F
class CustomModel(nn.Module):
    def __init__(self, embedding_size=512):
        super(CustomModel, self).__init__()
        # バックボーン1としてResNetをロード
        self.backbone1 = timm.create_model('eva_large_patch14_336.in22k_ft_in22k_in1k', drop_rate=ARGS["drop_rate"],pretrained=True, num_classes=int(embedding_size/2))
        # in_features1 = self.backbone1.head.out_features
   
        # バックボーン2としてDenseNetをロード
        self.backbone2 = timm.create_model('tf_efficientnet_b5.ns_jft_in1k', drop_rate=ARGS["drop_rate"],pretrained=True, num_classes=int(embedding_size/2))
        # print(self.backbone2.head.in_features)
        # in_features2 = self.backbone2.num_features

        self.head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_size, int(embedding_size/4)),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(int(embedding_size/4), ARGS["num_classes"])
        )
    def forward(self, input_x):
        # バックボーン1の特徴量抽出
        x1 = F.interpolate(input_x, size=(ARGS["image_size1"][0], ARGS["image_size1"][1]), mode='bilinear', align_corners=False)
        x2 = F.interpolate(input_x, size=(ARGS["image_size2"][0], ARGS["image_size2"][1]), mode='bilinear', align_corners=False)
        # print(x1.shape)
        x12 = self.backbone1(x1)
        x22 = self.backbone2(x2)
        
        x = torch.cat((x12, x22), dim=1)
        # ArcFaceヘッドを適用
        logits = self.head(x)
        return logits



def create_model(args):
    if ARGS["backbone_only"]:
        num_classes = ARGS["num_classes"]
        model = timm.create_model(ARGS["model_name"], pretrained=True, drop_rate = ARGS["drop_rate"], num_classes=num_classes)
    else:    
        model = CustomModel()
    return model

# %%
#sample_submit.csvを読み込みます
submit = pd.read_csv(f"./sample_submit.csv", header=None)
submit.head()

# %% [markdown]
# ## Inference

# %%
# test用のデータ拡張
image_transform_test = albu.Compose([
    # albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
    albu.Normalize(),
    AT.ToTensorV2()
    ])

test_dataset = CustomDataset(submit, image_transform_test, data_type="test")
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# %%
@torch.no_grad()
def valid_fn(model, dataloader, device):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    predict_list = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, images in bar:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            #出力にソフトマックス関数を適用
            predicts = outputs.softmax(dim=1)

            # # predicts の値が 0.999 以上の場合は 1 に変更
            predicts[predicts >= 1.0] = 1
            # # predicts の値が 0.0001 以下の場合は 0 に変更
            predicts[predicts <= 0.0] = 0

        predicts = predicts.cpu().detach().numpy()
        predict_list.append(predicts)
    predict_list = np.concatenate(predict_list, axis=0)
    #予測値が1である確率を提出します。
    predict_list = predict_list[:, 1]
    gc.collect()

    return predict_list

# %%
def inference(model_paths, dataloader, device):
    final_preds = []
    for i, path in enumerate(model_paths):
        # if (i == 2) or (i == 4) or (i == 5):
        #     model = create_model(ARGS)
        #     model = model.to(device)

        #     #学習済みモデルの読み込み
        #     model.load_state_dict(torch.load(path))
        #     model.eval()

        #     print(f"Getting predictions for model {i}")
        #     preds = valid_fn(model, dataloader, device)
        #     final_preds.append(preds)
        # else:
        #     continue

        model = create_model(ARGS)
        model = model.to(device)

        #学習済みモデルの読み込み
        model.load_state_dict(torch.load(path))
        model.eval()

        print(f"Getting predictions for model {i}")
        preds = valid_fn(model, dataloader, device)
        final_preds.append(preds)

    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds

# ご分割のとき
# def inference(model_paths, dataloader, device):
#     final_preds = []
#     for i, path in enumerate(model_paths):
#         model = create_model(ARGS)
#         model = model.to(device)
#         #学習済みモデルの読み込み
#         model.load_state_dict(torch.load(path))
#         model.eval()
#         print(f"Getting predictions for model {i+1}")
#         preds = valid_fn(model, dataloader, device)
#         final_preds.append(preds)
#     # ipdb.set_trace
#     predict_df = pd.DataFrame(final_preds)
#     predict_list = np.array(final_preds)
#     kukan_np =np.round(np.floor(predict_list*3)/3,decimals=5)
#     kukan_pd = pd.DataFrame(kukan_np)
#     kukan_mode = kukan_pd.mode(axis=0).T.to_numpy().tolist()
#     tasuketu_list = []
#     for i,num_list in enumerate(kukan_mode):
#         t_list= []
#         for num in num_list:
#             t_list += (predict_df[i][kukan_pd[i] == num].dropna().tolist())
#         tasuketu_list.append(sum(t_list)/len(t_list))
#     return tasuketu_list

# ご分割のとき
# def inference(model_paths, dataloader, device):
#     final_preds = []
#     for i, path in enumerate(model_paths):
#         model = create_model(ARGS)
#         model = model.to(device)
#         #学習済みモデルの読み込み
#         model.load_state_dict(torch.load(path))
#         model.eval()
#         print(f"Getting predictions for model {i+1}")
#         preds = valid_fn(model, dataloader, device)
#         final_preds.append(preds)
#     predict_df = pd.DataFrame(final_preds)
#     predict_list = np.array(final_preds)
#     kukan_np = np.floor(predict_list*5)/5
#     kukan_pd = pd.DataFrame(kukan_np)
#     kukan_mode = kukan_pd.mode(axis=0)[0:1].to_numpy().tolist()[0]
#     kukan_mode_list = [round(num, 2) for num in kukan_mode]
#     tasuketu_list = []
#     for i in range(len(final_preds[0])):
#         tasuketu_list.append(predict_df[i][kukan_pd[i] == kukan_mode_list[i]].mean())
#     return tasuketu_list
    # # final_preds_mean = np.mean(final_preds, axis=0)
    # # final_preds_median = np.median(final_preds, axis=0)
    # # return final_preds_mean, final_preds_median
    # return final_preds_mean
# %%
fold_list = []

MODEL_PATHS = [
    f"{ARGS['OUT_DIR']}/Score-Fold-{i}.bin" for i in range(ARGS["n_fold"])
]

predict_list = inference(MODEL_PATHS, test_loader, ARGS["device"])
submit[1] = predict_list
submit.head()


# %%
scores = 0.0003
submit.to_csv(f"{ARGS['OUT_DIR']}/submission_CV{scores:.4f}.csv", index=False, header=None)

# %% [markdown]
# ## SIGNATE CLIでSubmit(任意)
# * signate cli用のAPIキー(signate.json)を`/root/.signate/signate.json`に配置すればsubmit可能

# %%
# OUT_DIR="./"
# SUBMIT_FILE=f"{OUT_DIR}/submission_CV{scores:.4f}.csv"
# NOTE="StratifiedKfold5-vit_l_16pretrained+imgsize224'

# !pip install signate
# !cp /content/drive/MyDrive/SIGNATE/signate.json /root/.signate/signate.json
# !signate submit --competition-id=1106 {SUBMIT_FILE} --note {NOTE}

# %%
