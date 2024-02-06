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
  'OUT_DIR': './',
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
  "resize_layer" : True,
  "batchNorm": True,
  "num_albu": 4,
  'n_fold': 5,
  'epochs': 20,
  'drop_rate': 0.5,
  "patient": 5,
  'unfreeze_backbone_epoch': 0,

  'criterion': 'CrossEntropy',
#   'image_size': (600, 600),
#   'image_size': (512 ,512),
#   'image_size': (512 ,512),
#   'image_size': (448 ,448),
#   'image_size': (336, 336),
  'image_size': (672, 672),
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


LOGGER = get_logger(ARGS['OUT_DIR']+'train_okada')
set_seed(ARGS["seed"])

# %% [markdown]
# ## Create Folds

# %%
def create_folds(data, num_splits, seed):
    data["kfold"] = -1

    mskf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    labels = ["label"]
    data_labels = data[labels].values

    for f, (t_, v_) in enumerate(mskf.split(data, data_labels)):
        data.loc[v_, "kfold"] = f

    return data

train = pd.read_csv("train.csv")
train = create_folds(train, num_splits=ARGS["n_fold"], seed=ARGS["seed"])
print("Folds created successfully")

train.head()

# %%
train["label"].value_counts()

# %%
train['kfold'].value_counts()

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

# %%


# # Augumentation用
# image_transform = albu.Compose([
#     albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
#     albu.HorizontalFlip(p=0.5),
#     albu.VerticalFlip(p=0.5),
#     albu.RandomBrightnessContrast(p=0.3),
#     albu.RandomGamma(gamma_limit=(85, 115), p=0.3),
#     albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.10, rotate_limit=45, p=0.5),
#     albu.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     AT.ToTensorV2()
# ])

image_transform = albu.Compose([
    albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
    albu.RandomRotate90(p=0.1),
    albu.RandomContrast(p=0.1),
    albu.RandomGamma(p=0.1),
    albu.RandomBrightness(p=0.1),
    albu.HorizontalFlip(p=0.1),
    albu.VerticalFlip(p=0.1),
    albu.Blur(p=0.1),
    albu.Downscale(p=0.1),
    albu.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    AT.ToTensorV2()
    ])

# image_transform = [
#     albu.Compose([
#     albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
#     albu.Normalize(),
#     AT.ToTensorV2()
#     ]),
#     albu.Compose([
#     albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
#     albu.RandomRotate90(p=1.0),
#     albu.Normalize(),
#     AT.ToTensorV2()
#     ]),
#     # albu.Compose([
#     # albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
#     # albu.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=45, p=1.0),
#     # albu.Normalize(),
#     # AT.ToTensorV2()
#     # ]),
#     # albu.Compose([
#     # albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
#     # albu.RandomContrast(p=1.0),
#     # albu.Normalize(),
#     # AT.ToTensorV2()
#     # ]),
#     albu.Compose([
#     albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
#     albu.RandomBrightness(p=1.0),
#     albu.Normalize(),
#     AT.ToTensorV2()
#     ]),
#     albu.Compose([
#     albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
#     albu.RandomGamma(p=1.0),
#     albu.Normalize(),
#     AT.ToTensorV2()
#     ]),
#     albu.Compose([
#     albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
#     albu.HorizontalFlip(p=1.0),
#     albu.Normalize(),
#     AT.ToTensorV2()
#     ]),
#     albu.Compose([
#     albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
#     albu.VerticalFlip(p=1.0),
#     albu.Normalize(),
#     AT.ToTensorV2()
#     ]),
#     albu.Compose([
#     albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
#     albu.Downscale(p=1.0),
#     albu.Normalize(),
#     AT.ToTensorV2()
#     ]),
#     # albu.Compose([
#     # albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
#     # albu.JpegCompression(p=1.0),
#     # albu.Normalize(),
#     # AT.ToTensorV2()
#     # ]),
#     # albu.Compose([
#     # albu.RandomCrop(height=ARGS["image_size"][0], width=ARGS["image_size"][1], p=1.0),
#     # albu.Normalize(),
#     # AT.ToTensorV2()
#     # ]),
#     albu.Compose([
#     albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
#     albu.Blur(p=1.0),
#     albu.Normalize(),
#     AT.ToTensorV2()
#     ])
#                    ]
# Augumentation用
image_transform_valid = albu.Compose([
    albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
    albu.Normalize(),
    AT.ToTensorV2()
])

# %% [markdown]
# ## 学習用関数定義
# `torch.cuda.amp.autocast(enabled=ARGS["apex"]):`を入れると計算の高速化とメモリの節約ができます。
# %%
def train_one_epoch(model, optimizer, train_loader, device, epoch):
    model.train()
    dataset_size = 0
    running_loss = 0.0
    running_score = []
    running_score_y = []
    scaler = GradScaler(enabled=ARGS["apex"])
    unfreeze_backbone_epoch = ARGS["unfreeze_backbone_epoch"]

    train_loss = []
    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (images, targets) in bar:
        images = images.to(device)
        targets = targets.to(device)

        batch_size = targets.size(0)

        # バックボーンの凍結状態を判断して凍結/解除する
        if epoch > unfreeze_backbone_epoch:
            for param in model.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = False

        # # 最終層のみをトレーニングするために requires_grad を設定
        # for param in model.head.parameters():
        #     param.requires_grad = True

        with torch.cuda.amp.autocast(enabled=ARGS["apex"]):
            outputs = model(images)
            loss = criterion(ARGS, outputs, targets)

        scaler.scale(loss).backward()

        if ARGS["clip_grad_norm"] != "None":
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), ARGS["clip_grad_norm"])
        else:
            grad_norm = None

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        train_loss.append(loss.item())

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        running_score.append(outputs.detach().cpu().numpy())
        running_score_y.append(targets.detach().cpu().numpy())

        score = get_score(running_score_y, running_score)

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        Train_Acc=score[0],
                        Train_Auc=score[1],
                        LR=optimizer.param_groups[0]['lr']
                        )

    gc.collect()
    return epoch_loss, score

# %%
@torch.no_grad()
def valid_one_epoch(args, model, optimizer, valid_loader, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    preds = []
    valid_targets = []
    softmax = nn.Softmax()

    bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for step, (images, targets) in enumerate(valid_loader):
      images = images.to(args["device"])
      targets = targets.to(args["device"])
      batch_size = targets.size(0)
      with torch.no_grad():
        outputs = model(images)
        predict = outputs.softmax(dim=1)
        # predicts の値が 0.999 以上の場合は 1 に変更
        predict[predict >= 0.999] = 1
        # predicts の値が 0.0001 以下の場合は 0 に変更
        predict[predict <= 0.001] = 0
        loss = criterion(args, outputs, targets)

      running_loss += (loss.item() * batch_size)
      dataset_size += batch_size

      epoch_loss = running_loss / dataset_size

      preds.append(predict.detach().cpu().numpy())
      valid_targets.append(targets.detach().cpu().numpy())

      if len(set(np.concatenate(valid_targets))) == 1:
          continue
      score = get_score(valid_targets, preds)

      bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                      Valid_Acc=score[0],
                      Valid_Auc=score[1],
                      LR=optimizer.param_groups[0]['lr'])

    return epoch_loss, preds, valid_targets, score


# %%
def one_fold(model, optimizer, schedulerr, device, num_epochs, fold):

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_score = None
    best_prediction = None

    best_score = -np.inf
    past_val_loss = 1.0
    loss_count = 0

    for epoch in range(1, 1+num_epochs):
          
      train_epoch_loss, train_score = train_one_epoch(model, optimizer,
                                              train_loader=train_loader,
                                              device=device, epoch=epoch)

      train_acc, train_auc = train_score

      val_epoch_loss, predictions, valid_targets, valid_score = valid_one_epoch(ARGS,
                                                                                model,
                                                                                optimizer,
                                                                                valid_loader=valid_loader,
                                                                                epoch=epoch)
      
      if val_epoch_loss > past_val_loss:
          past_val_loss = val_epoch_loss
          loss_count += 1
          if loss_count == ARGS["patient"]:
              LOGGER.info("Early Stopping in epoch{epoch}")
              break
      else:
          past_val_loss = val_epoch_loss
          loss_count = 0

      valid_acc, valid_auc = valid_score

      LOGGER.info(f'Epoch {epoch} - Train Acc: {train_acc:.4f}  Train Auc: {train_auc:.4f}  Valid Acc: {valid_acc:.4f}  Valid Auc: {valid_auc:.4f}')
    #   LOGGER.info(f'Epoch {epoch} - avg_train_loss: {train_epoch_loss:.4f}  avg_val_loss: {val_epoch_loss:.4f}')
      LOGGER.info(f'Epoch {epoch} - Train Acc: {train_acc:.4f}  Valid Acc: {valid_acc:.4f}  Valid Auc: {valid_auc:.4f}')
    #   LOGGER.info(f'Epoch {epoch} - Train Acc: {train_acc:.4f}  Valid Acc: {valid_acc:.4f}')

    #   一回目のとき
      if best_score is None:
        best_score = valid_auc

        print(f"{b_}Validation Loss Improved (  ---> {valid_auc})")
        best_epoch_score = valid_auc
        best_model_wts = copy.deepcopy(model.state_dict())
        # PATH = f"Score-Fold-{fold}.bin"
        PATH = ARGS["OUT_DIR"] + f"Score-Fold-{fold}.bin"
        torch.save(model.state_dict(), PATH)
        # Save a model file from the current directory
        print(f"Model Saved{sr_}")

        best_prediction = np.concatenate(predictions, axis=0)[:,1]
          
      elif valid_auc >= best_score:
        best_score = valid_auc

        print(f"{b_}Validation Loss Improved ({best_epoch_score} ---> {valid_auc})")
        best_epoch_score = valid_auc
        best_model_wts = copy.deepcopy(model.state_dict())
        # PATH = f"Score-Fold-{fold}.bin"
        PATH = ARGS["OUT_DIR"] + f"Score-Fold-{fold}.bin"
        torch.save(model.state_dict(), PATH)
        # Save a model file from the current directory
        print(f"Model Saved{sr_}")

        best_prediction = np.concatenate(predictions, axis=0)[:,1]

    end = time.time()
    time_elapsed = end - start

    LOGGER.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    LOGGER.info("Best Score: {:.4f}".format(best_epoch_score))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_prediction, valid_targets

# %%
# def create_model(args):
#     model = models.vit_l_16(pretrained=True)
#     model.heads[0] = torch.nn.Linear(in_features=model.heads[0].in_features, out_features=args["num_classes"], bias=True)
#     return model

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # self.backbone = timm.create_model('swinv2_base_window16_256.ms_in1k', pretrained=True, num_classes=100)

        self.backbone = timm.create_model(ARGS["model_name"], pretrained=True, num_classes=128)
        self.in_features = self.backbone.num_classes
        if ARGS["batchNorm"]:
        # 中間層を定義
            self.head = nn.Sequential(
                nn.BatchNorm1d(self.in_features),
                nn.Dropout(ARGS["drop_rate"]),
                nn.Linear(self.in_features, 16),
                # nn.ReLU(),
                nn.Sigmoid(),
                nn.Linear(16, 2),
                nn.Sigmoid()
                # nn.Softmax()
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(ARGS["drop_rate"]),
                nn.Linear(self.in_features, 16),
                nn.Sigmoid(),
                nn.Linear(16, 2),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

class ResizeModel(nn.Module):
    def __init__(self):
        super(ResizeModel, self).__init__()
        self.resize_layer = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.backbone = timm.create_model(ARGS["model_name"], pretrained=True, drop_rate = ARGS["drop_rate"], num_classes=ARGS["num_classes"])


    def forward(self, x):
        x = self.resize_layer(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = self.backbone(x)
        return x

def create_model(args):
    if ARGS["backbone_only"]:
        model = timm.create_model(ARGS["model_name"], pretrained=True, drop_rate = ARGS["drop_rate"], num_classes=ARGS["num_classes"])
    elif ARGS["resize_layer"]:
        model = ResizeModel()
    else:    
        model = CustomModel()
    return model

def criterion(args, outputs, labels, class_weights=None):
    if args['criterion'] == 'CrossEntropy':
      return nn.CrossEntropyLoss(weight=class_weights).to(args["device"])(outputs, labels)
    elif args['criterion'] == "None":
        return None

def fetch_optimizer(optimizer_parameters, lr, betas, optimizer_name="Adam"):
    if optimizer_name == "Adam":
        optimizer = optim.Adam(optimizer_parameters, lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(optimizer_parameters, lr=lr, betas=betas)
    return optimizer

def fetch_scheduler(args, train_size, optimizer):

    if args['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=args['T_max'],
                                                   eta_min=args['min_lr'])
    elif args['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=args['T_0'],
                                                             eta_min=args['min_lr'])
    elif args['scheduler'] == "None":
        scheduler = None

    return scheduler

def get_score(y_trues, y_preds):

    predict_list, targets_list = np.concatenate(y_preds, axis=0), np.concatenate(y_trues)
    predict_list_proba = predict_list.copy()[:, 1]
    predict_list = predict_list.argmax(axis=1)

    accuracy = accuracy_score(predict_list, targets_list)
    auc_score = roc_auc_score(targets_list, predict_list_proba)

    return (accuracy, auc_score)

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

# クロスバリデーション
def prepare_loaders(args, train, image_transform, fold):
    df_train = train[train.kfold != fold].reset_index(drop=True)
    df_valid = train[train.kfold == fold].reset_index(drop=True)
    # train_data_list = []
    # valid_data_list = []
    # for i in range(len(image_transform)):
    #   train_data_list.append(CustomDataset(df_train, image_transform[i], data_type="train"))
    #   valid_data_list.append(CustomDataset(df_valid, image_transform[i], data_type="train"))

    train_dataset = CustomDataset(df_train, image_transform, data_type="train")
    # valid_dataset = ConcatDataset(valid_data_list)
    valid_dataset = CustomDataset(df_valid, image_transform_valid, data_type="train")
    print("学習データ数",len(train_dataset))
    print("学習データ数",len(valid_dataset))

    n_classes = ARGS["num_classes"]
    train_n_samples = int((ARGS["train_batch_size"]) / (n_classes))
    valid_n_samples = int((ARGS["valid_batch_size"]) / (n_classes))

    train_balanced_batch_sampler = BalancedBatchSampler(train_dataset, n_classes, train_n_samples)
    train_loader = DataLoader(train_dataset,
                              batch_size=args['train_batch_size'],
                            #   batch_sampler=train_balanced_batch_sampler,
                              num_workers=4,
                              shuffle=False, pin_memory=True
                              )
    
    valid_balanced_batch_sampler = BalancedBatchSampler(valid_dataset, n_classes, valid_n_samples)
    valid_loader = DataLoader(valid_dataset,
                            #   batch_sampler=valid_balanced_batch_sampler, 
                              batch_size=args['valid_batch_size'],
                              num_workers=4,
                              shuffle=False, pin_memory=True
                              )
    
    return train_loader, valid_loader

# %%
train_copy = train.copy()
LOGGER.info(ARGS)
for fold in range(0, ARGS['n_fold']):
    print(f"{y_}====== Fold: {fold} ======{sr_}")
    LOGGER.info(f"========== fold: {fold} training ==========")

    # Create Dataloaders
    train_loader, valid_loader = prepare_loaders(args=ARGS, train=train, image_transform=image_transform, fold=fold)

    model = create_model(ARGS)
    # if fold == 0:
    #     print(model)   
    # for p in model.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_uniform_(p)

    model = model.to(ARGS["device"])

    #損失関数・最適化関数の定義
    optimizer = fetch_optimizer(model.parameters(), optimizer_name=ARGS["optimizer"], lr=ARGS["learning_rate"], betas=(0.9, 0.999))

    scheduler = fetch_scheduler(args=ARGS, train_size=len(train_loader), optimizer=optimizer)

    model, predictions, targets = one_fold(model, optimizer, scheduler, device=ARGS["device"], num_epochs=ARGS["epochs"], fold=fold)

    train_copy.loc[train_copy[train_copy.kfold == fold].index, "oof"] = predictions

    del model, train_loader, valid_loader
    _ = gc.collect()
    torch.cuda.empty_cache()

scores = roc_auc_score(train_copy["label"].values, train_copy["oof"].values)
LOGGER.info(f"========== CV ==========")
LOGGER.info(f"CV: {scores:.4f}")

# %%
# OOF
train_copy.to_csv(ARGS['OUT_DIR'] + f'oof.csv', index=False)

# %% [markdown]
# ## 結果の表示

# %%
#sample_submit.csvを読み込みます
submit = pd.read_csv(f"./sample_submit.csv", header=None)
submit.head()

# %% [markdown]
# ## Inference

# %%
# test用のデータ拡張
image_transform_test = albu.Compose([
    albu.Resize(ARGS["image_size"][0], ARGS["image_size"][1]),
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
        model = create_model(ARGS)
        model = model.to(device)

        #学習済みモデルの読み込み
        model.load_state_dict(torch.load(path))
        model.eval()

        print(f"Getting predictions for model {i+1}")
        preds = valid_fn(model, dataloader, device)
        final_preds.append(preds)

    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds

# %%
MODEL_PATHS = [
    f"{ARGS['OUT_DIR']}/Score-Fold-{i}.bin" for i in range(ARGS["n_fold"])
]

predict_list = inference(MODEL_PATHS, test_loader, ARGS["device"])
submit[1] = predict_list
submit.head()

# %%
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



