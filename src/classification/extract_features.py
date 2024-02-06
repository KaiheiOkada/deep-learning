import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.cluster import DBSCAN
import torchvision.models as models
import torchvision.transforms as transforms

# モデルの定義
model = models.resnet50(pretrained=True)
model.eval()

# 他のモデルも定義可能
# model = models.vgg16(pretrained=True)
# model.eval()

# モデルの分類層の確認
classification_layer = model.fc
print(classification_layer)

# 特徴量の形状を取得
dummy_input = torch.randn(1, 3, 224, 224)
features = model.features(dummy_input)
feature_shape = features.shape
print(feature_shape)

# 画像から特徴量を抽出する関数の定義
def extract_features(image):
    # 画像の前処理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # 特徴量の抽出
    with torch.no_grad():
        features = model(input_batch)

    return features

# # 画像から特徴量を抽出する関数の実行
# image = ...  # 画像の読み込みや生成
# features = extract_features(image)

# データセットとデータローダーの定義
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(root='path/to/dataset', transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# データセットのすべてのデータに対して特徴量を取り出す
all_features = []
for images, _ in dataloader:
    batch_features = extract_features(images)
    all_features.append(batch_features)
all_features = torch.cat(all_features, dim=0)

# 取り出した特徴量に対してDBScanを用いてクラスタリングする
clustering = DBSCAN(eps=0.5, min_samples=5)
labels = clustering.fit_predict(all_features)
print(labels)
