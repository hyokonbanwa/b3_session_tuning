##ライブラリ
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.utils import make_grid

# モデル構造の表示
from torchinfo import summary

# その他
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
from PIL import Image

#チューニング
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random
import os
##------------------------------

##初期値
seed = 42 #シード値
tuningName = "tuning1"
##-----------------------------

##シード設定
#シードを固定したDataLoaderの実行結果を返す関数
def fix_seed_dataLoader(dataloader,**args):
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(dataloader,**args,worker_init_fn=seed_worker,generator=g)

#シードを固定化
def fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed) #cpuとcudaも同時に固定
    #torch.cuda.manual_seed(seed) #上記で呼び出される
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

fix_seed(seed=42)
##-----------------------------------

##デバイス設定
# GPUが利用できる場合はGPUを選択
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
##-----------------------------


##データの準備
def load_cifar10(transform, train:bool):
    cifar10_dataset = CIFAR10(
                        root='./data',
                        train=train,
                        download=True,
                        transform=transform
                        )
    return cifar10_dataset

#データ拡張
train_transform = transforms.Compose([
                            # transforms.RandomHorizontalFlip(p=0.5),
                            # transforms.RandomVerticalFlip(p=0.5),
                            # transforms.RandomRotation(degrees=30),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
])

test_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = load_cifar10(transform=test_transform, train=True) #訓練・検証用データなのでtransformしない
test_dataset = load_cifar10(transform=test_transform, train=False)

# データ数とデータ形状を確認
print(f'data size : {len(dataset)}')
print(f'data shape : {dataset[0][0].shape}')

# 学習用(Train)と検証用(Validation)への分割数を決める
valid_ratio = 0.1                               # 検証用データの割合を指定
valid_size = int(len(dataset) * valid_ratio)    # 検証用データの数
train_size = len(dataset) - valid_size          # 学習用データ = 全体 - 検証用データ
print(f'train samples : {train_size}')
print(f'valid samples : {valid_size}')

# 読み込んだデータをランダムに選んで分割
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
train_dataset.dataset.transform = train_transform #訓練用にtransform適用






# データローダーを作成
train_loader = fix_seed_dataLoader(
    train_dataset,          # データセットを指定
    batch_size=256,          # バッチサイズを指定
    shuffle=True,           # シャッフルの有無を指定
    drop_last=True,         # バッチサイズで割り切れないデータの使用の有無を指定
    pin_memory=True,        # 少しだけ高速化が期待できるおまじない
    num_workers=2          # DataLoaderのプロセス数を指定
)

valid_loader = fix_seed_dataLoader(
    valid_dataset,
    batch_size=256,
    shuffle=False,
    pin_memory=True,
    num_workers=2
) 


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

##--------------------------------------------------

##モデル定義
# Convolution層、BatchNormalization層(option)、ReLU層をまとめたブロック
# ブロックを作成することで、モデル定義のコードが煩雑になることを防ぐ
class EncoderBlock(nn.Module):
    def __init__(self, in_feature, out_future, use_bn=True):
        super().__init__()
        self.use_bn = use_bn

        self.in_feature = in_feature
        self.out_feature = out_future

        # BatchNormalizationを使用する場合、バイアス項は無くても良い
        # 標準化の際にバイアスもまとめて処理されるため
        self.conv = nn.Conv2d(in_feature, out_future, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_future)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out) if self.use_bn else x  # bn層を使わない場合はxを代入
        out = self.relu(out)
        return out

class Classifier(nn.Module):
    def __init__(self, class_num, enc_dim, in_w, in_h):
        super().__init__()

        # self.enc_dim = enc_dim
        self.in_w = in_w
        self.in_h = in_h
        self.fc_dim = enc_dim*4 * int(in_h/2/2/2) * int(in_w/2/2/2) #pooling回数分割る
        self.class_num = class_num

        self.encoder = nn.Sequential(
            EncoderBlock(3      , enc_dim),
            EncoderBlock(enc_dim, enc_dim),
            nn.MaxPool2d(kernel_size=2),#h,wが1/2

            EncoderBlock(enc_dim  , enc_dim*2),
            EncoderBlock(enc_dim*2, enc_dim*2),
            nn.MaxPool2d(kernel_size=2),

            EncoderBlock(enc_dim*2, enc_dim*4),
            EncoderBlock(enc_dim*4, enc_dim*4),
            nn.MaxPool2d(kernel_size=2),
        )


        self.fc = nn.Sequential(
            nn.Linear(self.fc_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.class_num),
        )
    
    def forward(self, x):
        out = self.encoder(x)
        out = out.view(-1, self.fc_dim)

        out = self.fc(out)

        return out
##--------------------------------------------

##重みの初期化
def initialize_weights(m):
    if isinstance(m, nn.Conv2d): # Convolution層が引数に渡された場合
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu') # kaimingの初期化
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)   # bias項は0に初期化
    elif isinstance(m, nn.BatchNorm2d):         # BatchNormalization層が引数に渡された場合
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):              # 全結合層が引数に渡された場合
        nn.init.kaiming_normal_(m.weight.data)  # kaimingの初期化
        nn.init.constant_(m.bias.data, 0)       # biasは0に初期化
##------------------------------------------

##モデル構築
# モデルの構築
model = Classifier(class_num=10, enc_dim=128, in_w=32, in_h=32)

# 計算に使用するデバイスへモデルを転送
model.to(device)

# 重みの初期化
model.apply(initialize_weights)
# model.applyは以下のコードと同じ動作をする
# for module in model.childlen():
#     initialize_weights(module)
##------------------------------------------

##損失関数と最適化手法
# 誤差関数(損失関数)を指定
criterion = nn.CrossEntropyLoss()  # 交差エントロピー
print(criterion)

# 最適化アルゴリズムを指定
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 最適化関数Adam
print(optimizer)





