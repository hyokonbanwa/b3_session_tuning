##----ライブラリ
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
# from torchvision.utils import make_grid

# モデル構造の表示
# from torchinfo import summary

# その他
import numpy as np
# import matplotlib.pyplot as plt
# import pathlib
from tqdm import tqdm
# from PIL import Image

#チューニング
from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime
import random
import os
##--------------------










##------------シード関係
#クロージャを利用してseed値を固定 https://qiita.com/naomi7325/items/57d141f2e56d644bdf5f
def initialize_fix_seed_dataLoader(seed):
    def fix_seed_dataLoader(dataloader,**args):#シードを固定したDataLoader　DataLoaderの代わりに使う
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(dataloader,**args,worker_init_fn=seed_worker,generator=g)
    
    return fix_seed_dataLoader

#シードを固定化する関数
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


##------------------------------------------------------

##-------------writer関係
def make_writer(tuningName):
    # now = datetime.now()
    cur_path = os.getcwd()
    #basename_without_ext = os.path.splitext(os.path.basename(__file__))[0] #pythonファイルの時ファイル名取得
    log_path = os.path.join(cur_path,"log","session2021_cnn-2",tuningName) #"このファイルのあるディレクトリ/log/fixseed/実行日時"以下にファイルを作成
    print(log_path)
    writer = SummaryWriter(log_dir=log_path)
    return writer

##------------データセット
def load_cifar10(transform, train:bool):
    cifar10_dataset = CIFAR10(
                        root='./data',
                        train=train,
                        download=True,
                        transform=transform
                        )
    return cifar10_dataset
##-------------------------

##----------モデル関係
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
##--------------------------------------------

##---------------------訓練関係
def train_model(train_loader,model,criterion,optimizer,device):
        #========== 学習用データへの処理 ==========#
    # モデルを学習モードに設定
    model.train()

    #ミニバッチ＝イテレーションごとの
    batch_loss = []
    batch_acc  = []

    # ミニバッチ単位で繰り返す
    for batch in train_loader:
        x, label = batch
        x = x.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # 学習のメイン処理
        optimizer.zero_grad()       # (1) パラメータの勾配を初期化
        out = model(x)              # (2) データをモデルに入力(順伝播)
        loss = criterion(out, label)    # (3) 誤差関数の値を算出
        loss.backward()             # (4) パラメータの勾配を算出(逆伝播)
        optimizer.step()            # (5) 勾配の情報からパラメータを更新

        # 正答率(accuracy)を算出
        pred_class = torch.argmax(out, dim=1)        # モデルの出力から予測ラベルを算出
        acc = torch.sum(pred_class == label)/len(label) # 予測ラベルと正解ラベルが一致すれば正解

        # 1バッチ分の誤差と正答率をリストに保存
        batch_loss.append(loss)
        batch_acc.append(acc)

    # バッチ単位の結果を平均し、1エポック分の誤差を算出
    train_avg_loss = torch.tensor(batch_loss).mean()
    train_avg_acc  = torch.tensor(batch_acc).mean()

    metrics = {"train_loss":train_avg_loss,"train_acc":train_avg_acc}

    return metrics
##-------------------------------------------

##---------------------検証関係
def valid_model(valid_loader,model,criterion,device):
        #========== 検証用データへの処理 ==========#
    # モデルを評価モードに設定
    model.eval()
    batch_loss = []
    batch_acc = []

    # 勾配計算に必要な情報を記録しないよう設定
    with torch.no_grad():
        # ミニバッチ単位で繰り返す
        for batch in valid_loader:
            x, label = batch
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # 順伝播と誤差の計算のみ行う
            out = model(x)
            loss = criterion(out, label)

            # 正答率(accuracy)を算出
            pred_class = torch.argmax(out, dim=1)
            acc = torch.sum(pred_class == label)/len(label)

            # 1バッチ分の誤差をリストに保存
            batch_loss.append(loss)
            batch_acc.append(acc)

        # バッチ単位の結果を平均し、1エポック分の誤差を算出
        valid_avg_loss = torch.tensor(batch_loss).mean()
        valid_avg_acc  = torch.tensor(batch_acc).mean()

    metrics = {"valid_loss":valid_avg_loss,"valid_acc":valid_avg_acc}
    return metrics
##----------------------------------------------------

##----------------------テスト関係
# モデルを評価モードに設定
def test_model(test_loader,model,criterion,device):
    model.eval()
    batch_loss = []
    batch_acc = []

    # 勾配計算に必要な情報を記録しないよう設定
    with torch.no_grad():
        # ミニバッチ単位で繰り返す
        for batch in test_loader:
            x, label = batch
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # 順伝播のみ行う
            out = model(x)
            batch_loss = criterion(out, label)

            # 正答率(accuracy)を算出
            pred_class = torch.argmax(out, dim=1)
            acc = torch.sum(pred_class == label)/len(label)

            # 1バッチ分のaccをリストに保存
            batch_acc.append(acc)

        # バッチ単位の結果を平均し、1エポック分の誤差を算出
        test_avg_loss = torch.tensor(batch_loss).mean().item()
        test_avg_acc  = torch.tensor(batch_acc).mean().item()

    metrics = {"test_loss":test_avg_loss,"test_acc":test_avg_acc}
    return metrics

##--------------------------------


def main():
    ##変更する変数
    seed = 42 #シード値
    tuningName = "tuning2"#tensorBoardでの識別用
    # 学習回数(エポック)を指定
    epochs = 3
    ##----------------

    fix_seed(seed) #seed固定化
    fix_seed_dataLoader = initialize_fix_seed_dataLoader(seed)#seed固定のDataLoader関数を作成　DataLoader関数の代わりに使用する
    writer = make_writer(tuningName)#記録用のライター作成
    device = "cuda" if torch.cuda.is_available() else "cpu" #デバイス設定
    print(f"device：{device}\n")


    #========== 学習と検証 ==========#
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
    dataset = load_cifar10(transform=test_transform, train=True) #訓練・検証用データが一緒なのでなのでtransformしない
   
    # データ数とデータ形状を確認
    print(f'data size : {len(dataset)}')
    print(f'data shape : {dataset[0][0].shape}\n')
    # 学習用(Train)と検証用(Validation)への分割数を決める
    valid_ratio = 0.1                               # 検証用データの割合を指定
    valid_size = int(len(dataset) * valid_ratio)    # 検証用データの数
    train_size = len(dataset) - valid_size          # 学習用データ = 全体 - 検証用データ
    print(f'train samples : {train_size}')
    print(f'valid samples : {valid_size}\n')
    # 読み込んだデータをランダムに選んで分割
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_dataset.dataset.transform = train_transform #訓練用のデータセットにtransform適用


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


    # モデルの構築
    model = Classifier(class_num=10, enc_dim=128, in_w=32, in_h=32)
    # 計算に使用するデバイスへモデルを転送
    model.to(device)
    # 重みの初期化
    model.apply(initialize_weights)
    # 誤差関数(損失関数)を指定
    criterion = nn.CrossEntropyLoss()  # 交差エントロピー
    print("criterion")
    print(criterion)
    print("\n")
    # 最適化アルゴリズムを指定
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # 最適化関数Adam
    print("optimizer")
    print(optimizer)
    print("\n")


    # 学習過程を保存するためのリストを用意
    history = {
        'train_loss': [],
        'train_acc' : [],
        'valid_loss': [],
        'valid_acc' : []
    }

    # エポックの数だけ繰り返す
    for epoch in range(epochs):
        #========== 1エポックの処理 ==========#
         # プログレスバーの用意
        loop = tqdm(train_loader, unit='batch', desc='| Train | Epoch {:>3} |'.format(epoch+1))#loop=train_loader

        train_metrics = train_model(loop,model,criterion,optimizer,device)#訓練
        valid_metrics = valid_model(valid_loader,model,criterion,device)#検証

        train_loss = train_metrics['train_loss']
        valid_loss = valid_metrics['valid_loss']
        train_acc = train_metrics['train_acc']
        valid_acc = valid_metrics['valid_acc']

        # 学習過程を記録
        history['train_loss'].append(train_metrics['train_loss'])
        history['valid_loss'].append(valid_metrics['valid_loss'])
        history['train_acc'].append(train_metrics['train_acc'])
        history['valid_acc'].append(valid_metrics['valid_acc'])

        #学習過程をtensorboardに記録
        writer.add_scalars(f'{tuningName}/loss', {'train_loss':train_loss, 'valid_loss':valid_loss}, epoch)
        writer.add_scalars(f'{tuningName}/acc', {'train_acc':train_acc, 'valid_acc':valid_acc}, epoch) 

        print(f"| Train | Epoch   {epoch+1} |: train_loss:{train_loss:.3f}, train_acc:{train_acc*100:3.3f}% | valid_loss:{valid_loss:.5f}, valid_acc:{valid_acc*100:3.3f}%")

    print('Finished Training\n')

    #========== テスト ==========#
    test_dataset = load_cifar10(transform=test_transform, train=False)

    test_loader = fix_seed_dataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    print(f"test : {len(test_dataset):5.0f} [set]")

    loop = tqdm(test_loader, unit='batch', desc='| Test | Epoch {:>3} |'.format(epoch+1))
    test_metrics = test_model(loop,model,criterion,device)

    test_loss = test_metrics['test_loss']
    test_acc = test_metrics['test_acc']

    print("テストデータに対する結果")
    test_loss_str = f"test_loss   ：{test_loss:3.5f}"
    print(test_loss_str)
    test_acc_str = f"test_acc    ：{test_acc*100:2.3f}%"
    print(test_acc_str)

    writer.add_text(f"{tuningName}/test_loss",str(test_loss),0)
    writer.add_text(f"{tuningName}/test_acc",str(test_acc),1)
    writer.close()



if __name__ == '__main__':
    main()




