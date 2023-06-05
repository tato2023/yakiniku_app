# 必要なモジュールのインポート
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn
#学習時に使ったのと同じ学習済みモデルをインポート
from torchvision.models import resnet34 

# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
    transforms.Resize(256),  # 画像のリサイズ
    transforms.CenterCrop(224),  # 画像の中心部分を切り抜き
    transforms.RandomHorizontalFlip(),  # ランダムな水平反転
    transforms.RandomRotation(10),  # ランダムな回転（-10度から+10度の範囲）
    transforms.RandomCrop(224),  # ランダムな位置で画像を切り抜き
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # ランダムな色の変化（明るさ、コントラスト、彩度）
    transforms.ToTensor(),  # テンソルに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正規化
])

#　ネットワークの定義
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        #学習時に使ったのと同じ学習済みモデルを定義
        self.feature = resnet34(pretrained=True) 
        self.fc = nn.Linear(1000, 11)

    def forward(self, x):
        #学習時に使ったのと同じ順伝播
        h = self.feature(x)
        h = self.fc(h)
        return h