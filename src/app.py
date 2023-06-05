import torch
from yakiniku import transform, Net # yakiniku.py から前処理とネットワークの定義を読み込み
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64
from flask import url_for

# 学習済みモデルをもとに推論する
def predict(img):
    # ネットワークの準備
    net = Net().cpu().eval()
    # # 学習済みモデルの重み（yakiniku.pt）を読み込み
    net.load_state_dict(torch.load('./src/yakiniku.pt', map_location=torch.device('cpu')))
    #　データの前処理
    img = transform(img)
    img =img.unsqueeze(0) # 1次元増やす
    #　推論
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y

#　推論したラベルからyakinikuの部位のイラストを返す関数
def getName(label):
    if label==0:
        return 'ハラミ: HARAMI'
    elif label==1:
        return 'ハツ: HATSU'
    elif label==2:
        return 'イチボ: ICHIBO'
    elif label==3:
        return 'カルビ: KARUBI'
    elif label==4:
        return 'レバー: LIVER'
    elif label==5:
        return 'マルチョウ: MARUCHO'
    elif label==6:
        return 'ミノ: MINO'
    elif label==7:
        return 'センマイ: SENMAI'
    elif label==8:
        return 'シマチョウ: SHIMACHO'
    elif label==9:
        return 'サーロイン: SIRLOIN'
    elif label==10:
        return 'タン: TONGUE'
 
#　推論したラベルからyakinikuの部位のイラストを返す関数
def getIllustration(label):
    if label==0:
        return '部位図ハラミ.png'
    elif label==1:
        return '部位図ハツ.png'
    elif label==2:
        return '部位図イチボ.png'
    elif label==3:
        return '部位図カルビ.png'
    elif label==4:
        return '部位図レバー.png'
    elif label==5:
        return '部位図マルチョウ.png'
    elif label==6:
        return '部位図ミノ.png'
    elif label==7:
        return '部位図センマイ.png'
    elif label==8:
        return '部位図シマチョウ.png'
    elif label==9:
        return '部位図サーロイン.png'
    elif label==10:
        return '部位図タン.png'


# Flask のインスタンスを作成
app = Flask(__name__, static_folder='static')

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

#　拡張子が適切かどうかをチェック
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'filename' not in request.files:
            return redirect(request.url)
        # データの取り出し
        file = request.files['filename']
        # ファイルのチェック
        if file and allowed_file(file.filename):

            #　画像ファイルに対する処理
            #　画像書き込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')
            #　画像データをバッファに書き込む
            image.save(buf, 'png')
            #　バイナリデータを base64 でエンコードして utf-8 でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            #　HTML 側の src の記述に合わせるために付帯情報付与する
            base64_data = 'data:image/png;base64,{}'.format(base64_str)

            # 入力された画像に対して推論
            pred = predict(image)
            yakinikuName_ = getName(pred)
            yakinikuIllustration_ = getIllustration(pred)
            return render_template('result.html', yakinikuName=yakinikuName_, yakinikuIllustration=yakinikuIllustration_, image=base64_data)
        return redirect(request.url)

    # GET メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')
    

# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)