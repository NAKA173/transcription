# Audio to MIDI セットアップガイド

新しいターミナルを開いたら、これだけ打てばOKです。

---

## クイックスタート（毎回これだけ）

```bash
cd transcription
bash start.sh
```

ブラウザで http://localhost:8000 を開く。以上。

---

## 初回セットアップ（最初の1回だけ）

### 1. Python 3.11以上を確認

```bash
python3 --version
```

`Python 3.11.x` 以上が表示されればOK。
なければ https://www.python.org/downloads/ からインストール。

### 2. リポジトリを取得

```bash
git clone https://github.com/naka173/transcription.git
cd transcription
```

### 3. 起動

```bash
bash start.sh
```

初回は依存パッケージのインストールに数分かかります。
2回目以降は数秒で起動します。

---

## Windows の場合

`start.bat` をダブルクリック。
または PowerShell / コマンドプロンプトで:

```
cd transcription
start.bat
```

---

## トラブルシューティング

### `ERR_ALPN_NEGOTIATION_FAILED` が出る

ブラウザのアドレスバーが `https://` になっている。
`http://localhost:8000` に変更する（httpの方）。

### `pip: command not found`

```bash
python3 -m pip install --upgrade pip
```

### `Permission denied` / `externally-managed-environment`

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
bash start.sh
```

### ポートが使用中

```bash
bash start.sh 8080
```

別のポート番号でブラウザを `http://localhost:8080` で開く。

### 楽器分離（Demucs）を使いたい

楽器分離はオプション機能です。使う場合は追加インストール:

```bash
pip install "transcription[separation]"
```

> PyTorch + Demucs で約2GBダウンロードされます。
> GPUがあれば自動で使われます。

---

## ファイル構成（参考）

```
transcription/
├── start.sh          ← Mac/Linux 起動スクリプト
├── start.bat         ← Windows 起動スクリプト
├── src/transcription/ ← アプリ本体
│   ├── main.py       ← Webサーバー
│   ├── transcriber.py ← MIDI変換エンジン
│   ├── preprocess.py  ← 音声前処理
│   ├── rhythm.py      ← リズム・拍子検出
│   ├── separator.py   ← 楽器分離(Demucs)
│   ├── scoring.py     ← 精度スコアリング
│   └── static/        ← WebUI
└── tests/            ← テスト
```
