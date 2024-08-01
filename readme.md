# 環境構築と実行方法
[pyenv](https://github.com/pyenv/pyenv)と[venv](https://docs.python.org/ja/3/library/venv.html)を使用する方法で説明します。
OSはWindowsを想定しています。

## CPU実行
```
pyenv install 3.12.4
pyenv local 3.12.4
python3 -m venv .venv
.venv\Scripts\activate
pip install transformers torch accelerate
python3 usage.py
```

## GPU実行
NVIDIAのGPUを搭載していればGPUを使って実行できます。

CUDAの[ダウンロードサイト](https://developer.nvidia.com/cuda-toolkit-archive)からCUDA Toolkitをダウンロードしてインストールしてください。  
※2024/08/01現在 pytorch は12.4までしか対応していないので、12.4.*をダウンロードしてください。

cuDNNのインストールも必要ですので、インストールしたcudaのバージョンに合わせて、[こちら](https://developer.nvidia.com/rdp/cudnn-archive)からダウンロードし、インストールしてください。

```
pyenv install 3.12.4
pyenv local 3.12.4
python3 -m venv .venv
.venv\Scripts\activate
pip install transformers torch --index-url https://download.pytorch.org/whl/cu124
pip install accelerate
python3 usage.py
```
※`https://download.pytorch.org/whl/cu124`の部分は[こちら](https://pytorch.org)の**INSTALL PYTORCH**を参考に、インストールしたcudaのバージョンに合わせてください。

## チャットについて
```
python3 chat.py
```
を実行することで、会話できます。

会話中に`exit`と入力すると会話が終了します。

# 実行エラーについて
実行時に下記のエラーが発生した場合は`libomp140.x86_64.dll`が足りないようです。
https://www.dllme.com/dll/files/libomp140_x86_64 からダウンロードしたDLLをPATHの通った場所(`.venv\Scripts`で良いと思う)に置くと解決します。  
※`dll`フォルダにダウンロードしたファイルを置いてあります。

```
OSError: [WinError 126] 指定されたモジュールが見つかりません。 Error loading "***\.venv\Lib\site-packages\torch\lib\fbgemm.dll" or one of its dependencies.
```

参考: https://discuss.pytorch.org/t/failed-to-import-pytorch-fbgemm-dll-or-one-of-its-dependencies-is-missing/201969
