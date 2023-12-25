## How to write commit message
- 🎨 `:art:`: UIやスタイルファイルの更新
- ⚡️ `:zap:`: パフォーマンス改善
- 🐛 `:bug:`: バグ修正
- 📝 `:memo:`: ドキュメンテーションの追加や更新
- 🚀 `:rocket:`: 新機能の追加
- 🚧 `:construction:`: 作業中

## Change Log

### 1.2.0　可読性のため、機械学習フローのクラス名を`*Model`から`*Flow`に変更しました。💥 BREAKING CHANGE
- 機械学習フローのファイルを分かりやすく、ファイル名を`base_model.py`->`base_flow.py`に、クラス名を`BaseModel`->`BaseFlow`に変更
- データセットごとにフローを走らせる従来の方法から、モデルごとのデフォルトフローを用いる方法に変更
したがって、これまでの`ex_*.py`プログラムは`DEPRECATED`となります

### 1.2.1 全ての実験は、`main.py`から実行されるように変更
- `ex_*.py`は全て`DEPRECATED`に変更
- :rocket: new feature `main.py`, `main.yml`, `_main.py`

### 1.2.2 視覚化用ユーティリティを追加

- :rocket: new feature `visualize_utils.py`

### 1.2.3 LightGBMモデルを使用した場合にハイパーパラメータチューニングを行うよう変更 💥 BREAKING CHANGE
- 1.2.3 <= VERSION < 1.3.0: LightGBMモデルは、`LightGBM+optuna`として動作します。

### 1.2.4 親プロセスが、子プロセスの例外をcatchするように変更　　💥 BREAKING CHANGE
- :rocket: new feature `imb_data.py` Imbalanced Datasetを新たにデータセットに追加
- :zap: 標準化のタイミングを変更　#5

### 1.2.5 :bug: Fix #3 配列サイズの不一致が発生する問題の修正 
- バグの発生箇所は不明、原因は参照コピー
- 以前の結果において、別のプロセスの結果を参照している可能性　💥 BREAKING CHANGE

### 1.3.0 　LightGBMモデルの設定を変更💥 BREAKING CHANGE
- LightGBM-> LightGBM+optunaに改名
- 従来のハイパーパラメータチューニングをしていないLightGBMを`LightGBM`として定k技
- Flowインスタンスをタスク実行終了後に明示的に削除
- モデルのデフォルトパラメータを明示化

### 1.4.0 前処理に標準化ではなく正規化を採用　💥 BREAKING CHANGE
- 一般的にニューラルネットワークには、前処理として正規化を採用されている背景によるもの
- 標準化のコードを削除

### 1.4.1 
- `ex_*.py`ファイルは、`archive`ディレクトリに移動されます。

## Future Change

### ディレクトリ構成の変更
- 機械学習フローが`flows`ディレクトリに統一されます。
- 機械学習に直接関わりのないPythonファイルは`utils`ディレクトリに統一されます。

### エラーハンドリングに関する追加予定機能

- 失敗した並列プロセスのパラメータは`error_params.json`に保存されます。
- Git push前に実行するテストコードが追加されます。

### MultiProcessingに関する変更

### TODO


