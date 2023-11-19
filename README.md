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

## Future Change
### 1.3.0 ディレクトリ構成が見直されます。
- 機械学習フローが`flows`ディレクトリに統一されます。
- 機械学習に直接関わりのないPythonファイルは`utils`ディレクトリに統一されます。
- `ex_*.py`ファイルは、`archive`ディレクトリに移動されます。
- 全ての実験プログラムは、`main.py`から呼び出されます。
- 全ての実験パラメータと設定は、`main.yml`に記述します。