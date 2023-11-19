
## How to write a commit message
- 🎨 `:art:`: UIやスタイルファイルの更新
- ⚡️ `:zap:`: パフォーマンス改善
- 🐛 `:bug:`: バグ修正
- 📝 `:memo:`: ドキュメンテーションの追加や更新
- 🚀 `:rocket:`: 新機能の追加
- 🚧 `:construction:`: 作業中

## Change Log

### 1.2.0　💥 BREAKING CHANGE
- 機械学習フローのファイルを分かりやすく、ファイル名を`base_model.py`->`base_flow.py`に、クラス名を`BaseModel`->`BaseFlow`に変更
- データセットごとにフローを走らせる従来の方法から、モデルごとのデフォルトフローを用いる方法に変更
したがって、これまでの`ex_*.py`プログラムは`DEPRECATED`となります

