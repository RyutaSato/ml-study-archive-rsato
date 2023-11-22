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

## Future Change

### ディレクトリ構成の変更
- 機械学習フローが`flows`ディレクトリに統一されます。
- 機械学習に直接関わりのないPythonファイルは`utils`ディレクトリに統一されます。
- `ex_*.py`ファイルは、`archive`ディレクトリに移動されます。
- 全ての実験プログラムは、`main.py`から呼び出されます。
- 全ての実験パラメータと設定は、`main.yml`に記述します。

### エラーハンドリングに関する追加予定機能

- 失敗した並列プロセスのパラメータは`error_params.json`に保存されます。
- 実験プロセスの実行前に、`main.yml`のvalidationが`_main.py`に追加されます。
- Git push前に実行するテストコードが追加されます。

### TODO
- 勾配ブースティングを使う
- 精度のでるデータセットを探す


### 確認されている不具合
- マルチプロセス使用時に、子プロセス終了後にGPUメモリが解放されない
```shell
Traceback (most recent call last):
  File "C:\Users\rsato\anaconda3\envs\ml\lib\concurrent\futures\process.py", line 246, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
  File "E:\ml-study-archive-rsato\base_flow.py", line 318, in run
    raise e
  File "E:\ml-study-archive-rsato\base_flow.py", line 310, in run
    self.train_and_predict()
  File "E:\ml-study-archive-rsato\base_flow.py", line 192, in train_and_predict
    _encoder.predict(x_train, verbose=0),  # type: ignore
  File "C:\Users\rsato\anaconda3\envs\ml\lib\site-packages\keras\utils\traceback_utils.py", line 70, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "C:\Users\rsato\anaconda3\envs\ml\lib\site-packages\tensorflow\python\eager\execute.py", line 54, in quick_execute
    tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
tensorflow.python.framework.errors_impl.ResourceExhaustedError: Graph execution error:

SameWorkerRecvDone unable to allocate output tensor. Key: /job:localhost/replica:0/task:0/device:CPU:0;899591e1dcfebd12;/job:localhost/replica:0/task:0/device:GPU:0;edge_11_IteratorGetNext;0:0
	 [[{{node IteratorGetNext/_2}}]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info. This isn't available when running in Eager mode.
 [Op:__inference_predict_function_819156]

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "E:\ml-study-archive-rsato\main.py", line 40, in <module>
    main()
  File "E:\ml-study-archive-rsato\main.py", line 36, in main
    logger.info(f"{k} is done: {futures[k].result()}")
  File "C:\Users\rsato\anaconda3\envs\ml\lib\concurrent\futures\_base.py", line 446, in result
    return self.__get_result()
  File "C:\Users\rsato\anaconda3\envs\ml\lib\concurrent\futures\_base.py", line 391, in __get_result
    raise self._exception
tensorflow.python.framework.errors_impl.ResourceExhaustedError: Graph execution error:

SameWorkerRecvDone unable to allocate output tensor. Key: /job:localhost/replica:0/task:0/device:CPU:0;899591e1dcfebd12;/job:localhost/replica:0/task:0/device:GPU:0;edge_11_IteratorGetNext;0:0
	 [[{{node IteratorGetNext/_2}}]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info. This isn't available when running in Eager mode.
 [Op:__inference_predict_function_819156]
```
> [!NOTE] 解決策1
>> GPU計算用プロセスを１つのみ用意し、Queueで計算メソッドを取得する。それ以外のプロセスは、CPUのみで演算する。

> 解決策2
>> 各`executor`を実行後に、`del`を呼び出す。

> 解決策3
>> プロセス数を固定し、Queueに計算メソッドを投げ、取得したプロセスが実行するように変更する。
>> その際に、各プロセスは、破壊的メソッドで内部を定義すること！

- 不定期に配列サイズの不一致エラーが発生する。(原因不明)
```sh
Shape of passed values is (a1, b1), indices imply (a2, b2)
```