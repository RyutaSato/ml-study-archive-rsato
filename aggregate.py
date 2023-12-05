import pandas as pd
DATASET = 'creditcardfraud'

# CSVファイルを読み込む
data = pd.read_csv(f'logs/{DATASET}.csv')

# 'model_name'が'RandomForest'の行だけを抽出し、コピーを作成
rf_data = data[data['model_name'] == 'RandomForest'].copy()

# 'importances.'で始まるカラム名を抽出
importance_columns = [col for col in data.columns if col.startswith('importances.')]
# 各カラムの値をソートし、最大の10個の値を取得
top_importances = {}
for col in importance_columns:
    top_importances[col] = rf_data[col].sort_values(ascending=False)[:10]
for i, (col, values) in enumerate(sorted(top_importances.items(), key=lambda item: item[1].max(), reverse=True)[:10]):
    rf_data.loc[:, f'{i+1}st_importance'] = col
# # 抽出するカラムを指定する
selected_columns = ['model_name', 
                    'version',
                    'encoder_param.layers', 
                    'dataset.ae_used_data',
                    'result.anomaly.f1-score',
                    'result.macro avg.f1-score',
                    *[f'{i+1}st_importance' for i in range(10)]
]

# 結果を保存
rf_data[selected_columns].to_csv(f'logs/filtered_{DATASET}_importances.csv', index=False)


# 新しいデータフレームをCSVファイルとして保存する
# new_data.to_csv(f'logs/filtered_{DATASET}.csv', index=False)

