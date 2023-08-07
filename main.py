import pandas as pd
from sklearn.metrics import confusion_matrix


def create_confusion_matrix(y_true, y_pred, classes):
    # 多クラス分類の混同行列を計算
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # 混同行列をデータフレームに変換
    df_confusion_matrix = pd.DataFrame(cm, index=classes, columns=classes)

    return df_confusion_matrix

if __name__ == '__main__':

    # 例としてクラスを'A', 'B', 'C'とした場合の使用例
    # y_trueとy_predは実際のデータに置き換えてください

    # クラスのリストを定義
    classes = ['A', 'B', 'C']

    # 例として適当なy_trueとy_predを定義
    y_true = ['A', 'B', 'A', 'C', 'A', 'C', 'B', 'C']
    y_pred = ['A', 'B', 'B', 'C', 'A', 'B', 'B', 'A']

    # 混同行列を作成
    confusion_matrix_df = create_confusion_matrix(y_true, y_pred, classes)

    # 混同行列を表示
    print("Confusion Matrix:")
    print(confusion_matrix_df.to_csv())
