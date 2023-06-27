def main():
    from sklearn.metrics import average_precision_score

    # モデルの出力と真のラベルを準備する
    y_true = [0, 1, 1, 0, 1]
    y_scores = [0.1, 0.8, 0.6, 0.3, 0.9]
    y_true_1 = [1, 0, 1, 0, 1]
    y_scores_1 = [0.8, 0.1, 0.6, 0.3, 0.9]

    # average_precision_scoreを計算する
    score = average_precision_score(y_true_1, y_scores)
    print("Average Precision Score:", score)
2
if __name__ == '__main__':
    main()