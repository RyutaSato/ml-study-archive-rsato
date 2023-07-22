x_train = []
x_test = []
import tensorflow as tf
import keras
from keras.layers import Dense
def main():
    model = keras.Sequential([
        Dense(units=19, activation='relu', input_dim=38, name='encoder1'),
        Dense(units=10, activation='relu', name='encoder2'),
        Dense(units=19, activation='relu'),
        Dense(units=38, activation='relu'),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x_train, x_train,
              epochs=1,  # データセットを使って学習する回数
              batch_size=32,
              validation_data=(x_train, x_train),  # 評価用データ（検証データ）の指定
              )
    x_pred = model.predict(x_test) # モデルを使って実際に，予測


if __name__ == '__main__':
    main()
