from keras import Model
import tensorflow as tf

class AutoEncoder(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
if __name__ == '__main__':
    print(tf.config.list_logical_devices())