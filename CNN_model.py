import tensorflow as tf
from tensorflow import keras


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.w1 = keras.layers.Conv2D(32, (3, 3), activation='relu',
                                      input_shape=(32, 32, 3))
        self.w2 = keras.layers.MaxPooling2D((2, 2))
        self.w3 = keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.w4 = keras.layers.MaxPooling2D((2, 2))
        self.w5 = keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.w6 = keras.layers.Flatten()
        self.w7 = keras.layers.Dense(64, activation='relu')
        self.w8 = keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        inputs = self.w1(inputs)
        inputs = self.w2(inputs)
        inputs = self.w3(inputs)
        inputs = self.w4(inputs)
        inputs = self.w5(inputs)
        inputs = self.w6(inputs)
        inputs = self.w7(inputs)
        inputs = self.w8(inputs)
        return inputs
