from tensorflow.keras.datasets import mnist
from tensorflow import keras

ACTIVATION = "relu"

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0

test_images = test_images / 255.0
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(1000, activation=ACTIVATION),
    keras.layers.Dense(1000, activation=ACTIVATION),
    keras.layers.Dense(1000, activation=ACTIVATION),
    keras.layers.Dense(1000, activation=ACTIVATION),
    keras.layers.Dense(1000, activation=ACTIVATION),
    keras.layers.Dense(1000, activation=ACTIVATION),
    keras.layers.Dense(1000, activation=ACTIVATION),
    keras.layers.Dense(1000, activation=ACTIVATION),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, validation_data=(
    test_images, test_labels), epochs=10, batch_size=1024)
