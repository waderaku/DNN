from tensorflow.keras.datasets import cifar10
from tensorflow import keras
import tensorflow as tf

from CNN_model import CNN

(train_images, train_labels), (test_images,
                               test_labels) = cifar10.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

cnn = CNN()

optimizer = tf.optimizers.Adam(lr=0.0001)
cnn.compile(optimizer)

cnn(train_images[0:1000])

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
# train_images = tf.image.rgb_to_grayscale(train_images)
# test_images = tf.image.rgb_to_grayscale(test_images)

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(32, 32)),
#     keras.layers.Dense(1000, activation=ACTIVATION),
#     keras.layers.Dense(1000, activation=ACTIVATION),
#     keras.layers.Dense(1000, activation=ACTIVATION),
#     keras.layers.Dense(1000, activation=ACTIVATION),
#     keras.layers.Dense(10, activation='softmax')
# ])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, validation_data=(
    test_images, test_labels), epochs=50, batch_size=1024)
