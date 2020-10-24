from tensorflow.keras.datasets import mnist
import tensorflow as tf
from DNN_model import DNN
import numpy as np

EPOCH_NUM = 100
BATCH_SIZE = 1024

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

n_labels = len(np.unique(train_labels))
train_labels = np.eye(n_labels)[train_labels]

layer_num_list = [1000, 1000, 1000, 10]

model = DNN(28*28, layer_num_list, BATCH_SIZE)
model.origin_build()
optimizer = tf.optimizers.Adam(lr=0.0001)
model.compile(optimizer)
for e in range(EPOCH_NUM):
    for _ in range(40):
        train = train_images[_*BATCH_SIZE: (_+1)*BATCH_SIZE]
        label = train_labels[_*BATCH_SIZE: (_+1)*BATCH_SIZE]
        with tf.GradientTape() as tape:

            predict = model.call(train)
            loss = model.calc_loss(label, predict)

            # 微分
            gradient = tape.gradient(
                loss, model.trainable_variables)

            # NNを更新する
            optimizer.apply_gradients(
                zip(gradient, model.trainable_variables))
    predict = model.predict(test_images)
    print(f'正答率：{sum(predict.numpy()==np.array(test_labels)) / len(test_labels)}')
