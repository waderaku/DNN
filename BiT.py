import tensorflow as tf
import tensorflow_hub as hub

import tensorflow_datasets as tfds
from PIL import Image
import requests
from io import BytesIO
from BiT_model import BigTransfer
from tensorflow.keras.datasets import cifar10
import numpy as np
# パラメータ一覧
DATA_DIR = "C:\\tmp\\bit_image"

MODEL_DIR = '/tmp/my_saved_bit_model/'

TRAIN_SPLIT = 0.9

NUM_CLASSES = 10

RESIZE_TO = 160  # 512
CROP_TO = 128  # 480

SCHEDULE_LENGTH = 500  # 10000, 20000
# [3000, 6000, 9000] [6000, 12000, 18000]
SCHEDULE_BOUNDARIES = [200, 300, 400]

BATCH_SIZE = 512
STEPS_PER_EPOCH = 10

EPOCHS_NUM = 10

MODEL_URL = "https://tfhub.dev/google/bit/m-r50x1/1"


def preprocess_image(image):
    image = np.array(image)
    img_reshaped = tf.reshape(
        image, [1, image.shape[0], image.shape[1], image.shape[2]])
    image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
    return image


def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = preprocess_image(image)
    return image

# データの前処理パイプライン
# 論文にこれやってって書いてあったので従っておいたほうがよさそう


def cast_to_tuple(features):
    return (features['image'], features['label'])


def preprocess_train(features):
    features['image'] = tf.image.random_flip_left_right(features['image'])
    features['image'] = tf.image.resize(
        features['image'], [RESIZE_TO, RESIZE_TO])
    features['image'] = tf.image.random_crop(
        features['image'], [CROP_TO, CROP_TO, 3])
    features['image'] = tf.cast(features['image'], tf.float32) / 255.0
    return features


def preprocess_test(features):
    features['image'] = tf.image.resize(
        features['image'], [RESIZE_TO, RESIZE_TO])
    features['image'] = tf.cast(features['image'], tf.float32) / 255.0
    return features


if __name__ == "__main__":
    module = hub.KerasLayer(MODEL_URL)
    model = BigTransfer(num_classes=NUM_CLASSES, module=module)
    (train_images, train_labels), (test_images,
                                   test_labels) = cifar10.load_data()

    dataset = tf.data.Dataset.from_tensor_slices(
        {'image': train_images, 'label': train_labels})

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset2 = dataset.prefetch(buffer_size=AUTOTUNE)

    num_train = int(len(train_images)*TRAIN_SPLIT)
    dataset2 = dataset2.shuffle(10000)
    ds_train = dataset2.take(num_train)
    ds_test = dataset2.skip(num_train)

    schedule_length = SCHEDULE_LENGTH * 512 / BATCH_SIZE
    pipeline_train = (ds_train
                      .shuffle(10000)
                      .repeat()
                      .map(preprocess_train, num_parallel_calls=8)
                      .batch(BATCH_SIZE)
                      .map(cast_to_tuple)
                      .prefetch(2))

    pipeline_test = (ds_test.map(preprocess_test, num_parallel_calls=1)
                     .map(cast_to_tuple)
                     .batch(BATCH_SIZE)
                     .prefetch(2))

    lr = 0.003 * BATCH_SIZE / 512

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=SCHEDULE_BOUNDARIES,
                                                                       values=[lr, lr*0.1, lr*0.001, lr*0.0001])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule, momentum=0.9)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['accuracy'])

    # これだと50エポックやると思うから少なくしてもいいと思う
    history = model.fit(
        pipeline_train,
        batch_size=BATCH_SIZE,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=int(SCHEDULE_LENGTH / STEPS_PER_EPOCH),  # EPOCHS_NUM
        validation_data=pipeline_test
    )
