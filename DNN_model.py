import tensorflow as tf


class DNN(tf.keras.Model):
    def __init__(self, inputs_shape, layer_num_list, batch_size):
        super().__init__()
        self.inputs_shape = inputs_shape
        self.layer_num_list = layer_num_list
        self.batch_size = batch_size
        self.flat_layer = tf.keras.layers.Flatten(input_shape=(28, 28))

    def origin_build(self):
        layer_list = []
        bias_list = []
        for index, layer_num in enumerate(self.layer_num_list):
            bias = self.add_weight(shape=(layer_num,),
                                   initializer=tf.random_uniform_initializer(), trainable=True)
            bias_list.append(bias)
            if index == 0:
                layer = self.add_weight(shape=(self.inputs_shape, layer_num),
                                        initializer=tf.random_uniform_initializer(), trainable=True)
                layer_list.append(layer)
                continue
            layer = self.add_weight(
                shape=(self.layer_num_list[index - 1], layer_num), trainable=True)
            layer_list.append(layer)
        self.layer_list = layer_list
        self.bias_list = bias_list

    def call(self, inputs):
        inputs = self.flat_layer(inputs)
        temp = tf.matmul(inputs, self.layer_list[0])
        temp = tf.add(temp, self.bias_list[0])
        for index, layer in enumerate(self.layer_list):
            if index == 0:
                continue
            temp = tf.matmul(temp, layer)
            temp = tf.add(temp, self.bias_list[index])
            temp = tf.nn.sigmoid(temp)
        return tf.nn.softmax(temp, axis=-1)

    def calc_loss(self, target, predict):
        return tf.nn.softmax_cross_entropy_with_logits(target, predict)

    def update(self, gradient, optimizer):
        optimizer.apply_gradients(
            zip(gradient, self.trainable_variables))

    def predict(self, inputs):
        return tf.argmax(self.call(inputs), axis=-1)

# class DNN():
#     def __init__(self, inputs_shape, layer_num_list, batch_size):
#         super().__init__()
#         self.inputs_shape = inputs_shape
#         self.layer_num_list = layer_num_list
#         self.batch_size = batch_size
#         self.flat_layer = tf.keras.layers.Flatten(input_shape=(28, 28))

#     def origin_build(self):
#         layer_list = []
#         bias_list = []
#         for index, layer_num in enumerate(self.layer_num_list):
#             bias = tf.Variable(shape=(layer_num,), dtype=tf.float32,
#                                trainable=True, initial_value=tf.ones(shape=(layer_num,), dtype=tf.float32))
#             bias_list.append(bias)
#             if index == 0:
#                 layer = tf.Variable(shape=(
#                     self.inputs_shape, layer_num), dtype=tf.float32,
#                     trainable=True, initial_value=tf.ones(shape=(self.inputs_shape, layer_num), dtype=tf.float32))
#                 layer_list.append(layer)
#                 continue
#             layer = tf.Variable(shape=(
#                 self.layer_num_list[index - 1], layer_num),
#                 trainable=True, dtype=tf.float32,
#                 initial_value=tf.ones(shape=(self.layer_num_list[index - 1], layer_num), dtype=tf.float32))
#             layer_list.append(layer)
#         self.layer_list = layer_list
#         self.bias_list = bias_list

#     def call(self, inputs):
#         inputs = self.flat_layer(inputs)
#         temp = tf.matmul(inputs, self.layer_list[0])
#         temp = tf.add(temp, self.bias_list[0])
#         for index, layer in enumerate(self.layer_list):
#             if index == 0:
#                 continue
#             temp = tf.matmul(temp, layer)
#             temp = tf.add(temp, self.bias_list[index])
#             temp = tf.nn.sigmoid(temp)
#         return tf.nn.softmax(temp, axis=-1)

#     def calc_loss(self, target, predict):
#         return tf.nn.softmax_cross_entropy_with_logits(target, predict)

#     def update(self, gradient, optimizer):
#         optimizer.apply_gradients(
#             zip(gradient, self.trainable_variables))

#     def predict(self, inputs):
#         return tf.argmax(self.call(inputs), axis=-1)
