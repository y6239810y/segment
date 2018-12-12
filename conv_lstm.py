# -*- coding:utf-8 -*-

from tools import *


class SegLSTMCell(object):
    def __init__(self, hidden_num, filter_size=[3, 3],
                 forget_bias=1.0, activation=tf.tanh, is_train=True,name="ConvLSTMCell"):
        self.hidden_num = hidden_num  # 就是经过分割之后输出通道数
        self.filter_size = filter_size  # 卷积核大小
        self.forget_bias = forget_bias  # 遗忘门的偏置
        self.activation = activation  # 需要使用的激活函数
        self.name = name
        self.is_train = is_train

    def zero_state(self, batch_size, height, width):
        return tf.zeros([batch_size, height, width, self.hidden_num * 2])  # ×2的原因是 这里合并了c(记忆输出)和h（单元输出）

    def __call__(self, inputs, state, scope=None):
        """Convolutional Long short-term memory cell (ConvLSTM)."""
        with tf.variable_scope(scope or self.name):  # "ConvLSTMCell"

            c, h = tf.split(value=state, axis=3, num_or_size_splits=2)
            # state.shape = batch_size * height * width * channel
            concat = seg_conv([inputs, h], 4 * self.hidden_num, self.filter_size,is_train=self.is_train)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=3)

            new_c = (c * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) *
                     self.activation(j))
            new_h = self.activation(new_c) * tf.sigmoid(o)
            new_state = tf.concat([new_c, new_h],3)    # 返回一个tensor

            return new_h, new_state


def seg_conv(args, output_size, filter_size, is_train):

    input = tf.concat(args,3)
    with tf.variable_scope('segment'):
        net = {}
        layers = [
            'CONV_LAST'
        ]
        layers_kernels = [
            {"kernel": filter_size, "stride": 1, "filter": output_size, "BN": True}  # CONV_LAST
        ]
        current = input
        for name, value in zip(layers, layers_kernels):
            if name[:4] == 'CONV':
                with tf.variable_scope(name):
                    conv_kernel = value['kernel']
                    conv_stride = value['stride']
                    conv_filter = value['filter']
                    is_bn = value['BN']
                    if (is_bn):
                        current = tf.contrib.slim.batch_norm(current, is_training=is_train, scope='BN')
                        current = Relu(current, name='RELU')
                    current = Conv2d(input=current, filter=conv_filter, kernel=conv_kernel, strides=conv_stride)
                    net[name] = current

            elif name[:-2] == 'POOL':
                with tf.variable_scope(name):
                    pool_way = value['pool_way']
                    pool_kernel = value['kernel']
                    pool_stride = value['stride']
                    current = pool_way(x=current, pool_size=pool_kernel, strides=pool_stride)
                net[name] = current

            elif name[:3] == 'RES':
                num = value['num']
                filter = value['filter']
                kernel = value['kernel']
                stride = value['stride']
                current = ResBlock(current, num, kernel, filter,stride,is_train, name)
                net[name] = current

            elif name[:-2] == 'UPSAMPLE':
                up_kernel = value['kernel']
                up_stride = value['stride']
                up_filter = value['filter']
                with tf.variable_scope(name):
                    current = Upsample_2d(input=current, kernel=up_kernel, filter=up_filter, strides=up_stride)
                net[name] = current

            elif name[:-2] == 'CONBINE':
                with tf.variable_scope(name):
                    layer_name = value['add_layer']
                    layer = net[layer_name]
                    current = tf.concat([current, layer], 3)
                net[name] = current
            elif name[:-2] == 'ADD':
                with tf.variable_scope(name):
                    w = tf.Variable(initial_value=[1], dtype=tf.float16)
                    layer_name = value['add_layer']
                    kernel = value['kernel']
                    layer = net[layer_name]
                    add_tensor = tf.contrib.slim.batch_norm(layer, is_training=is_train, scope='BN')
                    add_tensor = Relu(add_tensor, name='RELU')
                    add_tensor = Conv2d(add_tensor, filter=current.shape[-1], kernel=kernel)
                    current = current + tf.multiply(w, add_tensor)
                net[name] = current

    return net['CONV_LAST']


def ResBlock(input, nums, kernel, filter, stride,is_train, name):
    add_layer = Conv2d(input=input, filter=filter, strides=stride, kernel=[1, 1], layer_name='ADD_CONV')
    output = input
    with tf.variable_scope(name):
        for i in range(1, nums + 1):
            if i == 1:
                strides = stride
            else:
                strides = 1
            with tf.variable_scope('Bottleneck' + str(i)):
                BN_1 = tf.contrib.slim.batch_norm(output, is_training=is_train, scope='BN_1')
                x_1 = Relu(BN_1)
                conv_1 = Conv2d(input=x_1, strides=strides, filter=filter, kernel=kernel, layer_name='CONV_1')

                BN_2 = tf.contrib.slim.batch_norm(conv_1, is_training=is_train, scope='BN_2')
                x_2 = Relu(BN_2)
                conv = Conv2d(input=x_2, filter=filter, kernel=kernel, layer_name='CONV_2')

                output = add_layer + conv
                add_layer = output
    return output