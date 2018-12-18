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
            {"kernel": filter_size, "stride": 1, "filter": output_size, "norm":'group'}  # CONV_LAST
        ]
        current = input
        for name, value in zip(layers, layers_kernels):
            if name[:4] == 'CONV':
                with tf.variable_scope(name):
                    conv_kernel = value['kernel']
                    conv_stride = value['stride']
                    conv_filter = value['filter']
                    norm_type = value['norm']
                    current = norm(current,norm_type=norm_type, is_train=is_train, scope='NORM')
                    current = Relu(current, name='RELU')

                    current = Conv2d(input=current, filter=conv_filter, kernel=conv_kernel, strides=conv_stride)
                    net[name] = current

    return net['CONV_LAST']

