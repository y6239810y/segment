from tensorflow.python.training.moving_averages import assign_moving_average
import tensorflow as tf
import numpy as np

def Relu(x,name='relu'):
    return tf.nn.relu(x,name=name)

def Conv3d(input, filter, kernel, strides=1, layer_name="conv",activation=None):
    with tf.name_scope(layer_name):
        network = tf.layers.conv3d(inputs=input, use_bias=True, filters=filter, activation=activation, kernel_size=kernel, strides=strides,
                                   padding='SAME')
        return network

def Conv2d(input, filter, kernel, strides=1, layer_name="conv",activation=None):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter,activation=activation, kernel_size=kernel, strides=strides,
                                   padding='SAME')
        return network


def Upsample_3d(input, filter, kernel, strides=[2,2,2], layer_name="upsample",activation=None):
    with tf.name_scope(layer_name):
        network = tf.layers.conv3d_transpose(inputs=input, use_bias=False, filters=filter,activation=activation, kernel_size=kernel, strides=strides,
                                   padding='SAME')
        return network

def Upsample_2d(input, filter, kernel, strides=[2,2], layer_name="upsample",activation=None):
    with tf.name_scope(layer_name):
        network = tf.layers.conv3d_transpose(inputs=input, use_bias=True, filters=filter,activation=activation, kernel_size=kernel, strides=strides,
                                   padding='SAME')
        return network

def Average_pooling_2d(x, pool_size=[2,2], strides=2, padding='SAME',name='avg_pool'):
    with tf.variable_scope(name):
        return tf.layers.average_pooling3d(inputs=x, pool_size=pool_size, strides=strides, padding=padding,name=name)


def Max_Pooling_2d(x, pool_size=[3,3], strides=2, padding='SAME',name='max_pool'):
    with tf.variable_scope(name):
        return tf.layers.average_pooling3d(inputs=x, pool_size=pool_size, strides=strides, padding=padding,name=name)


def Average_pooling_3d(x, pool_size=[2,2,2], strides=2, padding='SAME',name='avg_pool'):
    with tf.variable_scope(name):
        return tf.layers.average_pooling3d(inputs=x, pool_size=pool_size, strides=strides, padding=padding,name=name)

def Max_Pooling_3d(x, pool_size=[3,3,3], strides=2, padding='SAME',name='max_pool'):
    with tf.variable_scope(name):
        return tf.layers.average_pooling3d(inputs=x, pool_size=pool_size, strides=strides, padding=padding,name=name)


def global_avg_pool(input, name="GlobalAvgPool"):
    shape = input.shape
    if len(shape)== 4:
        with tf.name_scope(name):
            inference = tf.reduce_mean(input, [1, 2])
    elif len(shape)== 5:
        with tf.name_scope(name):
            inference = tf.reduce_mean(input, [1, 2, 3])
    return inference

def Batch_norm(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = x.shape[-1:]
        moving_mean = tf.get_variable('mean', params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, tf.shape(x)[:-1], name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x

def Batch_norm_layer(x, train_phase, scope_bn):
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        axises = np.arange(len(x.shape) - 1)
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed