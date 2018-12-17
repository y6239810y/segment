from tensorflow.python.training.moving_averages import assign_moving_average
import tensorflow as tf
import numpy as np


def Relu(x, name='relu'):
    return tf.nn.relu(x, name=name)


def Conv3d(input, filter, kernel, strides=1, layer_name="conv", activation=None):
    with tf.name_scope(layer_name):
        network = tf.layers.conv3d(inputs=input, use_bias=True, filters=filter, activation=activation,
                                   kernel_size=kernel, strides=strides,
                                   padding='SAME')
        return network


def Conv2d(input, filter, kernel, strides=1, layer_name="conv", activation=None):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, activation=activation,
                                   kernel_size=kernel, strides=strides,
                                   padding='SAME')
        return network


def Depthwise_conv2d(input, filter, strides=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.nn.depthwise_conv2d(input=input, filter=filter, strides=[1, strides, strides, 1],
                                         padding='SAME')
        return network


def Upsample_3d(input, filter, kernel, strides=[2, 2, 2], layer_name="upsample", activation=None):
    with tf.name_scope(layer_name):
        network = tf.layers.conv3d_transpose(inputs=input, use_bias=False, filters=filter, activation=activation,
                                             kernel_size=kernel, strides=strides,
                                             padding='SAME')
        return network


def Upsample_2d(input, filter, kernel, strides=[2, 2], layer_name="upsample", activation=None):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d_transpose(inputs=input, use_bias=True, filters=filter, activation=activation,
                                             kernel_size=kernel, strides=strides,
                                             padding='SAME')
        return network


def Average_pooling_2d(x, pool_size=[2, 2], strides=2, padding='SAME', name='avg_pool'):
    with tf.variable_scope(name):
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=strides, padding=padding, name=name)


def Max_Pooling_2d(x, pool_size=[3, 3], strides=2, padding='SAME', name='max_pool'):
    with tf.variable_scope(name):
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=strides, padding=padding, name=name)


def Average_pooling_3d(x, pool_size=[2, 2, 2], strides=2, padding='SAME', name='avg_pool'):
    with tf.variable_scope(name):
        return tf.layers.average_pooling3d(inputs=x, pool_size=pool_size, strides=strides, padding=padding, name=name)


def Max_Pooling_3d(x, pool_size=[3, 3, 3], strides=2, padding='SAME', name='max_pool'):
    with tf.variable_scope(name):
        return tf.layers.average_pooling3d(inputs=x, pool_size=pool_size, strides=strides, padding=padding, name=name)


def global_avg_pool(input, name="GlobalAvgPool"):
    shape = input.shape
    if len(shape) == 4:
        with tf.name_scope(name):
            inference = tf.reduce_mean(input, [1, 2])
    elif len(shape) == 5:
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


def dice_coe(output, target, loss_type='jaccard', axis=(0, 1, 2), smooth=1e-5):
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    # old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    # new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice


def iou_coe(output, target, threshold=0.5, axis=(0, 1, 2), smooth=1e-5):
    """Non-differentiable Intersection over Union (IoU) for comparing the
    similarity of two batch of data, usually be used for evaluating binary image segmentation.
    The coefficient between 0 to 1, and 1 means totally match.

    Parameters
    -----------
    output : tensor
        A batch of distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        The target distribution, format the same with `output`.
    threshold : float
        The threshold value to be true.
    axis : tuple of integer
        All dimensions are reduced, default ``(1,2,3)``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.

    Notes
    ------
    - IoU cannot be used as training loss, people usually use dice coefficient for training, IoU and hard-dice for evaluating.

    """
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    # old axis=[0,1,2,3]
    # epsilon = 1e-5
    # batch_iou = inse / (union + epsilon)
    # new haodong
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou, name='iou_coe')
    return iou  # , pre, truth, inse, union


def dice_hard_coe(output, target, threshold=0.5, axis=(0, 1, 2), smooth=1e-5):
    """Non-differentiable Sørensen–Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation i.e. labels are binary.
    The coefficient between 0 to 1, 1 if totally match.

    Parameters
    -----------
    output : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        The target distribution, format the same with `output`.
    threshold : float
        The threshold value to be true.
    axis : tuple of integer
        All dimensions are reduced, default ``(1,2,3)``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    output = tf.cast(output > threshold, dtype=tf.float32)
    target = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)
    # old axis=[0,1,2,3]
    # hard_dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # hard_dice = tf.clip_by_value(hard_dice, 0, 1.0-epsilon)
    # new haodong
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    hard_dice = tf.reduce_mean(hard_dice, name='hard_dice')
    return hard_dice


def norm(x, norm_type, is_train, G=32, esp=1e-5, scope='group'):
    with tf.variable_scope('{}_norm'.format(scope)):
        if norm_type == 'none':
            output = x
        elif norm_type == 'batch':
            output = tf.contrib.layers.batch_norm(
                x, center=True, scale=True, decay=0.999,
                is_training=is_train, updates_collections=None
            )
        elif norm_type == 'group':
            # normalize
            # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
            x = tf.transpose(x, [0, 3, 1, 2])
            N, C, H, W = x.get_shape().as_list()
            G = min(G, C)
            x = tf.reshape(x, [-1, G, C // G, H, W])
            mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
            x = (x - mean) / tf.sqrt(var + esp)
            # per channel gamma and beta
            gamma = tf.Variable(tf.constant(1.0, shape=[C]), dtype=tf.float32, name='gamma')
            beta = tf.Variable(tf.constant(0.0, shape=[C]), dtype=tf.float32, name='beta')
            gamma = tf.reshape(gamma, [1, C, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1])

            output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
            # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
            output = tf.transpose(output, [0, 2, 3, 1])
        else:
            raise NotImplementedError
    return output
