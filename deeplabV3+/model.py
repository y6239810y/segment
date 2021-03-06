from tools import *
from conv_lstm import SegLSTMCell
import tensorflow.contrib.slim as slim
import os, shutil
from deeplab_model import deeplab_v3_plus_generator
_BATCH_NORM_DECAY = 0.9997

class LstmSegNet:
    def __init__(self, layers,
                 layers_kernels,
                 threshold=0.5,
                 save_path='.',
                 learning_rate=0.1,
                 decay_steps=300,
                 decay_rate=0.99,
                 batch_size=12,
                 width=512,
                 height=512,
                 resume=True,
                 loss_func="softmax"):

        self.x = tf.placeholder(tf.float32, name="input_data")  # 输入数据
        self.y = tf.placeholder(tf.int32, name="input_label")  # 实际标签
        self.batch_size = batch_size
        self.width = width
        self.height = height

        self.input = tf.reshape(self.x, [batch_size, width, height, 1])
        self.is_train = True  # 训练状态
        self.train_times = 0  # 已训练次数, 会把这个记录到tensorboard
        self.test_times = 0  # 已经测试次数, 会把这个记录到tensorboard
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = learning_rate
        self.lr = learning_rate
        self.threshold = threshold
        self.resume = resume
        self.sess = tf.Session()
        self.net = {}
        self.save_path = save_path
        self.layers = layers

        self.layers_kernels = layers_kernels

        current = self.input
        model = deeplab_v3_plus_generator(num_classes=2, output_stride=16, base_architecture='resnet_v2_101',
                                          pre_trained_model='', batch_norm_decay=_BATCH_NORM_DECAY)
        self.out = model(self.input)

        with tf.variable_scope('train'):  # 训练部分

            if loss_func == 'cross_entropy':
                self.class_weights = tf.placeholder(tf.float32, name='class_weights')

                self.weight_map = tf.reduce_sum(tf.multiply(tf.cast(self.y, tf.float32), self.class_weights), 3)  # 权值

                current = tf.squeeze(self.out)

                self.softmax_cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=current, labels=self.y)

                self.loss = tf.reduce_mean(self.softmax_cost * self.weight_map)  # 损失函数权值调整

                for num,key in enumerate(self.net.keys()):
                    if "SUPERVISE" in key:
                        supervise_current = self.net[key]
                        supervise_current = tf.squeeze(supervise_current)
                        supervise_softmax_cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=supervise_current, labels=self.y)
                        self.loss += 5 * self.lr * tf.reduce_mean(supervise_softmax_cost * self.weight_map)


            elif loss_func == 'dice':
                self.class_weights = tf.placeholder(tf.float32, name='class_weights')

                current = tf.squeeze(self.out)
                current = tf.nn.softmax(current, axis=3)

                self.obj_map, self.bg_map = tf.split(current, 2, 3)
                self.label_obj_map, self.label_bg_map = tf.split(self.y, 2, 3)

                self.obj_map = tf.squeeze(self.obj_map)

                self.loss = 1 - dice_coe(output=self.obj_map,
                                         target=tf.squeeze(tf.cast(self.label_obj_map, tf.float32)))

                for num,key in enumerate(self.net.keys()):
                    if "SUPERVISE" in key:
                        supervise_current = self.net[key]
                        supervise_current = tf.squeeze(supervise_current)
                        supervise_current = tf.nn.softmax(supervise_current, axis=3)
                        supervise_obj_map, supervise_bg_map = tf.split(supervise_current, 2, 3)
                        supervise_cost = 1 - dice_coe(output=supervise_obj_map,
                                         target=tf.squeeze(tf.cast(self.label_obj_map, tf.float32)))

                        self.loss += 5 * self.lr * supervise_cost

            elif loss_func == 'focal':
                self.class_weights = tf.placeholder(tf.float32, name='class_weights')

                current = tf.squeeze(self.out)

                self.loss = focal_loss(logits=current, onehot_labels=tf.squeeze(tf.cast(self.y, tf.float32)))

                for num,key in enumerate(self.net.keys()):
                    if "SUPERVISE" in key:
                        supervise_current = self.net[key]
                        supervise_current = tf.squeeze(supervise_current)

                        self.loss += 5 * self.lr * focal_loss(logits=supervise_current, onehot_labels=tf.squeeze(tf.cast(self.y, tf.float32)))


            else:
                self.obj_map, self.bg_map = tf.split(self.out, 2, 3)
                self.label_obj_map, self.label_bg_map = tf.split(self.y, 2, 3)

                self.obj_map = tf.squeeze(self.obj_map)
                self.obj_map = tf.nn.sigmoid(self.obj_map)

                self.dice_cost = 1 - dice_coe(output=self.obj_map,
                                              target=tf.squeeze(tf.cast(self.label_obj_map, tf.float32)))

                self.class_weights = tf.placeholder(tf.float32, name='class_weights')

                self.weight_map = tf.reduce_sum(tf.multiply(tf.cast(self.y, tf.float32), self.class_weights), 3)  # 权值

                current = tf.squeeze(current)

                self.softmax_cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=current, labels=self.y)

                self.loss = 0.8 * tf.reduce_mean(self.softmax_cost * self.weight_map) + 0.2 * tf.reduce_mean(
                    self.dice_cost)  # 损失函数权

            self.lr = tf.train.exponential_decay(self.learning_rate,
                                                 self.global_step,
                                                 decay_steps=decay_steps,
                                                 decay_rate=decay_rate,
                                                 staircase=True)

            self.train_op = tf.train.AdadeltaOptimizer(
                self.lr).minimize(self.loss, global_step=self.global_step)

            self.loss_summary = tf.summary.scalar('loss', self.loss)

        self.liver_result, self.liver_correct_prediction = self._get_result()

        with tf.variable_scope("accurary"):  # 计算准确度并保存到tensorboard
            self.liver_IOU = tf.placeholder(tf.float32, name="liver_IOU")

            self.liver_IOU = tf.reduce_mean(self.liver_IOU)

            self.liver_iou = tf.summary.scalar('liver_iou', self.liver_IOU)

            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            self._reload()

        self.merged = tf.summary.merge([self.loss_summary])

        self.train_writer = tf.summary.FileWriter(os.path.join(self.save_path, "tensorboard/train"), self.sess.graph)
        self.test_writer = tf.summary.FileWriter(os.path.join(self.save_path, "tensorboard/test"), self.sess.graph)

    def _get_result(self):  # 将网络单次执行结果 计算出准确度
        with tf.variable_scope("GetResult"):
            x = self.out

            x = tf.nn.softmax(x, axis=3)

            liver_result, liver_bg = tf.split(x, [1, 1], axis=3)

            label_liver, label_bg = tf.split(self.y, [1, 1], axis=3)  # 分离背景和前景

            liver_result = tf.squeeze(liver_result)

            label_liver = tf.squeeze(label_liver)

            # result_dice = dice_hard_coe(liver_result,label_liver,threshold=self.threshold)

            result_iou = iou_coe(liver_result, tf.cast(label_liver, tf.float32), threshold=self.threshold)

        return liver_result, result_iou

    def _reload(self):  # 重新载入模型
        if os.path.isdir(os.path.join(self.save_path, "model")):
            pass
        else:
            os.makedirs(os.path.join(self.save_path, "model"))

        if os.path.isdir(os.path.join(self.save_path, "model_best")):
            pass
        else:
            os.makedirs(os.path.join(self.save_path, "model_best"))

        ckpt = tf.train.get_checkpoint_state(os.path.join(self.save_path, "model"))
        if ckpt and ckpt.model_checkpoint_path and self.resume:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        else:
            print("Model creating...")

    def _store(self, isBest):  # 保存模型
        if isBest:
            self.saver.save(self.sess, os.path.join(self.save_path, "model/model.ckpt"))
            self.saver.save(self.sess, os.path.join(self.save_path, "model_best/model.ckpt"))
        else:
            self.saver.save(self.sess, os.path.join(self.save_path, "model/model.ckpt"))

    def _dense_block(self, input, nums, name):  # DenseNet模块
        self.net[name] = []
        self.net[name].append(input)
        with tf.variable_scope(name):
            output = input
            for i in range(1, nums + 1):
                inputs = input
                with tf.variable_scope('Bottleneck' + str(i)):
                    for j in range(1, i):
                        inputs = tf.concat([inputs, self.net[name][j]], 4)
                    BN_1 = tf.contrib.slim.batch_norm(inputs, is_training=self.is_train, scope='BN_1')
                    x_1 = Relu(BN_1)
                    conv_1 = Conv2d(input=x_1, filter=4 * self.k, kernel=[1, 1, 1], layer_name='CONV_1')

                    BN_2 = tf.contrib.slim.batch_norm(conv_1, is_training=self.is_train, scope='BN_2')
                    x_2 = Relu(BN_2)
                    conv = Conv2d(input=x_2, filter=self.k, kernel=[3, 3, 3], layer_name='CONV_2')

                    self.net[name].append(conv)
                    output = tf.concat([output, conv], 4)
            with tf.variable_scope('SeUnit'):
                avg_pooling = global_avg_pool(output)

                fullyConnectOne = tf.layers.dense(avg_pooling, units=output.shape[-1] // 16, name='FullyConnectOne')

                fullyConnectTwo = tf.layers.dense(fullyConnectOne, units=output.shape[-1], name='FullyConnectTwo')

                output = tf.multiply(tf.sigmoid(fullyConnectTwo), output, name='Scale')

        self.net[name].append(output)
        return output

    def _res_block(self, input, nums, filter, stride, norm_type, name):  # ResNet模块实现方法
        add_layer = Conv2d(input=input, filter=filter, strides=stride, kernel=[1, 1], layer_name='ADD_CONV')
        output = input

        for i in range(1, nums + 1):
            if i == 1:
                strides = stride
            else:
                strides = 1
            with tf.variable_scope('Bottleneck' + str(i)):
                # w = tf.Variable(initial_value=[1], dtype=tf.float32)
                norm_1 = norm(output, norm_type=norm_type, is_train=self.is_train, scope='NORM_1')
                x_1 = Relu(norm_1)

                conv_1 = Conv2d(input=x_1, strides=strides, filter=filter, kernel=[3, 3], layer_name='CONV_1')

                norm_2 = norm(conv_1, norm_type=norm_type, is_train=self.is_train, scope='NORM_2')

                x_2 = Relu(norm_2)
                conv = Conv2d(input=x_2, filter=filter, kernel=[3, 3], layer_name='CONV_2')

                output = add_layer + conv
                add_layer = output
        self.net[name] = output
        return output

    def _atrous_spatial_pyramid_pooling(self, inputs, scope, depth=512, reuse=None):

        with tf.variable_scope(scope, reuse=reuse):
            feature_map_size = tf.shape(inputs)
            image_level_features = tf.reduce_mean(inputs, [1, 2], name='image_level_global_pool', keepdims=True)
            image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1",
                                               activation_fn=None)
            image_level_features = tf.image.resize_bilinear(image_level_features,
                                                            (feature_map_size[1], feature_map_size[2]))

            at_pool1x1 = slim.conv2d(inputs, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)

            at_pool3x3_1 = slim.conv2d(inputs, depth, [3, 3], scope="conv_3x3_1", rate=6, activation_fn=None)

            at_pool3x3_2 = slim.conv2d(inputs, depth, [3, 3], scope="conv_3x3_2", rate=12, activation_fn=None)

            at_pool3x3_3 = slim.conv2d(inputs, depth, [3, 3], scope="conv_3x3_3", rate=18, activation_fn=None)

            net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                            name="concat")
            net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)

            return net

    def _run_accurary(self, IOU):  # 将网络外部计算出的整套结果准确度参数，存入tensorboard
        self.merged_recall = tf.summary.merge([self.liver_iou])
        result, step = self.sess.run((self.merged_recall, self.global_step),
                                     feed_dict={self.liver_IOU: IOU
                                                })
        if (self.is_train):  # 判断是不是在训练，如果是训练 存入train tensorboard 否则存入test tensorboard
            self.train_writer.add_summary(result, self.train_times)
            self.train_times = self.train_times + 1

        else:
            self.test_writer.add_summary(result, self.test_times)
            self.test_times = self.test_times + 1

    def _train(self, inputs, labels, weights):  # 开始执行训练的操作函数

        self.is_train = True
        if os.path.isdir(os.path.join(self.save_path, "tensorboard")):
            pass
        else:
            os.makedirs(os.path.join(self.save_path, "tensorboard"))

        _, loss, liver, liver_iou, learn_rate, result, step = self.sess.run((
            self.train_op, self.loss, self.liver_result, self.liver_correct_prediction,
            self.lr, self.merged, self.global_step),
            feed_dict={self.x: inputs, self.y: labels, self.class_weights: weights})

        self.train_writer.add_summary(result, step)
        return loss, liver, liver_iou, learn_rate, step

    def _val(self, inputs, labels, weights):  # 开始执行测试的操作函数
        self.is_train = False
        if os.path.isdir(os.path.join(self.save_path, "tensorboard")):
            pass
        else:
            os.makedirs(os.path.join(self.save_path, "tensorboard"))

        loss, liver, liver_iou, learn_rate, step = self.sess.run(
            (self.loss, self.liver_result, self.liver_correct_prediction, self.lr, self.global_step),
            feed_dict={self.x: inputs, self.y: labels, self.class_weights: weights})

        return loss, liver, liver_iou, learn_rate, step
