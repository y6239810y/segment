# coding:UTF-8
import tensorlayer as tl
import tensorflow as tf
import collections
import cv2
import os
import random
import  numpy as np
import platform
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TRAIN_TIME = 100000
sysstr = platform.system()

if sysstr == "Windows":
    BATCH_SIZE = 20
    root  = "E:/ILSVRC2012/"
else:
    BATCH_SIZE = 40
    root = "/opt/ILSVRC2012/img_train/"

annotations = tf.placeholder(tf.int64, name='annotations')
input_images = tf.placeholder(tf.float32, name='input_images')
global_step = tf.Variable(0, trainable=False)
class ReduceLayer(tl.layers.Layer):
    def __init__(
        self,
        layer = None,
        axis = None,
        name ='reduce_layer',
    ):
        # 校验名字是否已被使用（不变）
        tl.layers.Layer.__init__(self, name=name)

        # 本层输入是上层的输出（不变）
        self.inputs = layer.outputs

        # 输出信息（自定义部分）
        print("  I am DoubleLayer")

        # 本层的功能实现（自定义部分）
        self.outputs = tf.reduce_mean(self.inputs, axis=axis, name='avg_pool', keep_dims=True)

        # 获取之前层的参数（不变）
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        # 更新层的参数（自定义部分）
        self.all_layers.extend( [self.outputs] )

class SoftmaxLayer(tl.layers.Layer):
    def __init__(
        self,
        layer = None,
        name ='softmax_layer',
    ):
        # 校验名字是否已被使用（不变）
        tl.layers.Layer.__init__(self, name=name)

        # 本层输入是上层的输出（不变）
        self.inputs = layer.outputs

        # 输出信息（自定义部分）
        print("  I am DoubleLayer")

        # 本层的功能实现（自定义部分）
        self.outputs = tf.nn.softmax(self.inputs,name = name )

        # 获取之前层的参数（不变）
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        # 更新层的参数（自定义部分）
        self.all_layers.extend( [self.outputs] )

def conv_2d_same(input,shape,strides,is_train=True,padding='SAME',name='unit',batch_norm=True,act = tf.nn.relu):
    if batch_norm:
        bn = tl.layers.BatchNormLayer(
            layer = input,
            is_train=is_train,
            decay=0.999,
            act=act,
            epsilon=1e-05,
            name=name + '_BN'
        )
    else:
        bn = input

    output = tl.layers.Conv2dLayer(bn,
    shape = shape,  # 32 features for each 5x5 patch
    strides = strides,
    padding = padding,
    name = name)
    return output

def bottleneck(input,channel_end,channel_start,strides,is_train=True):
    input_channel = input.outputs._shape[3]
    if input_channel == channel_end:
        shortcut = tl.layers.PoolLayer(input,
                   ksize=[1, 1, 1, 1],
                   strides=strides,
                   padding='SAME',
                   pool = tf.nn.avg_pool,
                    )
    else:
        shortcut = tl.layers.Conv2dLayer(input,shape = [1,1,input_channel,channel_end],strides = strides,
                                act=tf.identity,name='conv_0')

    residual = conv_2d_same(input=input,shape = [1,1,input_channel,channel_start],strides = strides,is_train=is_train,name='conv_1')

    residual = conv_2d_same(input=residual, shape=[3, 3, channel_start, channel_start],is_train=is_train, strides=[1,1,1,1], name='conv_2')

    residual = conv_2d_same(input=residual, shape=[1, 1, channel_start, channel_end],is_train=is_train, strides=[1, 1, 1, 1],
                            name='conv_3')
    net = tl.layers.ElementwiseLayer([residual, shortcut],
                                             combine_fn=tf.add,
                                             name='conv_merge')

    return net

def stack_blocks_dense(inuput, blocks ,is_train=True):
  for i,block in enumerate(blocks):
    with tf.variable_scope('block_%d' % (i + 1)):
      for j, unit in enumerate(block.args):
        with tf.variable_scope('unit_%d' % (j + 1)):
          channel_end, channel_start,unit_stride = unit
          out = bottleneck(inuput,
                              channel_end=channel_end,
                              channel_start=channel_start,
                              strides=[1,unit_stride,unit_stride,1],
                              is_train = is_train)
  return out

def Resnet(input,blocks,class_num,is_train=True):
    conv = conv_2d_same(input=input,shape=[7,7,3,64],strides=[1,2,2,1],is_train=is_train,batch_norm=True,name = 'conv1')
    final_channels = blocks[-1].args[0][0]
    max_pool = tl.layers.PoolLayer(conv,
                   ksize=[1, 1, 1, 1],
                   strides=[1,2,2,1],
                   padding='SAME',
                   pool = tf.nn.max_pool,
                    name = 'max_pool'
                    )
    net = stack_blocks_dense(inuput=max_pool,blocks=blocks,is_train=True)

    net = tl.layers.BatchNormLayer(
            layer = net,
            act=tf.nn.relu,
            is_train=is_train,
            name = 'BN_end'
        )
    # avg_pool = tf.reduce_mean(net, [1, 2], name='avg_pool', keep_dims=True)
    avg_pool = ReduceLayer(net,[1,2],name='avg_pool')
    avg_pool = tl.layers.FlattenLayer(avg_pool, name='flatten')
    fc = tl.layers.DenseLayer(layer=avg_pool,n_units=1000,name='fc')
    return fc

def resnet101(class_num,is_train=True):
    # x = tf.constant(value=0,dtype=tf.float32,shape=[32,224,224,3])
    x = tf.reshape(input_images,shape=[BATCH_SIZE,224,224,3])
    input = tl.layers.InputLayer(x, name='input_layer')
    Block = collections.namedtuple('Block', ['scope', 'args'])
    blocks = [
          Block(
              'block1',[(256, 64, 1)] * 2 + [(256, 64, 2)]),
          Block(
              'block2', [(512, 128, 1)] * 3 + [(512, 128, 2)]),
          Block(
              'block3',[(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
          Block(
              'block4',[(2048, 512, 1)] * 3)]
    net = Resnet(input=input, blocks=blocks,class_num=class_num,is_train=is_train)

    return net


def get_classes():
    files = os.listdir(root)
    class_files = [{} for i in range(0,1000)]
    class_file = open(root + '/synset_words.txt')
    for line in class_file:
        file = [file for file in files if file[5:] == line[:9]][0]
        images = os.listdir(root+file)
        class_num = int(file[0:4])
        class_files[class_num]={'file_name':file[5:], 'path':root+file, 'class' :class_num,'class_name':line[10:-1],'images':[image for image in images]}
        # file_names.append({'class': [file for file in files if file[5:-4] == "%.4d" % (i)][0],class })
    # for i in range(1,1001):
    #     file_names.append({'class':[file for file in files if file[0:4]=="%.4d"%(i)][0],class})
    return class_files


def get_data(classes):
    datas = [i for i in range(1000)]
    image_names = [i for i in range(1000)]
    for i,clas in enumerate(classes):
        images_number = len(clas['images'])
        rd = random.randint(0,images_number-1)
        for j,image in enumerate(clas['images']):
            if j == rd:
                image_path = clas['path'] + '/' + image
                image_array = cv2.imread(image_path)
                # print("read image:" + image_path)
                datas[i]= image_array
                image_names[i] = image
                break
    return datas,image_names


def res_train(layer, learning_rate):
        fc = layer.outputs
        cls = tf.squeeze(fc)
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls,labels=annotations))

        lr= tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   decay_steps=5000,
                                                   decay_rate=0.9,
                                                   staircase=True)
        train_op = tf.train.AdadeltaOptimizer(
            lr).minimize(cost,global_step=global_step)
        tf.summary.scalar('cos', cost)
        correct_prediction = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(cls,1), annotations),tf.float32))
        tf.summary.scalar('accuracy', correct_prediction)
        return train_op, cost, correct_prediction ,lr

def res_test(layer):
    fc = layer.outputs
    correct_prediction = tf.argmax(fc)
    return correct_prediction


net = resnet101(1000)
a, b, c, d = res_train(net, learning_rate=0.5)
sess = tf.Session()
merged = tf.summary.merge_all()
if os.path.isdir("./tensorboard"):
    pass
else:
    os.mkdir('tensorboard')
writer = tf.summary.FileWriter("./tensorboard/",sess.graph)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
classes = get_classes()
if os.path.isdir("./model"):
    pass
else:
    os.mkdir('model')
ckpt = tf.train.get_checkpoint_state("./model")
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored...")

for i in range(TRAIN_TIME):
    datas,image_names = get_data(classes)
    ls = list(range(1000))
    random.shuffle(ls)
    batchs = np.zeros([BATCH_SIZE,224,224,3])
    labels = np.zeros([BATCH_SIZE])
    acc_avg = 0
    for j,k in enumerate(ls):
        input_image = cv2.resize(datas[k], (224, 224), interpolation=cv2.INTER_CUBIC)
        batchs[j%BATCH_SIZE,:,:,:]=input_image
        labels[j%BATCH_SIZE]=k
        if (j+1)%BATCH_SIZE == 0:
            _, cos, prediction, learn_rate,result, step = (
                sess.run((a,b,c,d,merged, global_step),
                         feed_dict={input_images: batchs, annotations: labels}))
            print("step:",step,"cos",np.mean(cos),"accuracy",prediction,"learn_rate",learn_rate)
            writer.add_summary(result, step)
            acc_avg += prediction
            batchs[:,:,:,:] = 0
            labels[:] = 0
    acc_avg = acc_avg/(1000/BATCH_SIZE)
    print()
    print("acc_avg",acc_avg)
    if (i%10==0):
        saver.save(sess, "./model/model.ckpt")
    # for j in ls:
    #     input_images = datas[j]
    #     input_images = cv2.resize(input_images, (224, 224), interpolation=cv2.INTER_CUBIC)

        # height = input_images.shape[0]
        # width = input_images.shape[1]
        # label = np.zeros([1000])
        # label[j] = 1

        # print(step,cos,prediction,learn_rate)
