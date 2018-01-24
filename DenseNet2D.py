import tensorlayer as tl
import tensorflow as tf
import numpy as np
import cv2
import os
from tools import *
import platform
import random
W_SIZE = 224
H_SIZE = 224
CLASS_NUM = 1000
LEARN_RATE = 0.1
TRAIN_TIME = 100000

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sysstr = platform.system()

if sysstr == "Windows":
    BATCH_SIZE = 20
    root  = "E:/ILSVRC2012/"
else:
    BATCH_SIZE = 40
    root = "/opt/ILSVRC2012/img_train/"


class DenseNet:
    def __init__(self,k):
        self.x = tf.placeholder(tf.float32,name="input_images")
        self.y = tf.placeholder(tf.int64,name="labels")
        self.input = tf.reshape(self.x,[BATCH_SIZE,W_SIZE,H_SIZE,3])
        self.global_step = tf.Variable(0, trainable=False)
        self.is_train = True
        self.decay = 0.5
        self.learning_rate = LEARN_RATE
        self.sess = tf.Session()
        self.k = k
        self.net = {}
        self.layers = [
            'transition_layers_1','denseblock_1','transition_layers_2','denseblock_2',
            'transition_layers_3', 'denseblock_3', 'transition_layers_4', 'denseblock_4',
            'finally_layers'
        ]
        self.layers_kernels = [
            {'conv_kernels':7,'conv_stride':2,'pool_way':Max_Pooling_2d,'pool_kernels':3,'pool_stride':2},
            {'nums':6},
            {'conv_kernels': 1, 'conv_stride': 1, 'pool_way': Average_pooling_2d, 'pool_kernels': 2, 'pool_stride': 2},
            {'nums': 12},
            {'conv_kernels': 1, 'conv_stride': 1, 'pool_way': Average_pooling_2d, 'pool_kernels': 2, 'pool_stride': 2},
            {'nums': 24},
            {'conv_kernels': 1, 'conv_stride': 1, 'pool_way': Average_pooling_2d, 'pool_kernels': 2, 'pool_stride': 2},
            {'nums': 16},
            {'pool_kernels',7}
                ]
        current = self.input
        for name, value in zip(self.layers, self.layers_kernels):
            if name[:-2] == 'transition_layers':
                with tf.variable_scope(name):
                    conv_kernels = [value['conv_stride'],value['conv_stride']]
                    conv_stride = value['conv_stride']
                    pool_way = value['pool_way']
                    pool_kernels = value['pool_kernels']
                    pool_stride = value['pool_stride']
                    if name == 'transition_layers_1':
                        current = Conv2d(input=current, filter=2*self.k, kernel=conv_kernels, strides=conv_stride)
                        current = pool_way(x=current,pool_size=pool_kernels,strides=pool_stride,name='Pool')
                    else:
                        shape = current.shape
                        current = tf.contrib.slim.batch_norm(current, is_training=self.is_train, scope='BN')
                        current = Relu(current)
                        current = Conv2d(input=current, filter=self.decay * (int)(shape[-1]), kernel=conv_kernels, strides=conv_stride)
                        current = pool_way(x=current, pool_size=pool_kernels, strides=pool_stride, name='Pool')
                    self.net[name] = current
            elif name[:-2] == 'denseblock':
                current = self.DenseBlock(current,value['nums'],self.k,name)
            else:
                current = tf.contrib.slim.batch_norm(current, is_training=self.is_train, scope='BN')
                current = Relu(current)
                current = global_avg_pool(current)
                self.net['global_avg_pool'] = current
                current = tf.layers.flatten(current)
                current = tf.layers.dense(current,units=CLASS_NUM,name='Fullyconnect')
                self.net[name] = current


        current = tf.squeeze(current)
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=current, labels=self.y))

        self.lr = tf.train.exponential_decay(self.learning_rate,
                                        self.global_step,
                                        decay_steps=5000,
                                        decay_rate=0.9,
                                        staircase=True)

        self.train_op = tf.train.AdadeltaOptimizer(
            self.lr).minimize(self.cost, global_step=self.global_step)
        tf.summary.scalar('cos', self.cost)
        self.correct_prediction = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(current, 1), self.y), tf.float32))
        tf.summary.scalar('accuracy', self.correct_prediction)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./tensorboard/", self.sess.graph)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.Reload()

    def Reload(self):
        if os.path.isdir("./model"):
            pass
        else:
            os.mkdir('model')
        ckpt = tf.train.get_checkpoint_state("./model")
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        else:
            print("Model creating...")

    def Store(self):
        self.saver.save(self.sess, "./model/model.ckpt")

    def DenseBlock(self,input, nums, k, name):
        self.net[name] = []
        self.net[name].append(input)
        with tf.variable_scope(name):
            output = input
            for i in range(1, nums + 1):
                inputs = input
                with tf.variable_scope('Bottleneck' + str(i)):
                    for j in range(1, i):
                        inputs = tf.concat([inputs, self.net[name][j]], 3)
                    BN_1 = tf.contrib.slim.batch_norm(inputs, is_training=self.is_train, scope='BN_1')
                    x_1 = Relu(BN_1)
                    conv_1 = Conv2d(input=x_1, filter=4*self.k, kernel=[1,1],layer_name='CONV_1')

                    BN_2 = tf.contrib.slim.batch_norm(conv_1, is_training=self.is_train, scope='BN_2')
                    x_2 = Relu(BN_2)
                    conv = Conv2d(input=x_2, filter=self.k, kernel=[3, 3], layer_name='CONV_2')

                    self.net[name].append(conv)
                    output = tf.concat([output, conv], 3)
        self.net[name].append(output)
        return output

    def Train(self,inputs,labels):
        if os.path.isdir("./tensorboard"):
            pass
        else:
            os.mkdir('tensorboard')

        _, cos, prediction, learn_rate, result, step = self.sess.run((self.train_op, self.cost, self.correct_prediction,
                                                                      self.lr, self.merged, self.global_step),feed_dict={self.x:inputs,self.y:labels})
        self.writer.add_summary(result, step)
        return cos, prediction, learn_rate, result, step

    def Test(self,inputs,labels):
        if os.path.isdir("./tensorboard"):
            pass
        else:
            os.mkdir('tensorboard')

        cos, prediction, learn_rate, result, step = self.sess.run((self.cost, self.correct_prediction,
                                                                      self.lr, self.merged, self.global_step),feed_dict={self.x:inputs,self.y:labels})
        self.writer.add_summary(result, step)
        return cos, prediction, learn_rate, result, step

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

model=DenseNet(32)

classes = get_classes()

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
            cos, prediction, learn_rate,result, step = model.Train(
                batchs,labels
            )
            print("step:",step,"cos",np.mean(cos),"accuracy",prediction,"learn_rate",learn_rate)

            acc_avg += prediction
            batchs[:,:,:,:] = 0
            labels[:] = 0
    acc_avg = acc_avg/(1000/BATCH_SIZE)
    print()
    print("acc_avg",acc_avg)
    if (i%10==0):
        model.Store()