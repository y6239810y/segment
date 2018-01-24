import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from tools import *
import socket
import random

CLASS_NUM = 1000
LEARN_RATE = 0.1
TRAIN_TIME = 100000
BATCH_SIZE = 1
TRANS_LAYER_SIZE = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
hostname = socket.gethostname()
total_artery_accurary = 0
total_airway_accurary = 0

if  hostname == "DESKTOP-0M79B8O":
    W_SIZE = 64
    H_SIZE = 64
    root  = "E:/multi_task_data_train/"
    root_test = "E:/multi_task_data_test/"
elif hostname == "DLserver":
    W_SIZE = 64
    H_SIZE = 64
    root = "/opt/train_data/multi_task_data_train/"
    root_test = "/opt/train_data/multi_task_data_test/"
else:
    W_SIZE = 64
    H_SIZE = 64
    root = "/media/yankai/file/multi_task_data_train/"
    root_test = "/media/yankai/file/multi_task_data_test/"


class DenseNet:
    def __init__(self,k):
        self.x = tf.placeholder(tf.float32,name="input_images")
        self.y = tf.placeholder(tf.int32,name="labels")
        self.input = tf.reshape(self.x,[BATCH_SIZE,-1,W_SIZE,H_SIZE,1])
        self.global_step = tf.Variable(0, trainable=False)
        self.is_train = True
        self.decay = 0.5
        self.train_times = 0
        self.test_times = 0
        self.learning_rate = LEARN_RATE
        self.sess = tf.Session()
        self.k = k
        self.net = {}
        self.layers = [
            'CONV_1', 'POOL_1', 'CONV_2', 'POOL_2', 'DENSE_1', 'TRANS_1', 'DENSE_2',
            'TRANS_2', 'DENSE_3', 'UPSAMPLE_1', 'CONBINE_1', 'DENSE_4', 'ADD_1', 'UPSAMPLE_2', 'CONBINE_2',
            'DENSE_5', 'ADD_2', 'UPSAMPLE_3', 'CONBINE_3', 'CONV_3'
        ]
        self.layers_kernels = [
            {"kernel": [7, 7, 7], "stride": 2, "filter": 64},  # CONV_1
            {"pool_way": Max_Pooling_3d, "kernel": [3, 3, 3], "stride": 2},  # POOL_1
            {"kernel": [5, 5, 5], "stride": 1, "filter": 256},  # CONV_2
            {"pool_way": Average_pooling_3d, "kernel": [3, 3, 3], "stride": 2},  # POOL_2
            {"num": 8},  # DENSE_1
            {},  # TRANS_1
            {"num": 8},  # DENSE_2
            {},  # TRANS_2
            {"num": 8},  # DENSE_3
            {"kernel": [3, 3, 3], "stride": [2, 2, 2], "filter": 64},  # UPSAMPLE_1
            {"add_layer": 'CONV_2', "kernel": [3, 3, 3]},  # CONBINE_1
            {"num": 5},  # DENSE_4

            {"add_layer": 'CONV_2', "kernel": [1, 1, 1]},  # ADD_1
            {"kernel": [3, 3, 3], "stride": [2, 2, 2], "filter": 64},  # UPSAMPLE_2
            {"add_layer": 'CONV_1', "kernel": [3, 3, 3]},  # CONBINE_2
            {"num": 5},  # DENSE_5

            {"add_layer": 'CONV_1', "kernel": [1, 1, 1]},  # ADD_2

            {"kernel": [3, 3, 3], "stride": [2, 2, 2], "filter": 16},  # UPSAMPLE_3

            {"add_layer": 'INPUT', "kernel": [3, 3, 3]},  # CONBINE_3
            {"kernel": [1, 1, 1], "stride": 1, "filter": 4}  # CONV_3
        ]
        current = self.input
        self.net['INPUT'] = current
        for name, value in zip(self.layers, self.layers_kernels):
            if name[:-2] == 'CONV':
                with tf.variable_scope(name):
                    conv_kernel = value['kernel']
                    conv_stride = value['stride']
                    conv_filter = value['filter']
                    current = Conv3d(input=current, filter=conv_filter, kernel=conv_kernel, strides=conv_stride)
                    self.net[name] = current

            elif name[:-2] == 'POOL':
                with tf.variable_scope(name):
                    pool_way = value['pool_way']
                    pool_kernel = value['kernel']
                    pool_stride = value['stride']
                    current = pool_way(x=current, pool_size=pool_kernel, strides=pool_stride)
                self.net[name] = current

            elif name[:-2] == 'DENSE':
                current = self.DenseBlock(current, value['num'], self.k, name)
                self.net[name] = current

            elif name[:-2] == 'TRANS':
                conv_filter = current.shape[-1]
                with tf.variable_scope(name):
                    current = tf.contrib.slim.batch_norm(current, is_training=self.is_train, scope='BN')
                    current = Relu(current, name='RELU')
                    current = Conv3d(current, filter=conv_filter // 2, kernel=[1, 1, 1])
                self.net[name] = current

            elif name[:-2] == 'UPSAMPLE':
                up_kernel = value['kernel']
                up_stride = value['stride']
                up_filter = value['filter']
                with tf.variable_scope(name):
                    current = Upsample_3d(input=current, kernel=up_kernel, filter=up_filter, strides=up_stride)
                self.net[name] = current

            elif name[:-2] == 'CONBINE':
                with tf.variable_scope(name):
                    layer_name = value['add_layer']
                    kernel = value['kernel']
                    layer = self.net[layer_name]
                    conbine_tensor = tf.contrib.slim.batch_norm(layer, is_training=self.is_train, scope='BN')
                    conbine_tensor = Relu(conbine_tensor, name='RELU')
                    conbine_tensor = Conv3d(conbine_tensor, filter=current.shape[-1] // 2, kernel=kernel)
                    current = tf.concat([current, conbine_tensor], 4)
                self.net[name] = current
            elif name[:-2] == 'ADD':
                with tf.variable_scope(name):
                    layer_name = value['add_layer']
                    kernel = value['kernel']
                    layer = self.net[layer_name]
                    add_tensor = tf.contrib.slim.batch_norm(layer, is_training=self.is_train, scope='BN')
                    add_tensor = Relu(add_tensor, name='RELU')
                    add_tensor = Conv3d(add_tensor, filter=current.shape[-1], kernel=kernel)
                    current = current + 0.5 * add_tensor
                self.net[name] = current
        with tf.variable_scope('train'):
            self.class_weights = tf.placeholder(tf.float32, name='class_weights')

            self.weight_map = tf.reduce_sum(tf.multiply(tf.cast(self.y,tf.float32), self.class_weights), 3)
            current = tf.squeeze(current)

            self.cost = tf.nn.softmax_cross_entropy_with_logits(logits=current, labels=self.y)

            self.loss = tf.reduce_mean(self.cost * self.weight_map)

            self.lr = tf.train.exponential_decay(self.learning_rate,
                                            self.global_step,
                                            decay_steps=5000,
                                            decay_rate=0.99,
                                            staircase=True)

            self.train_op = tf.train.AdadeltaOptimizer(
                self.lr).minimize(self.loss, global_step=self.global_step)
            self.loss_summary = tf.summary.scalar('loss', self.loss)

        self.airway_result, self.artery_result,self.lung_result,self.airway_correct_prediction,\
        self.artery_correct_prediction,self.lung_correct_prediction=self.GetResult()

        with tf.variable_scope("accurary"):
            self.artery_accurary = tf.placeholder(tf.float32, name="artery_accurary")
            self.airway_accurary = tf.placeholder(tf.float32, name="airway_accurary")
            self.lung_accurary = tf.placeholder(tf.float32, name="lung_accurary")

            self.artery_accurary = tf.reduce_mean(self.artery_accurary)
            self.airway_accurary = tf.reduce_mean(self.airway_accurary)
            self.lung_accurary = tf.reduce_mean(self.lung_accurary)

            self.artery_acc = tf.summary.scalar('artery_accurary', self.artery_accurary)
            self.airway_acc = tf.summary.scalar('airway_accurary', self.airway_accurary)
            self.lung_acc = tf.summary.scalar('lung_accurary', self.lung_accurary)


        self.merged = tf.summary.merge([self.loss_summary])
        self.train_writer = tf.summary.FileWriter("./tensorboard/train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter("./tensorboard/test", self.sess.graph)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.Reload()

    def GetResult(self):
        with tf.variable_scope("GetResult"):
            x = tf.squeeze(self.net['CONV_3'])
            out_result = tf.cast(tf.argmax(x, 3), tf.float32)
            constant = tf.cast(tf.equal(out_result, out_result), tf.float32)
            airway_result = tf.cast(tf.equal(out_result, constant), tf.float32)

            artery_result = tf.cast(tf.less(out_result, constant), tf.float32)

            lung_result = tf.cast(tf.equal(out_result, 2*constant), tf.float32)

            label_artery,label_airway,label_lung ,label_bg = tf.split(self.y, [1, 1, 1,1], axis=3)

            label_airway = tf.cast(tf.squeeze(label_airway),tf.float32)

            label_artery = tf.cast(tf.squeeze(label_artery),tf.float32)

            label_lung = tf.cast(tf.squeeze(label_lung), tf.float32)

            airway_intersection = label_airway * airway_result

            airway_union = tf.cast(tf.greater_equal((label_airway + airway_result), constant), tf.float32)

            airway_correct_prediction = tf.reduce_mean(airway_intersection) / tf.reduce_mean(airway_union)


            artery_intersection = label_artery * artery_result

            artery_union = tf.cast(tf.greater_equal((label_artery + artery_result), constant), tf.float32)

            artery_correct_prediction = tf.reduce_mean(artery_intersection) / tf.reduce_mean(artery_union)


            lung_intersection = label_lung * lung_result

            lung_union = tf.cast(tf.greater_equal((label_lung + lung_result), constant), tf.float32)

            lung_correct_prediction = tf.reduce_mean(lung_intersection) / tf.reduce_mean(lung_union)

        return airway_result,artery_result,lung_result,airway_correct_prediction,artery_correct_prediction,lung_correct_prediction

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
                        inputs = tf.concat([inputs, self.net[name][j]], 4)
                    BN_1 = tf.contrib.slim.batch_norm(inputs, is_training=self.is_train, scope='BN_1')
                    x_1 = Relu(BN_1)
                    conv_1 = Conv3d(input=x_1, filter=4*self.k, kernel=[1,1,1],layer_name='CONV_1')

                    BN_2 = tf.contrib.slim.batch_norm(conv_1, is_training=self.is_train, scope='BN_2')
                    x_2 = Relu(BN_2)
                    conv = Conv3d(input=x_2, filter=self.k, kernel=[3,3,3], layer_name='CONV_2')

                    self.net[name].append(conv)
                    output = tf.concat([output, conv], 4)
        self.net[name].append(output)
        return output

    # def Run(self,inputs,labels,weights):
    #     airway_result,artery_result,airway_union,artery_union,constant,test = self.sess.run(
    #         (self.airway_result,self.artery_result,self.airway_union,self.artery_union,self.constant,self.test
    #         ),
    #         feed_dict={self.x: inputs, self.y: labels, self.class_weights: weights})
    #     return airway_result,artery_result,airway_union,artery_union,constant,test
    def RunAccurary(self,artery_accurary,airway_accurary,lung_accurary):

        self.merged_accurary = tf.summary.merge([self.artery_acc,self.airway_acc,self.lung_acc])
        result, step = self.sess.run((self.merged_accurary, self.global_step), feed_dict={self.artery_accurary: artery_accurary,
                                                                                 self.airway_accurary: airway_accurary,
                                                                                 self.lung_accurary: lung_accurary
                                                                                })
        if(self.is_train):
            self.train_writer.add_summary(result,self.train_times)
            self.train_times = self.train_times + 1
        else:
            self.test_writer.add_summary(result, self.test_times)
            self.test_times = self.test_times + 1


    def Train(self,inputs,labels,weights):
        self.is_train = True
        if os.path.isdir("./tensorboard"):
            pass
        else:
            os.mkdir('tensorboard')


        _, loss, airway,artery,lung,airway_accurary,artery_accurary,lung_accurary,learn_rate, result, step = self.sess.run((
            self.train_op, self.loss,self.airway_result,self.artery_result,self.lung_result,
            self.airway_correct_prediction,self.artery_correct_prediction,self.lung_correct_prediction,
            self.lr, self.merged, self.global_step),feed_dict={self.x:inputs,self.y:labels,self.class_weights:weights})

        self.train_writer.add_summary(result, step)
        return loss,airway,artery,lung,airway_accurary,artery_accurary,lung_accurary,learn_rate, step

    def Test(self,inputs,labels,weights):
        self.is_train = False
        if os.path.isdir("./tensorboard"):
            pass
        else:
            os.mkdir('tensorboard')

        loss, airway, artery, lung, airway_accurary, artery_accurary, lung_accurary, learn_rate, step = self.sess.run(
            (
                self.loss, self.airway_result, self.artery_result, self.lung_result,
                self.airway_correct_prediction, self.artery_correct_prediction, self.lung_correct_prediction,
                self.lr, self.merged, self.global_step),
            feed_dict={self.x: inputs, self.y: labels, self.class_weights: weights})

        return loss, airway, artery, lung, airway_accurary, artery_accurary, lung_accurary, learn_rate, step

# model=DenseNet(32)
def set_Window(image, max, min):
    array = sitk.GetArrayFromImage(image)
    array_max = np.max(array)
    array_min = np.min(array)
    image_out = sitk.IntensityWindowing(image, array_min * 1.0, array_max * 1.0, min, max)
    return image_out


def get_data(path,file,size):
    print(file + "  reading")
    reader = sitk.ImageSeriesReader()
    dicoms = reader.GetGDCMSeriesFileNames(path+file+'/original1')
    reader.SetFileNames(dicoms)
    image = reader.Execute()
    dicom_array = sitk.GetArrayFromImage(set_Window(image, 1024, 0))

    label_array = np.zeros([dicom_array.shape[0],dicom_array.shape[1],dicom_array.shape[2],4])

    artery = reader.GetGDCMSeriesFileNames(path+file+'/artery')
    reader.SetFileNames(artery)
    image = reader.Execute()
    artery_array = sitk.GetArrayFromImage(image)


    airway = reader.GetGDCMSeriesFileNames(path+file+'/airway')
    reader.SetFileNames(airway)
    image = reader.Execute()
    airway_array = sitk.GetArrayFromImage(image)

    lung = reader.GetGDCMSeriesFileNames(path + file + '/lung')
    reader.SetFileNames(lung)
    image = reader.Execute()
    lung_array = sitk.GetArrayFromImage(image)


    label_array[:, :, :, 0] = artery_array
    label_array[:, :, :, 1] = ((airway_array - artery_array) == 1.0)
    label_array[:, :, :, 2] = ((lung_array - airway_array - artery_array) == 1.0)

    temp = airway_array + artery_array + lung_array
    temp = temp>0
    label_array[:, :, :, 3] = ~temp

    return dicom_array,label_array

def process_datas(dicom_array,size,train):
    indexs = []
    shape = dicom_array.shape
    plies = shape[0]
    w = shape[1]
    h = shape[2]
    if (train):
        step_size = size // 2
        c_nums = (plies + step_size - 1) // step_size - 1
        w_nums = (w + step_size - 1) // step_size - 1
        h_nums = (h + step_size - 1) // step_size - 1
    else:
        step_size = size
        c_nums = (plies + step_size - 1) // step_size
        w_nums = (w + step_size - 1) // step_size
        h_nums = (h + step_size - 1) // step_size
    count=0
    for i in range(c_nums):
        c_start = i * step_size
        c_end = c_start + size

        if c_end > plies:
            c_end = plies
            c_start = c_end - size
        else:
            pass
        for j in range(w_nums):
            w_start = j * step_size
            w_end = w_start + size

            if w_end > w:
                w_end = w
                w_start = w_end - size
            else:
                pass
            for k in range(h_nums):
                h_start = k * step_size
                h_end = h_start + size

                if h_end > h:
                    h_end = h
                    h_start = h_end - size
                else:
                    pass
                count+=1
                indexs.append([c_start,c_end,w_start,w_end,h_start,h_end])
    return indexs

def net_train(file_list,weights):
    for i, file in enumerate(file_list):
        data_array,label_array = get_data(root,file,1)
        out_vtk_airway = np.zeros(data_array.shape)
        out_vtk_artery = np.zeros(data_array.shape)
        out_vtk_lung = np.zeros(data_array.shape)

        indexs = process_datas(data_array,W_SIZE,train=True)
        data_list = list(indexs)
        random.shuffle(data_list)
        for j,index in enumerate(data_list):
            c_start, c_end, w_start, w_end, h_start, h_end = (index[k] for k in range(6))
            data = data_array[c_start:c_end, w_start:w_end, h_start:h_end]
            label = label_array[c_start:c_end, w_start:w_end, h_start:h_end]

            if (np.sum(label[:, :, :, 0] + label[:, :, :, 1])> 0):
                artery_percentage = np.sum(label[:, :, :, 0]) / np.size(label[:, :, :, 0])
                airway_percentage = np.sum(label[:, :, :, 1]) / np.size(label[:, :, :, 1])
                lung_percentage = np.sum(label[:, :, :, 2]) / np.size(label[:, :, :, 2])

                loss, airway, artery,lung, airway_accurary, artery_accurary,lung_accurary, learn_rate, step = net.Train(data, label,
                                                                                                             weights)
                out_vtk_artery[c_start:c_end, w_start:w_end, h_start:h_end] = artery
                out_vtk_airway[c_start:c_end, w_start:w_end, h_start:h_end] = airway
                out_vtk_lung[c_start:c_end, w_start:w_end, h_start:h_end] = lung
                if(j%10==0):
                    print('step ', step, ' artery_percentage ', artery_percentage, ' airway_percentage ',
                          airway_percentage,' lung_percentage ',lung_percentage,'learning_rate',learn_rate)
                    print(' loss ', loss, ' artery_accurary ', artery_accurary, ' airway_accurary ',
                          airway_accurary,'lung_accurary',lung_accurary)
                    print("=================================================================================")

        total_artery_accurary = np.sum(out_vtk_artery * label_array[:, :, :, 0] > 0) * 1.0 / np.sum(
            out_vtk_artery + label_array[:, :, :, 0] > 0) * 1.0
        total_airway_accurary = np.sum(out_vtk_airway * label_array[:, :, :, 1] > 0) * 1.0 / np.sum(
            out_vtk_airway + label_array[:, :, :, 1] > 0) * 1.0
        total_lung_accurary = np.sum(out_vtk_lung * label_array[:, :, :, 2] > 0) * 1.0 / np.sum(
            out_vtk_airway + label_array[:, :, :, 2] > 0) * 1.0

        a = (1-total_artery_accurary)/(3-total_artery_accurary-total_airway_accurary-total_lung_accurary)
        b = (1-total_airway_accurary)/(3-total_artery_accurary-total_airway_accurary-total_lung_accurary)
        c = (1-total_lung_accurary)/(3-total_artery_accurary-total_airway_accurary-total_lung_accurary)
        d = 0.01
        weights = [a*100,b*100,c*100,d*100]

        net.RunAccurary(total_artery_accurary,total_airway_accurary,total_lung_accurary)
        print("=================================================================================")
        print("=================================================================================")
        print("=================================================================================")
        print("time: " + str(times+1) + "  Dicom: " + str(i+1) + " total_artery_accurary: " + str(
            total_artery_accurary) + " total_airway_accurary: " + str(total_airway_accurary)
              + " total_lung_accurary: " + str(total_lung_accurary))
        print("\n\n\n")

        if os.path.isdir("./vtk"):
            pass
        else:
            os.mkdir('vtk')
        artery_output_vtk = sitk.GetImageFromArray(out_vtk_artery)
        sitk.WriteImage(artery_output_vtk, './vtk/out_artery_' + str(i+1) + '.vtk')
        airway_output_vtk = sitk.GetImageFromArray(out_vtk_airway)
        sitk.WriteImage(airway_output_vtk, './vtk/out_airway_' + str(i + 1) + '.vtk')
        lung_output_vtk = sitk.GetImageFromArray(out_vtk_lung)
        sitk.WriteImage(lung_output_vtk, './vtk/out_lung_' + str(i + 1) + '.vtk')

        del out_vtk_artery
        del out_vtk_airway
        del out_vtk_lung
        net.Store()

def net_test(file_test_list):
    for i, file in enumerate(file_test_list):
        data_array,label_array = get_data(root_test,file,1)
        out_vtk_airway = np.zeros(data_array.shape)
        out_vtk_artery = np.zeros(data_array.shape)
        out_vtk_lung = np.zeros(data_array.shape)
        indexs = process_datas(data_array,W_SIZE,train=True)
        data_list = list(indexs)
        random.shuffle(data_list)
        for index in data_list:
            c_start, c_end, w_start, w_end, h_start, h_end = (index[k] for k in range(6))
            data = data_array[c_start:c_end, w_start:w_end, h_start:h_end]
            label = label_array[c_start:c_end, w_start:w_end, h_start:h_end]

            # artery_percentage = np.sum(label[:, :, :, 0]) / np.size(label[:, :, :, 0])
            # airway_percentage = np.sum(label[:, :, :, 1]) / np.size(label[:, :, :, 1])
            # lung_percentage = np.sum(label[:, :, :, 2]) / np.size(label[:, :, :, 2])
            loss, airway, artery, lung, airway_accurary, artery_accurary, lung_accurary, learn_rate, step = net.Test(
                data, label,
                [70, 22,
                 5, 1])

            out_vtk_artery[c_start:c_end,w_start:w_end,h_start:h_end] = artery
            out_vtk_airway[c_start:c_end, w_start:w_end, h_start:h_end] = airway
            out_vtk_lung[c_start:c_end, w_start:w_end, h_start:h_end] = lung

        total_artery_accurary = np.sum(out_vtk_artery * label_array[:, :, :, 0] > 0) * 1.0 / np.sum(
            out_vtk_artery + label_array[:, :, :, 0] > 0) * 1.0
        total_airway_accurary = np.sum(out_vtk_airway * label_array[:, :, :, 1] > 0) * 1.0 / np.sum(
            out_vtk_airway + label_array[:, :, :, 1] > 0) * 1.0
        total_lung_accurary = np.sum(out_vtk_lung * label_array[:, :, :, 2] > 0) * 1.0 / np.sum(
            out_vtk_lung + label_array[:, :, :, 2] > 0) * 1.0

        net.RunAccurary(total_artery_accurary,total_airway_accurary,total_lung_accurary)

        print("=================================================================================")
        print("=================================================================================")
        print("=================================================================================")
        print("time: " + str(times + 1) + "  test: " + str(i + 1) + " total_artery_accurary: " + str(
            total_artery_accurary) + " total_airway_accurary: " + str(total_airway_accurary)
              + " total_lung_accurary: " + str(total_lung_accurary))
        print("\n\n\n")
        if os.path.isdir("./vtk"):
            pass
        else:
            os.mkdir('vtk')
        artery_output_vtk = sitk.GetImageFromArray(out_vtk_artery)
        sitk.WriteImage(artery_output_vtk, './vtk/artery_test_' + str(i + 1) + '.vtk')
        airway_output_vtk = sitk.GetImageFromArray(out_vtk_airway)
        sitk.WriteImage(airway_output_vtk, './vtk/airway_test_' + str(i + 1) + '.vtk')
        lung_output_vtk = sitk.GetImageFromArray(out_vtk_lung)
        sitk.WriteImage(lung_output_vtk, './vtk/lung_test_' + str(i + 1) + '.vtk')
        del out_vtk_artery
        del out_vtk_airway
        del out_vtk_lung

net = DenseNet(32)
file_list = os.listdir(root)
file_test_list = os.listdir(root_test)
weights = [0.8 * 100, 0.12 * 100,0.05 * 100,0.01 * 100]
for times in range(TRAIN_TIME):
    net_train(file_list,weights)
    net_test(file_test_list)
