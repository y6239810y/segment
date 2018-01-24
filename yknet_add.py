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
    W_SIZE = 128
    H_SIZE = 128
    root = "/opt/multi_task_data_train/"
    root_test = "/opt/multi_task_data_test/"
else:
    W_SIZE = 64
    H_SIZE = 64
    root = "/media/yankai/file/multi_task_data_train/"
    root_test = "/media/yankai/file/multi_task_data_test/"


class DenseNet:
    def __init__(self,k):
        self.x = tf.placeholder(tf.float32,name="input_images")
        self.y = tf.placeholder(tf.int32,name="labels")
        self.lung = tf.placeholder(tf.int32,name="lung")
        self.input = tf.reshape(self.x,[BATCH_SIZE,-1,W_SIZE,H_SIZE,1])
        self.global_step = tf.Variable(0, trainable=False)
        self.is_train = True
        self.decay = 0.5
        self.learning_rate = LEARN_RATE
        self.sess = tf.Session()
        self.k = k
        self.net = {}
        self.layers = [
            'CONV_1', 'POOL_1', 'CONV_2','UPSAMPLE_0','CONBINE_0','CONV_3','ADD_0','UPSAMPLE_1','CONBINE_1','CONV_4',
            'CONTINUE','POOL_2', 'DENSE_1', 'TRANS_1', 'DENSE_2',
            'TRANS_2', 'DENSE_3', 'TRANS_3', 'DENSE_4', 'UPSAMPLE_2', 'CONBINE_2', 'DENSE_5', 'ADD_1', 'UPSAMPLE_3', 'CONBINE_3',
            'DENSE_6', 'ADD_2', 'UPSAMPLE_4', 'CONBINE_4', 'CONV_5'
        ]
        self.layers_kernels = [
            {"kernel": [7, 7, 7], "stride": 2, "filter": 64},  # CONV_1
            {"pool_way": Max_Pooling_3d, "kernel": [3, 3, 3], "stride": 2},  # POOL_1
            {"kernel": [5, 5, 5], "stride": 1, "filter": 256},  # CONV_2
            {"kernel": [3, 3, 3], "stride": [2, 2, 2], "filter": 32},  # UPSAMPLE_0
            {"add_layer": 'CONV_1', "kernel": [3, 3, 3]},  # CONBINE_0
            {"kernel": [3, 3, 3], "stride": 1, "filter": 128},  # CONV_3
            {"add_layer": 'CONV_1', "kernel": [3, 3, 3]},    # ADD_0
            {"kernel": [3, 3, 3], "stride": [2, 2, 2], "filter": 16},  # UPSAMPLE_1
            {"add_layer": 'INPUT', "kernel": [3, 3, 3]},  # CONBINE_1
            {"kernel": [3, 3, 3], "stride": 1, "filter": 2},  # CONV_4
            {"name": 'CONV_2'},                             #   CONTINUE
            {"pool_way": Average_pooling_3d, "kernel": [3, 3, 3], "stride": 2},  # POOL_2
            {"num": 8},  # DENSE_1
            {},  # TRANS_1
            {"num": 8},  # DENSE_2
            {},  # TRANS_2
            {"num": 8},  # DENSE_3
            {},  # TRANS_3
            {"num": 8},  # DENSE_4
            {"kernel": [3, 3, 3], "stride": [2, 2, 2], "filter": 64},  # UPSAMPLE_2
            {"add_layer": 'CONV_2', "kernel": [3, 3, 3]},  # CONBINE_2
            {"num": 5},  # DENSE_5
            {"add_layer": 'CONV_2', "kernel": [1, 1, 1]},  # ADD_1
            {"kernel": [3, 3, 3], "stride": [2, 2, 2], "filter": 64},  # UPSAMPLE_3
            {"add_layer": 'CONV_1', "kernel": [3, 3, 3]},  # CONBINE_3
            {"num": 5},  # DENSE_6
            {"add_layer": 'CONV_1', "kernel": [1, 1, 1]},  # ADD_2
            {"kernel": [3, 3, 3], "stride": [2, 2, 2], "filter": 16},  # UPSAMPLE_4
            {"add_layer": 'INPUT', "kernel": [3, 3, 3]},  # CONBINE_4
            {"kernel": [1, 1, 1], "stride": 1, "filter": 3}  # CONV_5
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
                    current = current + 0.6 * add_tensor
                self.net[name] = current
            elif name == 'CONTINUE':
                layer_name = value['name']
                current = self.net[layer_name]
        with tf.variable_scope('train'):
            current_lung = tf.squeeze(self.net['CONV_4'])
            self.cost_lung = tf.nn.softmax_cross_entropy_with_logits(logits=current_lung, labels=self.lung)

            current = tf.squeeze(current)
            self.class_weights = tf.placeholder(tf.float32, name='class_weights')

            self.weight_map = tf.reduce_sum(tf.multiply(tf.cast(self.y, tf.float32), self.class_weights), 3)

            self.cost = tf.nn.softmax_cross_entropy_with_logits(logits=current, labels=self.y)

            self.loss = tf.reduce_mean(self.cost * self.weight_map) + 0.2*tf.reduce_mean(self.cost_lung)

            self.lr = tf.train.exponential_decay(self.learning_rate,
                                            self.global_step,
                                            decay_steps=100,
                                            decay_rate=0.99,
                                            staircase=True)

            self.train_op = tf.train.AdadeltaOptimizer(
                self.lr).minimize(self.loss, global_step=self.global_step)
            tf.summary.scalar('loss', self.loss)

        self.airway_result, self.artery_result, self.lung_result, self.airway_correct_prediction, self.artery_correct_prediction, self.lung_correct_prediction = self.GetResult()

        tf.summary.scalar('total_artery_accurary', total_artery_accurary)
        tf.summary.scalar('total_airway_accurary', total_airway_accurary)

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter("./tensorboard/train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter("./tensorboard/test", self.sess.graph)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.Reload()

    def GetResult(self):
        with tf.variable_scope("GetResult_lung"):
            x = tf.squeeze(self.net['CONV_4'])
            lung_result = tf.cast(tf.argmin(x, 3), tf.float32)
            label_lung,label_bg =  tf.split(self.lung, [1, 1], axis=3)
            constant = tf.cast(tf.equal(label_lung, label_lung), tf.float32)
            label_lung = tf.cast(tf.squeeze(label_lung), tf.float32)
            lung_intersection = label_lung * lung_result
            lung_union = tf.cast(tf.greater_equal((label_lung + lung_result), constant), tf.float32)
            lung_correct_prediction = tf.reduce_mean(lung_intersection) / tf.reduce_mean(lung_union)
        with tf.variable_scope("GetResult"):
            x = tf.squeeze(self.net['CONV_5'])
            out_result = tf.cast(tf.argmax(x, 3), tf.float32)
            constant = tf.cast(tf.equal(out_result, out_result), tf.float32)
            airway_result = tf.cast(tf.equal(out_result, constant), tf.float32)

            artery_result = tf.cast(tf.less(out_result, constant), tf.float32)

            label_artery,label_airway ,label_bg = tf.split(self.y, [1, 1, 1], axis=3)

            label_airway = tf.cast(tf.squeeze(label_airway),tf.float32)

            label_artery = tf.cast(tf.squeeze(label_artery),tf.float32)

            airway_intersection = label_airway * airway_result

            airway_union = tf.cast(tf.greater_equal((label_airway + airway_result), constant), tf.float32)
            zero = tf.constant([0])
            airway_correct_prediction = tf.reduce_mean(airway_intersection) / tf.reduce_mean(airway_union)

            artery_intersection = label_artery * artery_result

            artery_union = tf.cast(tf.greater_equal((label_artery + artery_result), constant), tf.float32)


            artery_correct_prediction = tf.reduce_mean(artery_intersection) / tf.reduce_mean(artery_union)

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
    def RunAccurary(self,artery_accurary,airway_accurary):
        self.artery_accurary = tf.placeholder(tf.float32, name="artery_accurary")
        self.airway_accurary = tf.placeholder(tf.float32, name="airway__accurary")
        tf.summary.scalar('artery_accurary',self.artery_accurary)
        tf.summary.scalar('airway_accurary', self.airway_accurary)
        result, step = self.sess.run((self.merged, self.global_step), feed_dict={self.artery_accurary: artery_accurary,
                                                                                 self.airway_accurary: airway_accurary})
        self.train_writer.add_summary(result,step)

    def Train(self,inputs,labels,labels_lung,weights):
        self.is_train = True
        if os.path.isdir("./tensorboard"):
            pass
        else:
            os.mkdir('tensorboard')


        _, loss, airway,artery,lung,airway_accurary,artery_accurary,lung_accurary,learn_rate, result, step = self.sess.run((
            self.train_op, self.loss,self.airway_result,self.artery_result,self.lung_result,
            self.airway_correct_prediction,self.artery_correct_prediction,self.lung_correct_prediction,
            self.lr, self.merged, self.global_step),feed_dict={self.x:inputs,self.y:labels,self.lung:labels_lung,self.class_weights:weights})

        self.train_writer.add_summary(result, step)
        return loss,airway,artery,lung,airway_accurary,artery_accurary,lung_accurary,learn_rate, step

    def Test(self,inputs,labels,labels_lung,weights):
        self.is_train = False
        if os.path.isdir("./tensorboard"):
            pass
        else:
            os.mkdir('tensorboard')

        loss, airway, artery, lung, airway_accurary, artery_accurary, lung_accurary, learn_rate, result, step = self.sess.run(
            (
                self.loss, self.airway_result, self.artery_result, self.lung_result,
                self.airway_correct_prediction, self.artery_correct_prediction, self.lung_correct_prediction,
                self.lr, self.merged, self.global_step),
            feed_dict={self.x: inputs, self.y: labels, self.lung: labels_lung, self.class_weights: weights})

        self.test_writer.add_summary(result, step)
        return loss,airway,artery,lung,airway_accurary,artery_accurary,lung_accurary,learn_rate, step

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

    label_array = np.zeros([dicom_array.shape[0],dicom_array.shape[1],dicom_array.shape[2],3])
    label_lung_array = np.zeros([dicom_array.shape[0], dicom_array.shape[1], dicom_array.shape[2], 2])

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

    temp = airway_array + artery_array
    temp = temp>0
    label_array[:, :, :, 2] = ~temp

    label_lung_array[:,:,:,0] = lung_array>0
    label_lung_array[:,:,:,1] = ~(lung_array>0)
    return dicom_array,label_array,label_lung_array

def process_datas(dicom_array,label_array,label_lung_array,size,train):
    datas = []
    labels = []
    indexs = []
    label_lung = []
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
                datas.append(dicom_array[c_start:c_end,w_start:w_end,h_start:h_end])
                labels.append(label_array[c_start:c_end,w_start:w_end,h_start:h_end,:])
                label_lung.append(label_lung_array[c_start:c_end,w_start:w_end,h_start:h_end,:])
                indexs.append([c_start,c_end,w_start,w_end,h_start,h_end])
    return datas,labels,label_lung,indexs

net = DenseNet(32)
file_list = os.listdir(root)
file_test_list = os.listdir(root_test)

for times in range(TRAIN_TIME):
    for i, file in enumerate(file_list):
        data_array,label_array,label_lung_array = get_data(root,file,2)
        out_vtk_airway = np.zeros(data_array.shape)
        out_vtk_artery = np.zeros(data_array.shape)
        out_vtk_lung = np.zeros(data_array.shape)
        datas,labels,labels_lung,indexs = process_datas(data_array,label_array,label_lung_array,W_SIZE,train=True)
        data_list = list(zip(datas,labels,labels_lung,indexs))
        random.shuffle(data_list)
        if (i>0):
            weight = [100,100,100]
        else:
            weight = [100,100,1]
        for data,label,label_lung,index in data_list:
            if (np.sum(label[:, :, :, 0] + label[:, :, :, 1]) > 0) or (index[0]*index[3]*index[5]==0):
                artery_percentage = np.sum(label[:, :, :, 0]) / np.size(label[:, :, :, 0])
                airway_percentage = np.sum(label[:, :, :, 1]) / np.size(label[:, :, :, 1])
                lung_percentage = np.sum(label_lung[:, :, :, 0]) / np.size(label_lung[:, :, :, 0])

                loss, airway, artery,lung, airway_accurary, artery_accurary,lung_accurary, learn_rate, step = net.Train(data, label,label_lung,
                                                                                                             weight)
                print('step ', step, ' artery_percentage ', artery_percentage, ' airway_percentage ',
                      airway_percentage,'lung_percentage',lung_percentage ,'learning_rate',learn_rate)
                print(' loss ', loss, ' artery_accurary ', artery_accurary, ' airway_accurary ',
                      airway_accurary,' lung_accurary ', lung_accurary)
                print("=================================================================================")
                c_start, c_end, w_start, w_end, h_start, h_end = (index[k] for k in range(6))
                out_vtk_artery[c_start:c_end, w_start:w_end, h_start:h_end] =  artery
                out_vtk_airway[c_start:c_end, w_start:w_end, h_start:h_end] = airway
                out_vtk_lung[c_start:c_end, w_start:w_end, h_start:h_end] = lung

        total_artery_accurary = np.sum(out_vtk_artery * label_array[:, :, :, 0] > 0) * 1.0 / np.sum(
            out_vtk_artery + label_array[:, :, :, 0] > 0) * 1.0
        total_airway_accurary = np.sum(out_vtk_airway * label_array[:, :, :, 1] > 0) * 1.0 / np.sum(
            out_vtk_airway + label_array[:, :, :, 1] > 0) * 1.0
        total_lung_accurary = np.sum(out_vtk_lung * label_lung_array[:, :, :, 0] > 0) * 1.0 / np.sum(
            out_vtk_lung + label_lung_array[:, :, :, 0] > 0) * 1.0

       # net.RunAccurary(total_artery_accurary,total_airway_accurary)
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


        del out_vtk_artery
        del out_vtk_airway
        del out_vtk_lung
        net.Store()

    for i, file in enumerate(file_test_list):
        data_array, label_array, label_lung_array = get_data(root_test, file, 2)
        out_vtk_airway = np.zeros(data_array.shape)
        out_vtk_artery = np.zeros(data_array.shape)
        out_vtk_lung = np.zeros(data_array.shape)
        datas, labels, labels_lung, indexs = process_datas(data_array, label_array, label_lung_array, W_SIZE,
                                                           train=True)
        data_list = list(zip(datas, labels, labels_lung, indexs))
        random.shuffle(data_list)
        for data, label, label_lung, index in data_list:
            artery_percentage = np.sum(label[:, :, :, 0]) / np.size(label[:, :, :, 0])
            airway_percentage = np.sum(label[:, :, :, 1]) / np.size(label[:, :, :, 1])
            lung_percentage = np.sum(label_lung[:, :, :, 0]) / np.size(label_lung[:, :, :, 0])

            loss, airway, artery, lung, airway_accurary, artery_accurary, lung_accurary, learn_rate, step = net.Test(
                data, label, label_lung,
                [100, 100,
                 100])

            print('test ', step, ' artery_percentage ', artery_percentage, ' airway_percentage ',
                  airway_percentage, 'lung_percentage', lung_percentage, 'learning_rate', learn_rate)
            print(' loss ', loss, ' artery_accurary ', artery_accurary, ' airway_accurary ',
                  airway_accurary, ' lung_accurary ', lung_accurary)
            print("=================================================================================")
            c_start, c_end, w_start, w_end, h_start, h_end = (index[k] for k in range(6))
            out_vtk_artery[c_start:c_end,w_start:w_end,h_start:h_end] = out_vtk_artery[c_start:c_end,w_start:w_end,h_start:h_end] + artery + (artery-1)
            out_vtk_airway[c_start:c_end, w_start:w_end, h_start:h_end] = out_vtk_airway[c_start:c_end,w_start:w_end,h_start:h_end] + airway + (airway-1)
            out_vtk_lung[c_start:c_end, w_start:w_end, h_start:h_end] = out_vtk_lung[c_start:c_end, w_start:w_end,
                                                                          h_start:h_end] + lung + (lung - 1)
        out_vtk_artery[out_vtk_artery>=0] = 1
        out_vtk_artery[out_vtk_artery < 0] = 0
        out_vtk_airway[out_vtk_airway >= 0] = 1
        out_vtk_airway[out_vtk_airway < 0] = 0

        out_vtk_lung[out_vtk_lung >= 0] = 1
        out_vtk_lung[out_vtk_lung < 0] = 0

        total_artery_accurary = np.sum(out_vtk_artery * label_array[:, :, :, 0] > 0) * 1.0 / np.sum(
            out_vtk_artery + label_array[:, :, :, 0] > 0) * 1.0
        total_airway_accurary = np.sum(out_vtk_airway * label_array[:, :, :, 1] > 0) * 1.0 / np.sum(
            out_vtk_airway + label_array[:, :, :, 1] > 0) * 1.0
        total_lung_accurary = np.sum(out_vtk_lung * label_lung_array[:, :, :, 0] > 0) * 1.0 / np.sum(
            out_vtk_lung + label_lung_array[:, :, :, 0] > 0) * 1.0
        # net.RunAccurary(total_artery_accurary,total_airway_accurary)

        print("=================================================================================")
        print("=================================================================================")
        print("=================================================================================")
        print("time: " + str(times + 1) + "  test: " + str(i + 1) + " total_artery_accurary: " + str(
            total_artery_accurary) + " total_airway_accurary: " + str(total_airway_accurary)+ " total_lung_accurary: " + str(total_lung_accurary))
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