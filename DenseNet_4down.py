import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from tools import *
import platform
import random
W_SIZE = 128
H_SIZE = 128
CLASS_NUM = 1000
LEARN_RATE = 0.1
TRAIN_TIME = 100000
TRANS_LAYER_SIZE = 1

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sysstr = platform.system()

if sysstr == "Windows":
    BATCH_SIZE = 1
    root  = "E:/multi_task_data/"
else:
    BATCH_SIZE = 1
    root = "/opt/multi_task_data_train/"


class DenseNet:
    def __init__(self,k):
        self.x = tf.placeholder(tf.float32,name="input_images")
        self.y = tf.placeholder(tf.int32,name="labels")
        self.input = tf.reshape(self.x,[BATCH_SIZE,-1,W_SIZE,H_SIZE,1])
        self.global_step = tf.Variable(0, trainable=False)
        self.is_train = True
        self.decay = 0.5
        self.learning_rate = LEARN_RATE
        self.sess = tf.Session()
        self.k = k
        self.net = {}
        self.layers = [
            'CONV_1','POOL_1','CONV_2','POOL_2','DENSE_1','TRANS_1','CONV_3','POOL_3','DENSE_2',
            'TRANS_2', 'UPSAMPLE_1', 'CONBINE_1', 'DENSE_3','TRANS_3','UPSAMPLE_2','CONBINE_2',
            'DENSE_4','TRANS_4','UPSAMPLE_3','CONBINE_3','DENSE_5','TRANS_5','UPSAMPLE_4','CONBINE_4',
            'CONV_4'
        ]
        self.layers_kernels = [
            {"kernel": [7, 7, 9], "stride": 2, "filter": 64},    # CONV_1
            {"pool_way":Max_Pooling_3d, "kernel": [3, 3, 3], "stride": 2},                  # POOL_1
            {"kernel": [5, 5, 5], "stride": 1, "filter": 256},   # CONV_2
            {"pool_way":Average_pooling_3d,"kernel": [3, 3, 3], "stride": 2},                  # POOL_2
            {"num":8},                                           # DENSE_1
            {},                                                  # TRANS_1
            {"kernel": [1, 1, 1], "stride": 1, "filter": 256},  # CONV_3
            {"pool_way":Average_pooling_3d,"kernel": [2, 2, 2], "stride": 2},                  # POOL_3
            {"num": 8},                                          # DENSE_2
            {},                                                  # TRANS_2
            {"kernel": [2, 2, 2], "stride": [2,2,2], "filter": 128},   # UPSAMPLE_1
            {"add_layer":'TRANS_1'},                             # CONBINE_1
            {"num": 2},                                          # DENSE_3
            {},                                                  # TRANS_3

            {"kernel": [3, 3, 3], "stride": [2,2,2], "filter": 64},    # UPSAMPLE_2
            {"add_layer": 'CONV_2'},                             # CONBINE_2
            {"num": 5},                                          # DENSE_4
            {},                                                  # TRANS_4

            {"kernel": [3, 3, 3], "stride": [2,2,2], "filter": 64},  # UPSAMPLE_3
            {"add_layer": 'CONV_1'},                           # CONBINE_3
            {"num": 5},                                        # DENSE_5
            {},                                                # TRANS_5

            {"kernel": [3, 3, 3], "stride": [2,2,2], "filter": 16},  # UPSAMPLE_4
            {"add_layer": 'INPUT'},                            # CONBINE_4
            {"kernel": [1, 1, 1], "stride": 1, "filter": 3}     #CONV_4

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
                    current = pool_way(x=current,pool_size=pool_kernel,strides=pool_stride)
                self.net[name] = current

            elif name[:-2] == 'DENSE':
                current = self.DenseBlock(current, value['num'], self.k, name)
                self.net[name] = current

            elif name[:-2] == 'TRANS':
                conv_filter = current.shape[-1]
                with tf.variable_scope(name):
                    current = tf.contrib.slim.batch_norm(current, is_training=self.is_train, scope='BN')
                    current = Relu(current,name='RELU')
                    current = Conv3d(current,filter=conv_filter,kernel=[1,1,1])
                self.net[name] = current

            elif name[:-2] == 'UPSAMPLE':
                up_kernel = value['kernel']
                up_stride = value['stride']
                up_filter = value['filter']
                with tf.variable_scope(name):
                    current = Upsample_3d(input=current,kernel=up_kernel,filter=up_filter,strides=up_stride)
                self.net[name] = current

            elif name[:-2] == 'CONBINE':
                with tf.variable_scope(name):
                    layer_name = value['add_layer']
                    layer = self.net[layer_name]
                    conbine_tensor = tf.contrib.slim.batch_norm(layer, is_training=self.is_train, scope='BN')
                    conbine_tensor = Relu(conbine_tensor,name='RELU')
                    conbine_tensor = Conv3d(conbine_tensor,filter=current.shape[-1]//2,kernel=[1,1,1])
                    current = tf.concat([current, conbine_tensor], 4)
                self.net[name] = current
        print()
        with tf.variable_scope('train'):
            self.class_weights = tf.placeholder(tf.float32, name='class_weights')

            self.weight_map = tf.reduce_sum(tf.multiply(tf.cast(self.y,tf.float32), self.class_weights), 3)
            current = tf.squeeze(current)

            self.cost = tf.nn.softmax_cross_entropy_with_logits(logits=current, labels=self.y)

            self.loss = tf.reduce_mean(self.cost * self.weight_map)

            self.lr = tf.train.exponential_decay(self.learning_rate,
                                            self.global_step,
                                            decay_steps=10000,
                                            decay_rate=0.9,
                                            staircase=True)

            self.train_op = tf.train.AdadeltaOptimizer(
                self.lr).minimize(self.loss, global_step=self.global_step)
            tf.summary.scalar('loss', self.loss)

        self.airway_result, self.artery_result,self.airway_correct_prediction,self.artery_correct_prediction=self.GetResult()

        tf.summary.scalar('air_way_accuracy', self.airway_correct_prediction)
        tf.summary.scalar('artery_correct_prediction', self.artery_correct_prediction)

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter("./tensorboard/train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter("./tensorboard/test", self.sess.graph)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        # self.Reload()

    def GetResult(self):
        with tf.variable_scope("GetResult"):
            x = tf.squeeze(self.net['CONV_4'])
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

        return airway_result,artery_result,airway_correct_prediction,artery_correct_prediction

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

    def Train(self,inputs,labels,weights):
        if os.path.isdir("./tensorboard"):
            pass
        else:
            os.mkdir('tensorboard')


        _, loss, airway,artery,airway_accurary,artery_accurary,learn_rate, result, step = self.sess.run((
            self.train_op, self.loss,self.airway_result,self.artery_result,
            self.airway_correct_prediction,self.artery_correct_prediction,
            self.lr, self.merged, self.global_step),feed_dict={self.x:inputs,self.y:labels,self.class_weights:weights})

        self.train_writer.add_summary(result, step)
        return loss,airway,artery,airway_accurary,artery_accurary,learn_rate, result, step

    def Test(self,inputs,labels):
        if os.path.isdir("./tensorboard"):
            pass
        else:
            os.mkdir('tensorboard')

        loss, airway, artery, airway_accurary, artery_accurary, learn_rate, result, step = self.sess.run((
            self.loss, self.airway_result, self.artery_result,
            self.airway_correct_prediction, self.artery_correct_prediction,
            self.lr, self.merged, self.global_step), feed_dict={self.x: inputs, self.y: labels})

        self.test_writer.add_summary(result, step)
        return loss, airway, artery, airway_accurary, artery_accurary, learn_rate, result, step

# model=DenseNet(32)
def set_Window(image, max, min):
    array = sitk.GetArrayFromImage(image)
    array_max = np.max(array)
    array_min = np.min(array)
    image_out = sitk.IntensityWindowing(image, array_min * 1.0, array_max * 1.0, min, max)
    return image_out


def get_data(file,size):
    print(file + "  reading")
    reader = sitk.ImageSeriesReader()
    dicoms = reader.GetGDCMSeriesFileNames(root+file+'/original1')
    reader.SetFileNames(dicoms)
    image = reader.Execute()
    dicom_array = sitk.GetArrayFromImage(set_Window(image, 1024, 0))

    label_array = np.zeros([dicom_array.shape[0],dicom_array.shape[1],dicom_array.shape[2],3])

    artery = reader.GetGDCMSeriesFileNames(root+file+'/airway')
    reader.SetFileNames(artery)
    image = reader.Execute()
    artery_array = sitk.GetArrayFromImage(image)


    airway = reader.GetGDCMSeriesFileNames(root+file+'/artery')
    reader.SetFileNames(airway)
    image = reader.Execute()
    airway_array = sitk.GetArrayFromImage(image)



    label_array[:, :, :, 0] = artery_array
    label_array[:, :, :, 1] = ((airway_array - artery_array) == 1.0)

    temp = airway_array + artery_array
    temp = temp>0
    label_array[:, :, :, 2] = ~temp

    return dicom_array,label_array

def process_datas(dicom_array,label_array,size):
    datas = []
    labels = []
    indexs = []
    shape = dicom_array.shape
    plies = shape[0]
    w = shape[1]
    h = shape[2]
    step_size = size//2
    c_nums = (plies + step_size - 1) // step_size
    w_nums = (w + step_size - 1) // step_size
    h_nums = (h + step_size - 1) // step_size
    count=0
    for i in range(c_nums-1):
        c_start = i * step_size
        c_end = c_start + size

        if c_end > plies:
            c_end = plies
            c_start = c_end - size
        else:
            pass
        for j in range(w_nums-1):
            w_start = j * step_size
            w_end = w_start + size

            if w_end > w:
                w_end = w
                w_start = w_end - size
            else:
                pass
            for k in range(h_nums - 1):
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
                indexs.append([c_start,c_end,w_start,w_end,h_start,h_end])
    return datas,labels,indexs

net = DenseNet(32)
file_list = os.listdir(root)
for times in range(TRAIN_TIME):
    for i, file in enumerate(file_list):
        data_array,label_array = get_data(file,2)
        out_vtk_airway = np.zeros(data_array.shape)
        out_vtk_artery = np.zeros(data_array.shape)
        datas,labels,indexs = process_datas(data_array,label_array,W_SIZE)
        data_list = list(zip(datas,labels,indexs))
        random.shuffle(data_list)
        for data,label,index in data_list:
            if  np.sum(label[:,:,:,0]+label[:,:,:,1])>0:
                loss, airway, artery, airway_accurary, artery_accurary, learn_rate, result, step = net.Train(data, label,
                                                                                                             [0.99, 0.99,
                                                                                                              0.01])
                print('step ',step, ' loss ',loss,' airway_accurary ',airway_accurary,' artery_accurary ',artery_accurary)
                c_start, c_end, w_start, w_end, h_start, h_end = (index[k] for k in range(6))
                out_vtk_artery[c_start:c_end,w_start:w_end,h_start:h_end] = artery
                out_vtk_airway[c_start:c_end, w_start:w_end, h_start:h_end] = airway

        total_artery_accurary = np.sum(out_vtk_artery * label_array[:, :, :, 0] > 0) * 1.0 / np.sum(
            out_vtk_artery + label_array[:, :, :, 0] > 0) * 1.0
        total_airway_accurary = np.sum(out_vtk_airway * label_array[:, :, :, 1] > 0) * 1.0 / np.sum(
            out_vtk_airway + label_array[:, :, :, 1] > 0) * 1.0
        print("------------------------------------")
        print("time: " + str(times+1) + "  Dicom: " + str(i+1) + " total_artery_accurary: " + str(
            total_artery_accurary) + " total_airway_accurary: " + str(total_airway_accurary))
        del out_vtk_artery
        del out_vtk_airway
        net.Store()
# vtk = np.zeros([512,512,200])


# classes = get_classes()

# for i in range(TRAIN_TIME):
#     datas,image_names = get_data(classes)
#     ls = list(range(1000))
#     random.shuffle(ls)
#     batchs = np.zeros([BATCH_SIZE,224,224,3])
#     labels = np.zeros([BATCH_SIZE])
#     acc_avg = 0
#     for j,k in enumerate(ls):
#         input_image = cv2.resize(datas[k], (224, 224), interpolation=cv2.INTER_CUBIC)
#         batchs[j%BATCH_SIZE,:,:,:]=input_image
#         labels[j%BATCH_SIZE]=k
#         if (j+1)%BATCH_SIZE == 0:
#             cos, prediction, learn_rate,result, step = model.Train(
#                 batchs,labels
#             )
#             print("step:",step,"cos",np.mean(cos),"accuracy",prediction,"learn_rate",learn_rate)
#
#             acc_avg += prediction
#             batchs[:,:,:,:] = 0
#             labels[:] = 0
#     acc_avg = acc_avg/(1000/BATCH_SIZE)
#     print()
#     print("acc_avg",acc_avg)
#     if (i%10==0):
#         model.Store()
