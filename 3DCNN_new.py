import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import os
import random
import platform

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

is_train = True
data_path_windows = "D:/study/liver"
data_path_linux = "/opt/liver"
def set_Window(image, max, min):
    array = sitk.GetArrayFromImage(image)
    array_max = np.max(array)
    array_min = np.min(array)
    image_out = sitk.IntensityWindowing(image, array_min * 1.0, array_max * 1.0, min, max)
    return image_out

def get_batch(times,number):
    sysstr = platform.system()
    if sysstr == "Windows":
        dir = data_path_windows + "/3Dircadb1." + str(times) +"/PATIENT_DICOM"
        dir2 = data_path_windows + "/3Dircadb1." + str(times) + "/MASKS_DICOM/liver"
    else:
        dir = data_path_linux + "/3Dircadb1." + str(times) + "/PATIENT_DICOM"
        dir2 = data_path_linux + "/3Dircadb1." + str(times) + "/MASKS_DICOM/liver"
    if not os.path.isdir(dir):
        return None
    reader = sitk.ImageSeriesReader()
    dicoms = reader.GetGDCMSeriesFileNames(dir)
    reader.SetFileNames(dicoms)
    image = reader.Execute()
    dicom_array = sitk.GetArrayFromImage(set_Window(image, 255, 0))

    # plies = dicom_array.shape[0]
    # batch_size = 32
    # step_size = batch_size//2
    # nums = (plies+step_size-1)//step_size
    # dicom_batch = []
    # label_batch = []
    # start = []
    # end = []

    if not os.path.isdir(dir2):
        return None
    reader2 = sitk.ImageSeriesReader()
    dicoms2 = reader.GetGDCMSeriesFileNames(dir2)
    reader2.SetFileNames(dicoms2)
    image2 = reader2.Execute()
    # sitk.WriteImage(image2,'liver'+ str(times) + '.vtk')
    mask_dicom_array = sitk.GetArrayFromImage(image2)


    if number == 1:
        dicom_array = dicom_array
        mask_dicom_array = mask_dicom_array
    elif number == 2:
        dicom_array = np.rot90(dicom_array,axes=(1,2))
        mask_dicom_array = np.rot90(mask_dicom_array, axes=(1, 2))
    elif number == 3:
        dicom_array = np.rot90(np.rot90(dicom_array, axes=(1, 2)), axes=(1,2))
        mask_dicom_array = np.rot90(np.rot90(mask_dicom_array, axes=(1, 2)), axes=(1, 2))
    else:
        dicom_array = np.rot90(np.rot90(np.rot90(dicom_array, axes=(1, 2)), axes=(1, 2)),axes=(1, 2))
        mask_dicom_array = np.rot90(np.rot90(np.rot90(mask_dicom_array, axes=(1, 2)), axes=(1, 2)), axes=(1, 2))

    mask_dicom_array = mask_dicom_array.transpose(1,2,0)
    dicom_array = dicom_array.transpose(1, 2, 0)

    # for i in range(0,nums-1):
    #     start.append(i * step_size)
    #     end.append(start[i]+batch_size)
    #     if end[i] > plies:
    #         end[i] = plies
    #         start[i] = end[i] -batch_size
    #     else:
    #         pass
    #     # print(str(start[i]) + "-" + str(end[i]))
    #     label_batch.append(mask_dicom_array[start[i]:end[i],:,:])
    #     dicom_batch.append(dicom_array[start[i]:end[i],:,:])
    # dicom_array = dicom_batch[num]
    # mask_dicom_array = label_batch[num]

    mask_dicom_array = mask_dicom_array > 0
    #    start = mask_dicom_array.nonzero()[0][0]
    #    end = mask_dicom_array.nonzero()[0][len(mask_dicom_array.nonzero()[0]) - 1]
    label_array = np.zeros([mask_dicom_array.shape[0], mask_dicom_array.shape[1], mask_dicom_array.shape[2], 2])
    label_array[:, :, :, 0] = mask_dicom_array
    label_array[:, :, :, 1] = ~mask_dicom_array

    dicom_array = dicom_array[::2,::2,:]
    label_array = label_array[::2,::2,:,:]

    return dicom_array,label_array

def get_weight(pre, label, weight):
    if (weight+(label-pre)) > 1:
        return 0.99

    elif(weight+(label-pre)) < 0.5:
        return 0.5
    else:
        if (label-pre) > 0:
            return 0.9
        else:
            return weight+(label-pre)

def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def conv3d_with_relu(input, w, b, strides, number, padding='SAME', name=None, ):
    conv = tf.nn.conv3d(input, w, strides, padding=padding, name=name)
    conv_bias = tf.nn.bias_add(conv, b)
    return tf.nn.relu(conv_bias, name='hidden' + str(number))


def mean_pooling(input, shape, strides, name, padding='SAME'):
    return tf.nn.avg_pool3d(input, shape, strides=strides, padding=padding, name=name)


def double_size(input):
    x = tf.shape(input)[1]
    y = tf.shape(input)[2]
    z = tf.shape(input)[3]
    c = tf.shape(input)[4]
    return tf.reshape(input, [1, z * 2, x * 2, y * 2, c // 8])


layer = dict()
net = {}
layer_name = ['conv_1', 'add_conv_1', 'pooling_1', 'add_conv_2', 'conv_2', 'pooling_2',
              'add_conv_3', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7',
              'transpose_1', 'conv_8', 'transpose_2', 'conv_9',
              'transpose_3', 'conv_10', 'conv_11', 'log_regression']
layer_value = [
    ([7, 7, 9, 1, 96], [1, 2, 2, 2, 1]),  # conv_1
    ([7, 7, 9, 96, 128], [1, 1, 1, 1, 1]),  # add_conv_1
    ([1, 3, 3, 2, 1], [1, 2, 2, 2, 1]),  # pooling_1
    ([5, 5, 5, 96, 512], [1, 1, 1, 1, 1]),  # add_conv_2
    ([5, 5, 5, 96, 256], [1, 1, 1, 1, 1]),  # conv_2
    ([1, 3, 3, 2, 1], [1, 2, 2, 2, 1]),  # pooling_2
    ([3, 3, 3, 256, 512], [1, 1, 1, 1, 1]),  # add_conv_3
    ([3, 3, 3, 256, 512], [1, 1, 1, 1, 1]),  # conv_3
    ([3, 3, 3, 512, 512], [1, 1, 1, 1, 1]),  # conv_4
    ([3, 3, 3, 512, 512], [1, 1, 1, 1, 1]),  # conv_5
    ([3, 3, 3, 512, 512], [1, 1, 1, 1, 1]),  # conv_6
    ([3, 3, 3, 512, 512], [1, 1, 1, 1, 1]),  # conv_7
    ([3, 3, 3, 64, 512], [1, 2, 2, 2, 1]),  # transpose_1
    ([3, 3, 3, 64, 512], [1, 1, 1, 1, 1]),  # conv_8
    ([3, 3, 3, 64, 512], [1, 2, 2, 2, 1]),  # transpose_2
    ([3, 3, 3, 64, 128], [1, 1, 1, 1, 1]),  # conv_9
    ([3, 3, 3, 16, 128], [1, 2, 2, 2, 1]),  # transpose_3
    ([3, 3, 3, 16, 16], [1, 1, 1, 1, 1]),  # conv_10
    ([3, 3, 3, 16, 2], [1, 1, 1, 1, 1]),  # conv_11
    ()  # log_regression
]

input = tf.placeholder(tf.float32, name='input')
annotations = tf.placeholder(tf.float32, name='annotations')
current = tf.reshape(input, [1, 256, 256, -1, 1])
x = current
count = 0
loss_weight = 0
for name, value in zip(layer_name, layer_value):
    if name[:4] == 'conv':
        strides = value[1]
        kernels = weight_variable(value[0], name=name + '_w')
        loss_weight = tf.nn.l2_loss(kernels) + loss_weight
        bias = bias_variable([value[0][4]], name=name + '_b')
        current = conv3d_with_relu(current, kernels, bias, strides, count)
        net[name] = current
        current = tf.layers.batch_normalization(current, training=is_train)
        net[name + "_BN"] = current
    elif name[:4] == 'pool':
        strides = value[1]
        kernels = value[0]
        current = mean_pooling(current, kernels, strides, name)
        net[name] = current
    elif name[:3]=='add':
        strides = value[1]
        kernels = weight_variable(value[0], name=name + '_w')
        loss_weight = tf.nn.l2_loss(kernels) + loss_weight
        bias = bias_variable([value[0][4]], name=name + '_b')
        net[name] = conv3d_with_relu(current, kernels, bias, strides, name)

    elif name[:9] == 'transpose':
        strides = value[1]
        kernels = weight_variable(value[0], name=name + '_w')
        if name[-1] == '1':
            current = tf.nn.conv3d_transpose(current + 0.6 * net['add_conv_3'], kernels,
                                             [1, tf.shape(net['conv_2'])[1], tf.shape(net['conv_2'])[2],
                                              tf.shape(net['conv_2'])[3], 64], strides=[1,2,2,2,1], padding="SAME")
        elif name[-1] == '2':
            current = tf.layers.batch_normalization(current, training=is_train)
            net[name + "_BN"] = current
            current = tf.nn.conv3d_transpose(current + 0.6 * net['add_conv_2'], kernels,
                                             [1, tf.shape(net['conv_1'])[1], tf.shape(net['conv_1'])[2],
                                              tf.shape(net['conv_1'])[3], 64], strides, padding="SAME")
        else:
            current = tf.layers.batch_normalization(current, training=is_train)
            net[name + "_BN"] = current
            current = tf.nn.conv3d_transpose(current + 0.6 * net['add_conv_1'], kernels,
                                             [1, tf.shape(x)[1], tf.shape(x)[2],
                                              tf.shape(x)[3], 16], strides, padding="SAME")
        net[name] = current
        # current = double_size(current)
    else:
        current = current
        net[name] = current
    count = count + 1
class_weights = tf.placeholder(tf.float32, name='class_weights')

weight_map = tf.reduce_sum(tf.multiply(annotations, class_weights), 3)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=annotations,
                                                        logits=tf.squeeze(net['log_regression']))
loss = tf.reduce_mean(cross_entropy * weight_map) + loss_weight * 0.001

global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(0.1,
                                           global_step,
                                           decay_steps=1000,
                                           decay_rate=0.95,
                                           staircase=True)

train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

index = tf.argmin(tf.squeeze(net['log_regression']), 3)

# dicom_input, label_input = get_batch(1)

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
if os.path.isdir("./model"):
    pass
else:
    os.mkdir('model')
ckpt = tf.train.get_checkpoint_state("./model")
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored...")
best_avg_acc = 0
last_avg_acc=0
avg_add=0
bias_w = 0.9
white_proportion = 0.1  #liver percentage
label_proportion = 0.1
t = 0
dicom_count=0
for i in range(1000):
    avg_acc = 0
    countt = 0
    if avg_add>0:
        ls=ls
    else:
        ls = list(range(1,46))
        random.shuffle(ls)
    dicom_count = 0
    for j in range(1,46):
        is_train = True
        dicom_count += 1
        z=random.randint(1,4)
        t=t+1
        dicom_array, mask_dicom_array=get_batch(j,z)
        plies = dicom_array.shape[2]
        batch_size = 32
        step_size = batch_size//2
        nums = (plies+step_size-1)//step_size
        output_sum=np.zeros([dicom_array.shape[0],dicom_array.shape[1],dicom_array.shape[2]])
        ls2=list(range(0,nums-1))
        random.shuffle(ls2)
        for k in ls2:
            start = k*step_size
            end = start + batch_size
            if end>plies:
                end = plies
                start=end-batch_size
            else:
                pass
            label_input = mask_dicom_array[:, :, start:end,:]
            dicom_input = dicom_array[:,:,start:end]

            bias_w = get_weight(white_proportion, label_proportion, bias_w)

            label_proportion = np.mean(label_input[:, :, :, 0])



            a,b = (sess.run((train_step,tf.shape(net['add_conv_1'])),
                            feed_dict={input: dicom_input, annotations: label_input, class_weights: [bias_w, 1-bias_w]}))


            output, steps,loss_all,lr = sess.run((index, global_step,loss,learning_rate),
                                     feed_dict={input: dicom_input, annotations: label_input,class_weights: [bias_w, 1-bias_w]})

            white_proportion = np.sum(output) / output.size
            output_sum[:,:,start:end] = output

            acc = np.sum((output * label_input[:, :, :, 0]) > 0) * 1.0 / np.sum((output + label_input[:, :, :, 0]) > 0) * 1.0
            print("step: "+str(steps) + " Dicom: " +str(j)+ " start-end: " + str(start)+"~" + str(end) +"  accuracy: "+str(acc)+"  learning_rate: "+str(lr)+" white_proportion:"+str(white_proportion)+" roate:"+str(z))
        accuracy = np.sum((output_sum * mask_dicom_array[:, :, :, 0]) > 0) * 1.0 / np.sum(
            (output_sum + mask_dicom_array[:, :, :, 0]) > 0) * 1.0
        print("------------------------------------")
        print("time: "+ str(dicom_count) +"  Dicom: "+ str(j) + " accuracy: " +str(accuracy))
        print('\n')

    for j in range(46,50):
        is_train = False
        countt = countt + 1
        dicom_array, mask_dicom_array = get_batch(j,1)
        plies = dicom_array.shape[2]
        batch_size = 32
        step_size = batch_size // 2
        nums = (plies + step_size - 1) // step_size
        output_sum = np.zeros([dicom_array.shape[0], dicom_array.shape[1], dicom_array.shape[2]])
        for k in range(nums-1):
            start = k * step_size
            end = start + batch_size
            if end > plies:
                end = plies
                start = end - batch_size
            else:
                pass
            label_input = mask_dicom_array[:, :, start:end, :]
            dicom_input = dicom_array[:, :, start:end]
            output = sess.run((index),feed_dict={input: dicom_input})
            output_sum[:, :, start:end] = output
        accuracy = np.sum((output_sum * mask_dicom_array[:, :, :, 0]) > 0) * 1.0 / np.sum(
            (output_sum + mask_dicom_array[:, :, :, 0]) > 0) * 1.0
        avg_acc = avg_acc + accuracy
        output_sum = output_sum.transpose(2,0,1)
        image_output = sitk.GetImageFromArray(output_sum)
        sitk.WriteImage(image_output, 'out_' + str(j) + '.vtk')
        print("Dicom: "+ str(j) + "  accuracy:  "+str(accuracy))
        print('\n')
    avg_acc = avg_acc/countt
    if(avg_acc>best_avg_acc):
        best_avg_acc = avg_acc
        saver.save(sess,"./model/model.ckpt")
    else:
        pass
    avg_add = avg_acc-last_avg_acc
    print("------------------------------------")
    print("avg_accuracy: " + str(avg_acc) +"  add_accuracy: " +str(avg_add) +"  best_avg_acc:"+str(best_avg_acc))
    print("------------------------------------")
    print('\n')
    last_avg_acc = avg_acc
# for j in range(1,34):
#     dicom_input, label_input = get_batch(j,1)
#     output= sess.run((index),
#                       feed_dict={input: dicom_input, annotations: label_input, class_weights: [0.95, 0.05]})
#     accuracy = np.sum((output * label_input[:, :, :, 0]) > 0)* 1.0 / np.sum((output + label_input[:, :, :, 0]) > 0)* 1.0
#     output = output * 255.0
#     image_output = sitk.GetImageFromArray(output)
#     dicom_input = sitk.GetImageFromArray(dicom_input*1.0)
#     image_output.SetSpacing([1, 1, 1.5])
#     sitk.WriteImage(image_output, 'out_' + str(j) + '.vtk')
#     sitk.WriteImage(dicom_input, 'input_' + str(j) + '.vtk')
#     print(str(j) + ': ' + str(accuracy))


# output= sess.run((index),
#                   feed_dict={input: dicom_input})
# output = output * 255.0
# image_output = sitk.GetImageFromArray(output)
# image_output.SetSpacing([1, 1, 1.5])
# sitk.WriteImage(image_output, 'out_test' + '.vtk')
# out = output + dicom_input
# image_out = sitk.GetImageFromArray(out)
# image_out.SetSpacing([1, 1, 1.5])
# sitk.WriteImage(image_out, 'out_dicom' + '.vtk')









