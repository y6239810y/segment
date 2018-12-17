from get_data import *
from statistics import *
import numpy as np
import random,re

def net_train(model, root, weights, times, filter_no_liver = 10):  # 训练执行步骤

    C_SIZE = model.batch_size
    W_SIZE = model.width
    H_SIZE = model.height
    THRESHOLD = model.threshold

    file_list = [file for file in os.listdir(root) if int(re.sub("\D", "", file)) <= 130and "volume" in file]

    random.shuffle(file_list)

    for i, file in enumerate(file_list):

        data_array, label_array = read_data(root, file, random.randint(0,4))

        out_vtk_liver = np.zeros(data_array.shape)

        indexs = chop_datas(data_array, W_SIZE, H_SIZE, C_SIZE, train=True)  # 获取到每个长方体的起始点的坐标索引
        data_list = list(indexs)
        random.shuffle(data_list)
        for j, index in enumerate(data_list):  # 遍历索引，并将长方体分别送入网络执行训练
            c_start, c_end, w_start, w_end, h_start, h_end = (index[k] for k in range(6))
            data = data_array[c_start:c_end, w_start:w_end, h_start:h_end]
            label = label_array[c_start:c_end, w_start:w_end, h_start:h_end]

            if (np.sum(label[:, :, :, 0]) > 0 or j % filter_no_liver == 0):  # 过滤掉部分无目标的长方体
                liver_percentage = np.sum(label[:, :, :, 0]) / np.size(label[:, :, :, 0])

                loss, liver,liver_iou,learn_rate, step = model._train(data, label, weights)
                out_vtk_liver[c_start:c_end, w_start:w_end, h_start:h_end] = liver

                if (j % 1 == 0):
                    print('step ', step, ' liver_percentage ', liver_percentage, 'learning_rate', learn_rate)
                    print(' loss ', loss, ' liver_iou ', liver_iou)
                    print("=================================================================================")

        over = (i == len(file_list) - 1)
        ious = []
        for thr in [0.4,0.5,0.6]:
            out_vtk_liver_with_thr = 1.0*(out_vtk_liver>thr)
            label_array_obj = 1.0*(label_array[:, :, :, 0] > thr)

            total_iou = np.sum(1.0 * out_vtk_liver_with_thr * label_array_obj)/(1.0*np.sum((out_vtk_liver_with_thr+label_array_obj)>0))
            ious.append(total_iou)

        add_train(times,model.save_path,file,ious,over)


        out_vtk_liver = 1.0 * (out_vtk_liver > THRESHOLD)
        label_array_obj = 1.0 * (label_array[:, :, :, 0] > THRESHOLD)

        total_dice = np.sum(2.0 * out_vtk_liver * label_array_obj) / (1.0 * np.sum(out_vtk_liver + label_array_obj))
        total_iou = np.sum(1.0 * out_vtk_liver * label_array_obj) / (
                    1.0 * np.sum((out_vtk_liver + label_array_obj) > 0))

        print(total_dice,total_iou)
        model._run_accurary(total_iou)  # 将整体准确度上传tensorboard
        print("=================================================================================")
        print("=================================================================================")
        print("=================================================================================")
        print(
            "time: " + str(times) + "  Dicom: " + str(i + 1) + " total_dice: " + str(total_dice),
            " total_iou: " + str(total_iou),
        )
        print("\n\n\n")
        del out_vtk_liver