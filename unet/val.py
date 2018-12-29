from get_data import *
import numpy as np
import random, re
from statistics import *
import time
from skimage import measure


def net_val(model, root, weights, times, down_sample):  # 执行评估步骤
    avg_liver = []
    C_SIZE = model.batch_size
    W_SIZE = model.width
    H_SIZE = model.height
    THRESHOLD = model.threshold

    file_list = [file for file in os.listdir(root) if
                 int(re.sub("\D", "", file)) > 110 and int(re.sub("\D", "", file)) <= 130 and "volume" in file]

    for i, file in enumerate(file_list):
        t0 = time.time()
        data_array, label_array = read_data(root, file, 1, down_sample)

        out_vtk_liver = np.zeros(data_array.shape)

        indexs = chop_datas(data_array, W_SIZE, H_SIZE, C_SIZE, train=False)  # 获取到每个长方体的起始点的坐标索引
        data_list = list(indexs)

        for j, index in enumerate(data_list):  # 遍历索引，并将长方体分别送入网络执行训练
            c_start, c_end, w_start, w_end, h_start, h_end = (index[k] for k in range(6))
            data = data_array[c_start:c_end, w_start:w_end, h_start:h_end]
            label = label_array[c_start:c_end, w_start:w_end, h_start:h_end]

            loss, liver, liver_iou, learn_rate, step = model._val(data, label, weights)

            out_vtk_liver[c_start:c_end, w_start:w_end, h_start:h_end] = liver

        over = (i == len(file_list) - 1)
        ious = []
        for thr in [0.4, 0.5, 0.6]:
            out_vtk_liver_with_thr = 1.0 * (out_vtk_liver > thr)
            label_array_obj = 1.0 * (label_array[:, :, :, 0] > thr)

            total_iou = np.sum(1.0 * out_vtk_liver_with_thr * label_array_obj) / (
                    1.0 * np.sum((out_vtk_liver_with_thr + label_array_obj) > 0))
            ious.append(total_iou)

        add_val(times, model.save_path, file, ious, over)

        out_vtk_liver = 1.0 * (out_vtk_liver > THRESHOLD)
        label_array_obj = 1.0 * (label_array[:, :, :, 0] > THRESHOLD)

        # post process
        # pred_seg = measure.label(out_vtk_liver, 4)
        # props = measure.regionprops(pred_seg)
        #
        # max_area = 0
        # max_index = 0
        # for index, prop in enumerate(props, start=1):
        #     if prop.area > max_area:
        #         max_area = prop.area
        #         max_index = index
        #
        # pred_seg[pred_seg != max_index] = 0
        # pred_seg[pred_seg == max_index] = 1
        #
        # out_vtk_liver = pred_seg.astype(np.uint8)

        total_dice = np.sum(2.0 * out_vtk_liver * label_array_obj) / (1.0 * np.sum(out_vtk_liver + label_array_obj))
        total_iou = np.sum(1.0 * out_vtk_liver * label_array_obj) / (
                    1.0 * np.sum((out_vtk_liver + label_array_obj) > 0))
        print("before: ", ious, "after: ", total_iou)
        model._run_accurary(total_iou)  # 将整体准确度上传tensorboard
        print("=================================================================================")
        print("=================================================================================")
        print("=================================================================================")
        print(
            "time: " + str(times) + "  test: " + str(i + 1) + " total_dice: " + str(total_dice),
            " total_iou: " + str(total_iou),
        )
        t = time.time() - t0
        print("cost time:", t)
        print("\n\n\n")



        if os.path.exists(os.path.join(model.save_path, 'vtk')):
            pass
        else:
            os.makedirs(os.path.join(model.save_path, 'vtk'))

        origin_dicom = sitk.ReadImage(os.path.join(root, file))
        seg_result = sitk.GetImageFromArray(out_vtk_liver)

        seg_result.SetDirection(origin_dicom.GetDirection())
        seg_result.SetOrigin(origin_dicom.GetOrigin())
        seg_result.SetSpacing(origin_dicom.GetSpacing())

        sitk.WriteImage(seg_result, os.path.join(os.path.join(model.save_path, 'vtk'), file))

        avg_liver.append(total_iou)

        del out_vtk_liver,seg_result, origin_dicom

    return avg_liver
