import SimpleITK as sitk
import os
import numpy as np


def read_data(path, file, rotate, normalize='gaosi'):  # 获得整套CT数据和标签，并且转换成numpy格式
    image = sitk.ReadImage(os.path.join(path, file))
    print(file, image.GetSpacing())
    dicom_array = sitk.GetArrayFromImage(image)
    if normalize == 'gaosi':
        avg = np.mean(dicom_array)
        std = np.std(dicom_array)
        dicom_array = (dicom_array - avg) / std
    else:
        max = np.max(dicom_array)
        min = np.min(dicom_array)
        dicom_array = (dicom_array - min) / (max - min)

    label_array = np.zeros([dicom_array.shape[0], dicom_array.shape[1], dicom_array.shape[2], 2])

    image = sitk.ReadImage(os.path.join(path, file.replace("volume", "segmentation")))
    liver_array = sitk.GetArrayFromImage(image)

    if rotate == 1 or rotate == 4:  # 根据rotate对ct进行图像旋转
        dicom_array = dicom_array
        liver_array = liver_array

    elif rotate == 2:
        dicom_array = np.rot90(dicom_array, axes=(1, 2))
        liver_array = np.rot90(liver_array, axes=(1, 2))

    elif rotate == 3:
        dicom_array = np.rot90(np.rot90(dicom_array, axes=(1, 2)), axes=(1, 2))
        liver_array = np.rot90(np.rot90(liver_array, axes=(1, 2)), axes=(1, 2))

    else:
        dicom_array = np.rot90(np.rot90(np.rot90(dicom_array, axes=(1, 2)), axes=(1, 2)), axes=(1, 2))
        liver_array = np.rot90(np.rot90(np.rot90(liver_array, axes=(1, 2)), axes=(1, 2)), axes=(1, 2))

    label_array[..., 0][liver_array > 0] = 1

    temp = liver_array
    temp = temp > 0
    label_array[:, :, :, 1] = ~temp

    return dicom_array, label_array


def chop_datas(dicom_array, w_size, h_size, c_size, train):  # 将CT数据切割成设置固定大小的长方体
    indexs = []
    shape = dicom_array.shape
    plies = shape[0]
    w = shape[1]
    h = shape[2]
    if (train):
        w_step_size = w_size // 2
        h_step_size = h_size // 2
        c_step_size = c_size // 3
        c_nums = (plies + c_step_size - 1) // c_step_size - 1
        w_nums = (w + w_step_size - 1) // w_step_size - 1
        h_nums = (h + h_step_size - 1) // h_step_size - 1
    else:
        w_step_size = w_size
        h_step_size = h_size
        c_step_size = c_size
        c_nums = (plies + c_step_size - 1) // c_step_size
        w_nums = (w + w_step_size - 1) // w_step_size
        h_nums = (h + h_step_size - 1) // h_step_size
    count = 0
    for i in range(c_nums):
        c_start = i * c_step_size
        c_end = c_start + c_size

        if c_end > plies:
            c_end = plies
            c_start = c_end - c_size
        else:
            pass
        for j in range(w_nums):
            w_start = j * w_step_size
            w_end = w_start + w_size

            if w_end > w:
                w_end = w
                w_start = w_end - w_size
            else:
                pass
            for k in range(h_nums):
                h_start = k * h_step_size
                h_end = h_start + h_size

                if h_end > h:
                    h_end = h
                    h_start = h_end - h_size
                else:
                    pass
                count += 1
                indexs.append([c_start, c_end, w_start, w_end, h_start, h_end])
    return indexs
