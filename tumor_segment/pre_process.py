"""

获取固定取样方式下的训练数据
首先将灰度值超过upper和低于lower的灰度进行截断
然后调整slice thickness，然后将slice的分辨率调整为256*256
只有包含肝脏以及肝脏上下 expand_slice 张slice作为训练样本
最后将输入数据分块，以轴向 stride 张slice为步长进行取样

网络输入为256*256*size
当前脚本依然对金标准进行了缩小，如果要改变，直接修改第70行就行
"""

import os
import shutil
from time import time
import re
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage

upper = 200
lower = -200

stride = 3  # 取样的步长
down_scale = 1
slice_thickness = 1


# root = '/mnt/data/dataset/liver/'


def read_dicom(path):
    print(path)
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        dicoms = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicoms)
        image = reader.Execute()
        return image
    else:
        image = sitk.ReadImage(path)
        return image


# 用来记录产生的数据的序号


def process_tumor():
    root = '/workspace/mnt/group/alg-pro/yankai/segment/data/pre_process'

    new_ct_dir = '/workspace/mnt/group/alg-pro/yankai/segment/data/pre_process_tumor'
    new_seg_dir = '/workspace/mnt/group/alg-pro/yankai/segment/data/pre_process_tumor'

    file_list = [file for file in os.listdir(root) if
                 'volume' in file and (int(re.sub("\D", "", file)) <= 130 or int(re.sub("\D", "", file)) > 150)]
    print(file_list)
    for ct_file in file_list:

        ct_dir = os.path.join(root, ct_file)
        seg_dir = ct_dir.replace('volume', 'segmentation')

        seg = read_dicom(seg_dir)
        seg_array = sitk.GetArrayFromImage(seg)

        ct = read_dicom(ct_dir)
        ct_array = sitk.GetArrayFromImage(ct)
        ct_array = (seg_array>0)*ct_array

        ct_array[ct_array==0] = -200

        # avg = np.mean(ct_array)
        # std = np.std(ct_array)
        # ct_array = (ct_array - avg) / std

        ct_array = ct_array.astype(np.int16)



        seg_array = seg_array>1

        seg_array = seg_array.astype(np.int16)



        print(np.sum(seg_array))



        new_ct = sitk.GetImageFromArray(ct_array)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing(ct.GetSpacing())

        new_seg = sitk.GetImageFromArray(seg_array)
        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing(ct.GetSpacing())

        name_num = int(re.sub("\D", "", ct_file))

        new_ct_name = 'volume-' + str(name_num if name_num<=130 else name_num-20) + '.nii'
        new_seg_name = new_ct_name.replace('volume','segmentation')

        print("write ", new_ct_name)
        print("write ", new_seg_name)
        sitk.WriteImage(new_ct, os.path.join(new_ct_dir, new_ct_name))
        sitk.WriteImage(new_seg, os.path.join(new_seg_dir, new_seg_name))

        print('{} have {} slice left'.format(ct_file, seg_array.shape[0]))


process_tumor()

