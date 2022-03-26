import os

import nibabel as nib
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from scipy import ndimage

from dataset import load_3d_datapath_label


def read_nifti_file(path):
    """
    Read and load volume
    """
    nii_path = os.path.join(path, os.path.split(path)[1] + '.nii')
    volume = nib.load(nii_path).get_fdata()
    volume = volume.swapaxes(0, 2)  # from WHD to DHW

    return volume


def itensity_normalize(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    """
    pixels = volume[volume < 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out[volume == 0] = out_random[volume == 0]  # make value 0 pixels follow Gaussian distribution.
    return out


def crop_volume(volume, lung_appear_index, lung_disappear_index):
    """
    crop each image to 270x400,and remove non-lung slices
    """
    return volume[lung_appear_index:lung_disappear_index, 100:380, 50:450]


def resize_volume(volume, new_depth, new_height, new_width):
    """
    Resize the volume to the new size
    """
    [depth, height, width] = volume.shape
    scale = [new_depth * 1.0 / depth, new_height * 1.0 / height, new_width * 1.0 / width]
    out = ndimage.interpolation.zoom(volume, scale, order=0)

    return out


def process_volume(path_dic):
    """
    crop,resize and normalize volume
    """
    path = path_dic['image_path']
    dir = path_dic['dir'][:10]
    lung_appear_index = path_dic['appear_index']
    lung_disappear_index = path_dic['disappear_index']

    volume = read_nifti_file(path)
    volume = crop_volume(volume, lung_appear_index, lung_disappear_index)
    _, height, width = volume.shape
    volume = resize_volume(volume, 100, height, width)
    volume = itensity_normalize(volume)

    ct_array = np.array([volume])

    np.save("/data/zengnanrong/lung_seg_normal_resize/" + dir + "_h280_w400_d100.npy", ct_array)
    print('save: ' + dir + "_h280_w400_d100.npy")


if __name__ == "__main__":
    train_valid_label_path = '/data/zengnanrong/label_match_ct_4_range_del1524V2_train_valid.xlsx'
    train_valid_data_root_path = '/data/LUNG_SEG/train_valid/'
    train_valid_datapath_label = load_3d_datapath_label(train_valid_data_root_path, train_valid_label_path)

    test_label_path = '/data/zengnanrong/label_match_ct_4_range_test.xlsx'
    test_data_root_path = os.path.join('/data/LUNG_SEG/test/')
    test_datapath_label = load_3d_datapath_label(test_data_root_path, test_label_path)

    data_dic = []
    for i in range(4):
        data_dic.extend(train_valid_datapath_label[i])
        data_dic.extend(test_datapath_label[i])

    # process_volume(train_valid_datapath_label[0][0])

    pool = Pool(8)
    pool.map(process_volume, data_dic)
    pool.close()
    pool.join()
