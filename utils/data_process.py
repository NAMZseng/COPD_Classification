import pandas as pd
import numpy as np
import os
import nibabel as nib
from scipy import ndimage
from pathos.multiprocessing import ProcessingPool as Pool


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 128
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_width, current_height, current_depth = img.shape
    # Compute depth factor
    depth_factor = desired_depth / current_depth
    width_factor = desired_width / current_width
    height_factor = desired_height / current_height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path_dic):
    """Read and resize volume"""
    path = path_dic['ctpath']
    dir = path_dic['dir']
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)

    ctarray = np.array([volume])

    np.save("/data/zengnanrong/lung_seg_normal_resize/" + dir + "_hw128_d128.npy", ctarray)
    print('save: ' + dir + "_hw128_d128.npy")


if __name__ == "__main__":
    # traindata = pd.read_excel(io=r'/data/zengnanrong/label_match_ct_4_range_train_valid.xlsx')
    # traindata = np.array(traindata)
    # train_paths = []
    # for i in range(len(traindata)):
    #     path = "/data/LUNG_SEG/train_valid/" + traindata[i, 1] + '/'
    #     dirs = os.listdir(path)
    #     ctpath = path + dirs[0] + '/' + dirs[0] + '.nii'
    #     train_paths.append({'ctpath': ctpath, 'dir': traindata[i, 1]})

    valdata = pd.read_excel(io=r'/data/zengnanrong/label_match_ct_4_range_test.xlsx')
    valdata = np.array(valdata)
    val_paths = []
    for i in range(len(valdata)):
        path = "/data/LUNG_SEG/test/" + valdata[i, 1] + '/'
        dirs = os.listdir(path)
        ctpath = path + dirs[0] + '/' + dirs[0] + '.nii'
        val_paths.append({'ctpath': ctpath, 'dir': valdata[i, 1]})

    pool = Pool(6)
    pool.map(process_scan, val_paths)
    pool.close()
    pool.join()
