import os
import random

import pandas as pd
import SimpleITK as sitk
import numpy as np
import nibabel as nib


def label_preprocess(label_path, output_path):
    """
    将label文件表中的文件名与CTDATA文件夹中的名称对应
    表里的V2--->CT图像的V1
    表里的V3--->CT图像的V2
    表里的V4--->CT图像的V3
    :param label_path:
    :param output_path:
    :return:
    """

    label_data = pd.read_excel(label_path, sheet_name='Sheet1')

    for i in range(len(label_data['subject'])):
        version_num = int(label_data['subject'][i][9]) - 1
        label_data['subject'][i] = label_data['subject'][i][:9] + str(version_num)

        if label_data['GOLDCLA'][i] == 5:
            # 将级别5的改为级别4，使得级别4的样本数与级别1-3的基本相同
            label_data['GOLDCLA'][i] = 4

    label_data.sort_values(by='subject', inplace=True)
    pd.DataFrame(label_data).to_excel(output_path, sheet_name='Sheet1')


def exist_lung(image_path):
    image = sitk.ReadImage(image_path)
    image_array = np.squeeze(sitk.GetArrayFromImage(image))
    for x in range(100, len(image_array[0])):
        for y in range(len(image_array[1])):
            if image_array[x][y] > 0:
                return True
    return False


def find_lung_range(label_path, data_root_path, output_path):
    """
    读取data_root_path中的CT图像，确定每例病人的含肺图像的范围，并将最先出现的和最后消失的含肺图像索引存入在output_path的excel文件中
    :param label_path:
    :param data_root_path:
    :return:
    """
    ct_dir = []
    for item in os.listdir(data_root_path):
        if os.path.isdir(os.path.join(data_root_path, item)):
            ct_dir.append(item)

    # 确保与label文件中的名称顺序对应
    ct_dir.sort()

    label_df = pd.read_excel(label_path, sheet_name='Sheet1')
    label_df.insert(label_df.shape[1], 'appear_index', 0)
    label_df.insert(label_df.shape[1], 'disappear_index', 0)

    for i in range(len(ct_dir)):
        data_dir_name = ct_dir[i]
        if data_dir_name == label_df['subject'][i]:
            path = os.path.join(data_root_path, data_dir_name)
            for root, dirs, files in os.walk(path):
                if len(files) == 0:
                    continue

                len_files = len(files)
                files.sort()
                for appear_index in range(len_files):
                    image_path = os.path.join(root, files[appear_index])
                    if exist_lung(image_path):
                        print(i)
                        label_df['appear_index'][i] = appear_index
                        break
                for index in range(len_files):
                    disappear_index = len_files - 1 - index
                    image_path = os.path.join(root, files[disappear_index])
                    if exist_lung(image_path):
                        print(i)
                        label_df['disappear_index'][i] = disappear_index
                        break

    label_df.to_excel(output_path)


def load_2d_datapath_label(data_root_path, label_path, cut_pic_num):
    """
    2d-densenet的数据加载
    加载每一张DICOM图像的路径，并为其加上对应标签
    :param data_root_path:
    :param label_path:
    :param cut: 是否截取包含肺区域的图像
    :return:
    """
    ct_dir = []
    for item in os.listdir(data_root_path):
        if os.path.isdir(os.path.join(data_root_path, item)):
            ct_dir.append(item)

    # 确保与label文件中的名称顺序对应
    ct_dir.sort()

    label_df = pd.read_excel(label_path, sheet_name='Sheet1')

    data_path_with_label = [[], [], [], []]

    for i in range(len(ct_dir)):
        data_dir_name = ct_dir[i]

        if data_dir_name == label_df['subject'][i]:
            path = os.path.join(data_root_path, data_dir_name)
            for root, dirs, files in os.walk(path):
                if len(files) == 0:
                    continue

                if cut_pic_num == 'remain':
                    pass
                elif cut_pic_num == 'precise':
                    files.sort()
                    appear_idx = label_df['appear_index'][i]
                    disappear_idx = label_df['disappear_index'][i]

                    if '.xml' in files[0].lower():
                        appear_idx = appear_idx + 1
                        disappear_idx = disappear_idx + 1

                    files = files[appear_idx:disappear_idx + 1]
                elif cut_pic_num == 'rough':
                    files.sort()
                    start_idx = int(len(files) / 6)
                    end_idx = len(files) - start_idx
                    files = files[start_idx:end_idx]

                for item in files:
                    if '.dcm' in item.lower():
                        image_path = os.path.join(root, item)
                        # 训练时预测的标签范围为[0,3]
                        label = label_df['GOLDCLA'][i] - 1
                        data_path_with_label[label].append(
                            {'image_path': image_path, 'label': label, 'dir': os.path.split(root)[1]})

    return data_path_with_label


def load_3d_datapath_label(data_root_path, label_path):
    """
    3d-densenet的数据加载
    加载每个病人的图像路径，并为其加上对应标签
    :param data_root_path:
    :param label_path:
    :return:
    """
    ct_dir = []
    for item in os.listdir(data_root_path):
        if os.path.isdir(os.path.join(data_root_path, item)):
            ct_dir.append(item)

    # 确保与label文件中的名称顺序对应
    ct_dir.sort()

    label_df = pd.read_excel(label_path, sheet_name='Sheet1')

    data_path_with_label = [[], [], [], []]

    for i in range(len(ct_dir)):
        data_dir_name = ct_dir[i]

        if data_dir_name == label_df['subject'][i]:
            path = os.path.join(data_root_path, data_dir_name)

            for root, dirs, files in os.walk(path):
                if len(dirs) == 1:
                    image_path = os.path.join(root, dirs[0])
                    # 训练时预测的标签范围为[0,3]
                    label = label_df['GOLDCLA'][i] - 1
                    data_path_with_label[label].append({'image_path': image_path, 'label': label, 'dir': dirs[0],
                                                        'appear_index': label_df['appear_index'][i],
                                                        'disappear_index': label_df['disappear_index'][i]})

    return data_path_with_label


def load_dicom_series(data_dic, cut_pic_num):
    path = data_dic['image_path']

    nii_path = os.path.join(path, os.path.split(path)[1] + '.nii')
    if os.path.exists(nii_path):
        image_array = nib.load(nii_path).get_data()
        # nib加载的3d数据格式为height*width*depth, 需要调转成depth*width*height
        image_array = image_array.swapaxes(0, 2)
    else:
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, series_IDs[0])
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        image3D = series_reader.Execute()
        image_array = sitk.GetArrayFromImage(image3D)

    if cut_pic_num == 'remain':
        pass
    elif cut_pic_num == 'precise':
        appear_idx = data_dic['appear_index']
        disappear_index = data_dic['disappear_index']
        image_array = image_array[appear_idx:disappear_index]
    elif cut_pic_num == 'rough':
        start_idx = int(len(image_array) / 6)
        end_idx = len(image_array) - start_idx
        image_array = image_array[start_idx:end_idx]

    image_array_cut = []

    cut_slice_num = 20

    # 抽法一：将每个人的CT图像分成cut_slice_num份，每份中等距抽取一张
    step = int(len(image_array) / cut_slice_num)
    index = random.sample(range(0, step), 1)
    index = index[0]
    image_array_cut.append(image_array[index])
    for i in range(1, cut_slice_num):
        index = index + step
        image_array_cut.append(image_array[index])

    # 抽法二：每例病人的数据分成10块，每次随机抽取一块，再从这块取cut_slice_num张图用做该轮训练
    # block_index = random.sample(range(1, 11), 1)
    # block_index = block_index[0]
    #
    # block_size = int(len(image_array) / 10)
    # slice_indexes = random.sample(range(block_size * (block_index - 1), block_size * block_index), cut_slice_num)
    # slice_indexes.sort()
    # for index in slice_indexes:
    #     image_array_cut.append(image_array[index])

    # 抽法三：多实例，将每个人的CT图像分成cut_slice_num份，每份中随机抽取一张
    # step = int(len(image_array) / cut_slice_num)
    # for i in range(0, cut_slice_num):
    #     index = random.sample(range(i * step, (i + 1) * step), 1)
    #     index = index[0]
    #     image_array_cut.append(image_array[index])

    return image_array_cut


def load_data(data_dic, cut_pic_size, cut_pic_num):
    path = data_dic['image_path']
    if os.path.isfile(path):
        dicom_image = sitk.ReadImage(path)
        image_array = sitk.GetArrayFromImage(dicom_image)
        if cut_pic_size:
            # 裁剪成1*432*432
            pass
    else:
        image_array_3d = load_dicom_series(data_dic, cut_pic_num)
        if cut_pic_size:
            pass
        image_array = []
        # make the shape of image_array from (depth,high,width) to (channel,depth,high,width)
        # here channel = 1
        image_array.append(image_array_3d)

    return image_array


if __name__ == "__main__":
    # 肺部CT原始图像
    # data_root_path = "/data/zengnanrong"
    # label_path = os.path.join(data_root_path, 'label.xlsx')
    # output_path = os.path.join(data_root_path, 'label_match_ct_4.xlsx')
    # label_preprocess(label_path, output_path)

    # 经过lungmask Unet-R231模型分割后的肺部区域标图像
    # data_root_path = "/data/zengnanrong/R231/"
    # label_path = '/data/zengnanrong/label_match_ct_4.xlsx'
    # output_path = '/data/zengnanrong/label_match_ct_4_range.xlsx'
    # find_lung_range(label_path, data_root_path, output_path)

    # 分割后的肺部CT图像
    # data_root_path = "/data/zengnanrong/R231/"
    # label_path = '/data/zengnanrong/label_match_ct_4_range.xlsx'
    # data_root_path = "/data/zengnanrong/CTDATA/test/"
    # label_path = '/data/zengnanrong/label_match_ct_4_range_test.xlsx'
    # data_root_path = "/data/zengnanrong/CTDATA/train_valid/"
    data_root_path = "/data/LUNG_SEG/train_valid/"
    label_path = '/data/zengnanrong/label_match_ct_4_range_train_valid.xlsx'
    # data = load_2d_datapath_label(data_root_path, label_path, False, False)
    data = load_3d_datapath_label(data_root_path, label_path)
    print(data[0][0])
    print(len(data[0]))
    print(len(data[1]))
    print(len(data[2]))
    print(len(data[3]))
    print(len(data[0]) + len(data[1]) + len(data[2]) + len(data[3]))
