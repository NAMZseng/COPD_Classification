import os

import pandas as pd
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from PIL import Image


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
    label_df = pd.read_excel(label_path, sheet_name='Sheet1')
    label_df.insert(label_df.shape[1], 'appear_index', 0)
    label_df.insert(label_df.shape[1], 'disappear_index', 0)

    for i in range(label_df['subject']):
        path = os.path.join(data_root_path, label_df['subject'][i])
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


def load_1316_datapath_label(data_root_path):
    data_path_with_label = [[], [], [], []]
    for root, dirs, files in os.walk(data_root_path):
        if len(dirs) == 0:  # 此时已经进入具体某个标签的文件夹下
            for image in files:
                image_path = os.path.join(root, image)
                index = int(os.path.split(root)[1])
                data_path_with_label[index].append(
                    {'image_path': image_path, 'label': index, 'dir': str(index) + '/' + image})

    return data_path_with_label


def load_2d_datapath_label(data_root_path, label_path, cut_pic_num):
    """
    2d网络的数据加载
    加载每一张DICOM图像的路径，并为其加上对应标签
    :param data_root_path:
    :param label_path:
    :param cut: 是否截取包含肺区域的图像
    :return:
    """
    label_df = pd.read_excel(label_path, sheet_name='Sheet1')

    data_path_with_label = [[], [], [], []]

    for i in range(len(label_df['subject'])):
        path = os.path.join(data_root_path, label_df['subject'][i])
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
    3d网络的数据加载
    加载每个病人的图像路径，并为其加上对应标签
    :param data_root_path:
    :param label_path:
    :return:
    """
    label_df = pd.read_excel(label_path, sheet_name='Sheet1')

    data_path_with_label = [[], [], [], []]

    for i in range(len(label_df['subject'])):
        path = os.path.join(data_root_path, label_df['subject'][i])

        for root, dirs, files in os.walk(path):
            if len(dirs) == 1:
                image_path = os.path.join(root, dirs[0])
                # 训练时预测的标签范围为[0,3]
                label = label_df['GOLDCLA'][i] - 1
                data_path_with_label[label].append({'image_path': image_path, 'label': label, 'dir': dirs[0],
                                                    'appear_index': label_df['appear_index'][i],
                                                    'disappear_index': label_df['disappear_index'][i]})

    return data_path_with_label


def load_3d_npy_datapath_label(data_root_path, label_path):
    """
    加载陈梓然处理的3d npy数据路径，并为其加上对应标签
    :param data_root_path:
    :param label_path:
    :return:
    """
    label_df = pd.read_excel(label_path, sheet_name='Sheet1')

    # data_path_with_label = [[], [], [], []]
    scale_num = 4  # 4个尺度
    data_path_with_label = [[[] for j in range(4)] for i in range(scale_num)]  # 创建(4,4,0)的三维数组

    multi_scale = ['_h280_w400_d100']

    for i in range(len(label_df)):
        # 训练时预测的标签范围为[0,3]
        label = label_df['GOLDCLA'][i] - 1
        for scale in range(len(multi_scale)):
            scale_image_path = os.path.join(data_root_path, label_df['subject'][i] + multi_scale[scale] + '.npy')
            if os.path.exists(scale_image_path):
                data_path_with_label[scale][label].append(
                    {'image_path': scale_image_path, 'label': label, 'dir': label_df['subject'][i], 'index': i})

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

    # image_array_cut = []
    #
    # cut_slice_num = 100

    # 抽法一：将每个人的CT图像分成cut_slice_num份，每份中等距抽取一张
    # step = int(len(image_array) / cut_slice_num)
    # index = random.sample(range(0, step), 1)
    # index = index[0]
    # image_array_cut.append(image_array[index])
    # for i in range(1, cut_slice_num):
    #     index = index + step
    #     image_array_cut.append(image_array[index])

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

    # return image_array_cut

    #  for swin-transformer
    return image_array


# train_valid_lrf_path = '/data/zengnanrong/label_original_match_ct_4_train_valid.xlsx'
# test_lrf_path = '/data/zengnanrong/label_original_match_ct_4_test.xlsx'
# train_valid_lrf_df = pd.read_excel(train_valid_lrf_path, sheet_name='Sheet1')
# train_valid_lrf = np.array(train_valid_lrf_df)
# test_lrf_df = pd.read_excel(test_lrf_path, sheet_name='Sheet1')
# test_lrf = np.array(test_lrf_df)


def load_data(data_dic, cut_pic_size, cut_pic_num, phase):
    path = data_dic['image_path']
    if path[-4:] == '.png':  # for 1316 dataset
        image = Image.open(path)  # (512, 512)
        image_array = [np.array(image)]  # (1, 512, 512)
        image_array = np.array(image_array)
        z, x, y = image_array.shape
        if cut_pic_size:
            image_array = zoom(image_array, (1, 224 / x, 224 / y))
    elif path[-4:] == '.npy':
        image_array = np.load(path)  # (1,D,H,W)

        # 获取对应的组学特征
        index = data_dic['index']
        if phase == 'train_valid':
            lrf = train_valid_lrf[index][4:]
        elif phase == 'test':
            lrf = test_lrf[index][4:]
        return image_array, lrf


    elif os.path.isfile(path):  # for single dicom slice
        dicom_image = sitk.ReadImage(path)
        image_array = sitk.GetArrayFromImage(dicom_image)  # (1, 512, 512)
        z, x, y = image_array.shape
        if cut_pic_size:
            image_array = zoom(image_array, (1, 224 / x, 224 / y))
    else:  # for dicom series or nii
        image_array = load_dicom_series(data_dic, cut_pic_num)  # (N, 512, 512)
        z, x, y = image_array.shape
        if cut_pic_size:
            image_array = zoom(image_array, (100 / z, 224 / x, 224 / y))
        # image_array = zoom(image_array, (100 / z, 1, 1))
        image_array = image_array.tolist()

        # make the shape of image_array from (depth,high,width) to (channel,depth,high,width), here channel = 1
        image_array = [image_array]

        # for efficientV2 and swin-transformer, make image_array_3d shape:(depth,channel,high,width)
        # for i in range(len(image_array)):
        #     image_array[i] = [image_array[i]]

        image_array = np.array(image_array)

    return image_array


def count_person_result(input_file, output_file):
    """
    将每个病例的所有测试图像的四个等级的预测概率求平均
    :param input_file:
    :param output_file:
    :return:
    """
    input_df = pd.read_excel(input_file, sheet_name='Sheet1')
    input_df = input_df.sort_values(by='dirs')

    output_list = []
    count = 0
    temp_row = [0.0, 0.0, 0.0, 0.0, 0, 0, 'test']
    for i in range(len(input_df['dirs'])):
        temp_row[0] = temp_row[0] + input_df['p0'][i]
        temp_row[1] = temp_row[1] + input_df['p1'][i]
        temp_row[2] = temp_row[2] + input_df['p2'][i]
        temp_row[3] = temp_row[3] + input_df['p3'][i]
        count = count + 1

        if i + 1 < len(input_df['dirs']) and input_df['dirs'][i] is not input_df['dirs'][i + 1]:
            for j in range(4):
                temp_row[j] = temp_row[j] / count
            temp_row[4] = temp_row[:4].index(max(temp_row[:4]))
            temp_row[5] = input_df['label_gt'][i]
            temp_row[6] = input_df['dirs'][i]
            output_list.append(temp_row)

            count = 0
            temp_row = [0.0, 0.0, 0.0, 0.0, 0, 0, 'test']

        if i + 1 == len(input_df['dirs']):
            # last line
            for j in range(4):
                temp_row[j] = temp_row[j] / count
            temp_row[4] = temp_row[:4].index(max(temp_row[:4]))
            temp_row[5] = input_df['label_gt'][i]
            temp_row[6] = input_df['dirs'][i]
            output_list.append(temp_row)

    df = pd.DataFrame(output_list, columns=['p0', 'p1', 'p2', 'p3', 'label-pre', 'label_gt', 'dirs'])
    df.to_excel(output_file)


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
    data_root_path = "/data/LUNG_SEG/train_valid/"
    # data_root_path = "/data/zengnanrong/dataset1316/test"
    # data_root_path = "/data/zengnanrong/lung_seg_normal_resize"
    label_path = '/data/zengnanrong/label_match_ct_4_range_del1524V2_train_valid.xlsx'
    # data = load_2d_datapath_label(data_root_path, label_path, False, False)
    data = load_3d_datapath_label(data_root_path, label_path)
    # data = load_3d_npy_datapath_label(data_root_path, label_path)
    # data = load_1316_datapath_label(data_root_path)
    print(data[0][0])