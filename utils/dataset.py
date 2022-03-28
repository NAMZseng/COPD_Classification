import os

import numpy as np
import pandas as pd


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
    加载npy数据的路径，并为其加上对应标签
    :param data_root_path:
    :param label_path:
    :return:
    """
    label_df = pd.read_excel(label_path, sheet_name='Sheet1')

    scale_num = 1  # 4个尺度
    data_path_with_label = [[[] for j in range(4)] for i in range(scale_num)]  # 创建(4,4,0)的三维数组

    # multi_scale = ['_h280_w400_d100', '_h156_w224_d300', '_h140_w200_d400']
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


train_valid_lrf_path = '/data/zengnanrong/label_original_match_ct_4_train_valid.xlsx'
test_lrf_path = '/data/zengnanrong/label_original_match_ct_4_test.xlsx'
train_valid_lrf_df = pd.read_excel(train_valid_lrf_path, sheet_name='Sheet1')
train_valid_lrf = np.array(train_valid_lrf_df)
test_lrf_df = pd.read_excel(test_lrf_path, sheet_name='Sheet1')
test_lrf = np.array(test_lrf_df)


def load_data(data_dic, phase):
    path = data_dic['image_path']
    if path[-4:] == '.npy':
        image_array = np.load(path)  # (1,D,H,W)

        # 获取对应的组学特征
        index = data_dic['index']
        if phase == 'train_valid':
            lrf = train_valid_lrf[index][4:]
        elif phase == 'test':
            lrf = test_lrf[index][4:]
        return image_array, lrf


def count_person_result(input_file, output_file):
    """
    将每个病例的所有测试图像的四个等级的预测概率求平均
    :param input_file:
    :param output_file:
    :return:
    """
    input_df = pd.read_excel(input_file, sheet_name='Sheet1')
    # TODO 解决排序后结果没有写回的问题
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
    data_root_path = "/data/LUNG_SEG/train_valid/"
    # data_root_path = "/data/zengnanrong/lung_seg_normal_resize"
    label_path = '/data/zengnanrong/label_match_ct_4_range_del1524V2_train_valid.xlsx'
    data = load_3d_datapath_label(data_root_path, label_path)
    # data = load_3d_npy_datapath_label(data_root_path, label_path)
    print(data[0][0])
