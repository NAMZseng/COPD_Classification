import os
import pandas as pd
import numpy as np


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


def fix_index():
    """
    fix lung appear and disappear index, make them satisfy (disappear_index - appear_index) % 50 == 0
    """
    input_file = '/data/zengnanrong/label_match_ct_4_range_test.xlsx'
    output_file = '/data/zengnanrong/label_match_ct_4_range_test_fixed.xlsx'

    input_df = pd.read_excel(input_file, sheet_name='Sheet1')
    for i in range(len(input_df['subject'])):
        remainder = (input_df['disappear_index'][i] - input_df['appear_index'][i]) % 50
        if remainder != 0:
            add_num = remainder / 2
            input_df['appear_index'][i] += add_num
            input_df['disappear_index'][i] -= remainder - add_num

    input_df.to_excel(output_file, sheet_name='Sheet1', index=False, header=True)
