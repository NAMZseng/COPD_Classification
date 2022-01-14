import os
import sys

import pandas as pd
import random
import torch
from torch import nn
from torch.autograd import Variable
from dataset import load_2d_datapath_label, load_data, load_3d_datapath_label
from datetime import datetime
from models.densenet import densenet121
from models.densenet_3d import generate_model

from torch.utils.tensorboard import SummaryWriter
from global_settings import CHECKPOINT_PATH, LOG_DIR, TIME_NOW, RESULT_DIR

import argparse


def next_batch(batch_size, index_in_total, data, cut_pic_size, cut_pic_num, phase):
    start = index_in_total
    index_in_total += batch_size
    total_num = len(data)

    # 最后一个batch
    if total_num < index_in_total < total_num + batch_size:
        index_in_total = total_num

    end = index_in_total

    batch_images = []
    batch_labels = []
    batch_dirs = []

    for i in range(start, end):
        if i < total_num:
            image = load_data(data[i], cut_pic_size, cut_pic_num)
            batch_images.append(image)

            label = data[i]['label']
            batch_labels.append(label)

            if phase == 'test':
                batch_dirs.append(data[i]['dir'])

    return batch_images, batch_labels, batch_dirs, index_in_total


def train(net, net_dim, use_gpu, train_data, valid_data, cut_pic_size, cut_pic_num, batch_size, num_epochs, optimizer,
          criterion, save_model_name):
    prev_time = datetime.now()

    # use tensorboard
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, net_dim + '_densenet', TIME_NOW))

    phase = 'train_valid'
    max_vail_acc = 0.0

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    for epoch in range(num_epochs):
        random.shuffle(train_data)
        train_loss = 0.0
        train_acc = 0
        index_in_trainset = 0

        net = net.train()

        if len(train_data) % batch_size == 0:
            batch_num = int(len(train_data) / batch_size)
        else:
            batch_num = int(len(train_data) / batch_size) + 1

        for batch in range(batch_num):
            batch_images, batch_labels, _, index_in_trainset = next_batch(batch_size, index_in_trainset, train_data,
                                                                          cut_pic_size, cut_pic_num, phase)
            batch_images = torch.tensor(batch_images, dtype=torch.float)

            if use_gpu:
                batch_images = Variable(torch.tensor(batch_images).cuda())
                batch_labels = Variable(torch.tensor(batch_labels).cuda())
            else:
                batch_images = Variable(torch.tensor(batch_images))
                batch_labels = Variable(torch.tensor(batch_labels))

            optimizer.zero_grad()  # 清除上一个batch计算的梯度,因为pytorch默认会累积梯度
            output = net(batch_images)
            loss = criterion(output, batch_labels)  # 计算损失
            loss = loss.requires_grad_()
            loss.backward()  # 计算梯度
            optimizer.step()  # 梯度更新

            train_loss += loss.data
            _, pred_label = output.max(1)
            num_correct = pred_label.eq(batch_labels).sum()
            train_acc += num_correct

        # 评估
        net = net.eval()
        with torch.no_grad():
            if len(valid_data) % batch_size == 0:
                batch_num = int(len(valid_data) / batch_size)
            else:
                batch_num = int(len(valid_data) / batch_size) + 1

            max_valid_acc = 0.0
            min_valid_loss = 1000.0

            if net_dim == '3D':
                valid_epoch = 5
            else:
                valid_epoch = 1
            for i in range(valid_epoch):
                valid_loss = 0
                valid_acc = 0
                index_in_validset = 0
                for batch in range(batch_num):
                    batch_images, batch_labels, _, index_in_validset = next_batch(batch_size, index_in_validset,
                                                                                  valid_data,
                                                                                  cut_pic_size, cut_pic_num, phase)
                    batch_images = torch.tensor(batch_images, dtype=torch.float)

                    if use_gpu:
                        batch_images = Variable(torch.tensor(batch_images).cuda())
                        batch_labels = Variable(torch.tensor(batch_labels).cuda())
                    else:
                        batch_images = Variable(torch.tensor(batch_images))
                        batch_labels = Variable(torch.tensor(batch_labels))

                    output = net(batch_images)
                    loss = criterion(output, batch_labels)
                    valid_loss += loss.data
                    _, pred_label = output.max(1)
                    num_correct = pred_label.eq(batch_labels).sum()
                    valid_acc += num_correct

                if valid_acc / len(valid_data) > max_valid_acc:
                    max_valid_acc = valid_acc / len(valid_data)
                if valid_loss / len(valid_data) < min_valid_loss:
                    min_valid_loss = valid_loss / len(valid_data)

        epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch + 1,
                   train_loss / len(train_data), train_acc / len(train_data),
                   min_valid_loss, max_valid_acc))

        writer.add_scalars('Loss', {'Train': train_loss / len(train_data), 'Valid': min_valid_loss}, epoch + 1)
        writer.add_scalars('Accuracy', {'Train': train_acc / len(train_data), 'Valid': max_valid_acc}, epoch + 1)

        if valid_acc / len(valid_data) > max_vail_acc:
            max_vail_acc = valid_acc / len(valid_data)
            torch.save(net, os.path.join(CHECKPOINT_PATH, net_dim + '_densenet', save_model_name))

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time
        print(epoch_str + time_str)

    writer.close()


def test(net_dim, use_gpu, test_data, cut_pic_size, cut_pic_num, batch_size, save_model_name, result_file):
    phase = 'test'

    net = torch.load(os.path.join(CHECKPOINT_PATH, save_model_name))
    net = net.eval()

    with torch.no_grad():
        if len(test_data) % batch_size == 0:
            batch_num = int(len(test_data) / batch_size)
        else:
            batch_num = int(len(test_data) / batch_size) + 1

        max_test_acc = 0.0

        if net_dim == '3D':
            test_epoch = 5
        else:
            test_epoch = 1
        for i in range(test_epoch):
            test_acc = 0
            index_in_testset = 0
            label_list = []
            probability_predicted_list = []
            label_predicted_list = []
            dirs_list = []
            for batch in range(batch_num):
                batch_images, batch_labels, batch_dirs, index_in_testset = next_batch(batch_size, index_in_testset,
                                                                                      test_data, cut_pic_size,
                                                                                      cut_pic_num, phase)
                batch_images = torch.tensor(batch_images, dtype=torch.float)

                if use_gpu:
                    batch_images = Variable(torch.tensor(batch_images).cuda())
                    batch_labels = Variable(torch.tensor(batch_labels).cuda())
                else:
                    batch_images = Variable(torch.tensor(batch_images))
                    batch_labels = Variable(torch.tensor(batch_labels))

                output = net(batch_images)
                softmax = nn.Softmax(dim=1)
                output_softmax = softmax(output)

                _, pred_label = output.max(1)
                num_correct = pred_label.eq(batch_labels).sum()
                test_acc += num_correct

                label_list.extend(batch_labels.cpu().numpy().tolist())
                probability_predicted_list.extend(output_softmax.cpu().numpy().tolist())
                label_predicted_list.extend(pred_label.cpu().numpy().tolist())
                dirs_list.extend(batch_dirs)

            print("Test Acc: %f" % (test_acc / len(test_data)))

            if (test_acc / len(test_data)) > max_test_acc:
                max_test_acc = test_acc / len(test_data)
                df = pd.DataFrame(probability_predicted_list, columns=['p0', 'p1', 'p2', 'p3'])
                df.insert(df.shape[1], 'label-pre', label_predicted_list)
                df.insert(df.shape[1], 'label_gt', label_list)
                df.insert(df.shape[1], 'dirs', dirs_list)
                df.to_excel(os.path.join(RESULT_DIR, net_dim + '_densenet', result_file))


def count_person_result(input_file, output_file):
    """
    将每个病例的所有测试图像的四个等级的预测概率求平均
    :param input_file:
    :param output_file:
    :return:
    """
    input_df = pd.read_excel(input_file, sheet_name='Sheet1')
    input_df.sort_values(by='dirs', inplace=True)

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
            temp_row[0] = temp_row[0] / count
            temp_row[1] = temp_row[1] / count
            temp_row[2] = temp_row[2] / count
            temp_row[3] = temp_row[3] / count
            temp_row[4] = temp_row[:4].index(max(temp_row[:4]))
            temp_row[5] = input_df['label_gt'][i]
            temp_row[6] = input_df['dirs'][i]
            output_list.append(temp_row)

            count = 0
            temp_row = [0.0, 0.0, 0.0, 0.0, 0, 0, 'test']

        if i + 1 == len(input_df['dirs']):
            # last line
            temp_row[0] = temp_row[0] / count
            temp_row[1] = temp_row[1] / count
            temp_row[2] = temp_row[2] / count
            temp_row[3] = temp_row[3] / count
            temp_row[4] = temp_row[:4].index(max(temp_row[:4]))
            temp_row[5] = input_df['label_gt'][i]
            temp_row[6] = input_df['dirs'][i]
            output_list.append(temp_row)

    df = pd.DataFrame(output_list, columns=['p0', 'p1', 'p2', 'p3', 'label-pre', 'label_gt', 'dirs'])
    df.to_excel(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_dim', type=str, default='3D', choices=['2D', '3D'], help='使用2D还是3D DenseNet')
    parser.add_argument('--data_root_path', type=str, default='/data/zengnanrong/CTDATA/', help='输入数据的根路径')
    parser.add_argument('--cut_pic_size', type=bool, default=False, help='是否将图片裁剪为432*432')
    parser.add_argument('--cut_pic_num', type=str, choices=['remain', 'precise', 'rough'], default='remain',
                        help='是否只截去不含肺区域的图像，remain:不截，保留原始图像的个数，precise:精筛，rough:粗筛，直接截去上下各1/6的图像数量')
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否只使用GPU')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size, 2d:20, 3d:2')
    parser.add_argument('--num_epochs', type=int, default=50, help='num of epochs')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate,2D:0.5，3D：0.2')
    parser.add_argument('--save_model_name', type=str, default='DenseNet121_debug.pkl',
                        help='model save name')
    parser.add_argument('--result_file', type=str, default='test_debug_50epoch_dir.xlsx',
                        help='test result filename')
    parser.add_argument('--cuda_device', type=str, choices=['0', '1'], default='1', help='使用哪块GPU')

    args_in = sys.argv[1:]
    args = parser.parse_args(args_in)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    torch.cuda.empty_cache()

    channels = 1
    num_classes = 4  # 4分类

    train_valid_label_path = '/data/zengnanrong/label_match_ct_4_range_train_valid.xlsx'
    test_label_path = '/data/zengnanrong/label_match_ct_4_range_test.xlsx'

    train_valid_data_root_path = os.path.join(args.data_root_path, 'train_valid')
    test_data_root_path = os.path.join(args.data_root_path, 'test')

    if args.net_dim == '2D':
        train_valid_datapath_label = load_2d_datapath_label(train_valid_data_root_path, train_valid_label_path,
                                                            args.cut_pic_num)
        test_datapath_label = load_2d_datapath_label(test_data_root_path, test_label_path, args.cut_pic_num)
        # 2D DenseNet
        net = densenet121(channels, num_classes, args.use_gpu, args.drop_rate)
    elif args.net_dim == '3D':
        train_valid_datapath_label = load_3d_datapath_label(train_valid_data_root_path, train_valid_label_path)
        test_datapath_label = load_3d_datapath_label(test_data_root_path, test_label_path)
        # 3D DenseNet
        net = generate_model(121, args.use_gpu, n_input_channels=channels, num_classes=num_classes,
                             drop_rate=args.drop_rate)

    train_data = []
    valid_data = []
    test_data = []

    for label in range(4):
        # 每个标签的数据按 训练集：验证集：测试集 6:1:3
        train_index = int(len(train_valid_datapath_label[label]) * 6 / 7)

        while train_valid_datapath_label[label][train_index]['dir'] == \
                train_valid_datapath_label[label][train_index + 1]['dir']:
            train_index = train_index + 1

        train_index = train_index + 1

        train_data.extend(train_valid_datapath_label[label][:train_index])
        valid_data.extend(train_valid_datapath_label[label][train_index:])
        test_data.extend(test_datapath_label[label])

    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    train(net, args.net_dim, args.use_gpu, train_data, valid_data, args.cut_pic_size, args.cut_pic_num, args.batch_size,
          args.num_epochs, optimizer, criterion, args.save_model_name)
    test(args.net_dim, args.use_gpu, test_data, args.cut_pic_size, args.cut_pic_num, args.batch_size,
         args.save_model_name,
         args.result_file)
