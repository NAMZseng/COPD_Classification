import os
import sys

import pandas as pd
import random
import torch
from torch import nn
from torch.autograd import Variable
from dataset import load_datapath_label, load_data
from datetime import datetime
from models.densenet import densenet121

from torch.utils.tensorboard import SummaryWriter
from global_settings import CHECKPOINT_PATH, LOG_DIR, TIME_NOW

import argparse


def next_batch(batch_size, index_in_total, data):
    start = index_in_total
    index_in_total += batch_size
    total_num = len(data)

    # 最后一个batch
    if total_num < index_in_total < total_num + batch_size:
        index_in_total = total_num

    end = index_in_total

    batch_images = []
    batch_labels = []

    for i in range(start, end):
        if i < total_num:
            image_path = data[i]['image_path']
            image = load_data(image_path)
            batch_images.append(image)

            label = data[i]['label']
            batch_labels.append(label)

    return batch_images, batch_labels, index_in_total


def train(net, use_gpu, train_data, valid_data, batch_size, num_epochs, optimizer, criterion, save_model_name):
    prev_time = datetime.now()

    # use tensorboard
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, 'densenet121', TIME_NOW))

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
            batch_images, batch_labels, index_in_trainset = next_batch(batch_size, index_in_trainset, train_data)
            batch_images = torch.tensor(batch_images, dtype=torch.float)

            if use_gpu:
                batch_images = Variable(torch.tensor(batch_images).cuda())
                batch_labels = Variable(torch.tensor(batch_labels).cuda())
            else:
                batch_images = Variable(torch.tensor(batch_images))
                batch_labels = Variable(torch.tensor(batch_labels))

            optimizer.zero_grad()
            output = net(batch_images)
            loss = criterion(output, batch_labels)
            loss = loss.requires_grad_()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            _, pred_label = output.max(1)
            num_correct = pred_label.eq(batch_labels).sum()
            train_acc += num_correct

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        # 评估
        valid_loss = 0
        valid_acc = 0
        index_in_validset = 0
        net = net.eval()

        with torch.no_grad():
            if len(valid_data) % batch_size == 0:
                batch_num = int(len(valid_data) / batch_size)
            else:
                batch_num = int(len(valid_data) / batch_size) + 1

            for batch in range(batch_num):
                batch_images, batch_labels, index_in_validset = next_batch(batch_size, index_in_validset,
                                                                           valid_data)
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

        epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch + 1, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data)))

        writer.add_scalar('Train Loss', train_loss / len(train_data), epoch + 1)
        writer.add_scalar('Train Acc', train_acc / len(train_data), epoch + 1)
        writer.add_scalar('Valid loss', valid_loss / len(valid_data), epoch + 1)
        writer.add_scalar('Valid Acc', valid_acc / len(valid_data), epoch + 1)

        if valid_acc / len(valid_data) > max_vail_acc:
            max_vail_acc = valid_acc / len(valid_data)
            torch.save(net, os.path.join(CHECKPOINT_PATH, save_model_name))

        prev_time = cur_time
        print(epoch_str + time_str)

    writer.close()


def test(use_gpu, test_data, batch_size, save_model_name, result_file):
    test_acc = 0
    index_in_testset = 0
    label_list = []
    outpres_list = []
    prelabels_list = []

    net = torch.load(os.path.join(CHECKPOINT_PATH, save_model_name))
    net = net.eval()

    with torch.no_grad():
        if len(test_data) % batch_size == 0:
            batch_num = int(len(test_data) / batch_size)
        else:
            batch_num = int(len(test_data) / batch_size) + 1

        for batch in range(batch_num):
            batch_images, batch_labels, index_in_testset = next_batch(batch_size, index_in_testset,
                                                                      test_data)
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
            outpres_list.extend(output_softmax.cpu().numpy().tolist())
            prelabels_list.extend(pred_label.cpu().numpy().tolist())

        print("Test Acc: %f" % (test_acc / len(test_data)))

        df = pd.DataFrame(outpres_list, columns=['p0', 'p1', 'p2', 'p3'])
        df.insert(df.shape[1], 'label-pre', prelabels_list)
        df.insert(df.shape[1], 'label_gt', label_list)
        df.to_excel(result_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', type=str, help='输入数据的根路径')
    parser.add_argument('--cut', type=bool, help='是否只截取含肺区域图像')
    parser.add_argument('--use_gpu', type=bool, help='是否只截取含肺区域图像')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--num_epochs', type=int, help='num of epochs')
    parser.add_argument('--save_model_name', type=str, help='model save name')
    parser.add_argument('--result_file', type=str, help='test result file path')
    parser.add_argument('--cuda_device', type=str, help='CUDA_VISIBLE_DEVICES')

    argsin = sys.argv[1:]
    args = parser.parse_args(argsin)

    label_path = os.path.join(args.data_root_path, 'label_match_ct_4_range.xlsx')

    data = load_datapath_label(args.data_root_path, label_path, args.cut)
    train_data = []
    valid_data = []
    test_data = []

    for label in range(4):
        random.shuffle(data[label])

        # 每个标签的数据按 训练集：验证集：测试集 6:1:3
        train_index = int(len(data[label]) * 0.6)
        valid_index = int(len(data[label]) * 0.7)

        train_data.extend(data[label][:train_index])
        valid_data.extend(data[label][train_index:valid_index])
        test_data.extend(data[label][valid_index:])

    channels = 1
    out_features = 4  # 4分类
    pretrained = False  # 是否使用已训练模型

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    net = densenet121(channels, out_features, args.use_gpu, pretrained)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train(net, args.use_gpu, train_data, valid_data, args.batch_size, args.num_epochs, optimizer, criterion,
          args.save_model_name)
    test(args.use_gpu, test_data, args.batch_size, args.save_model_name, args.result_file)

"""
方案一：不处理数据
 nohup python train.py \
 --data_root_path /data/zengnanrong/CTDATA/ \
 --cut False \
 --use_gpu True \
 --batch_size 20 \
 --num_epochs 30 \
 --save_model_name DenseNet121_30epoch.pkl \
 --result_file ./result/test_30epoch.xlsx \
 --cuda_device 1 \
 > out_30epoch.log &
 
 方案二：删去非肺区域的图像
  nohup python train.py \
 --data_root_path /data/zengnanrong/CTDATA/ \
 --cut True \
 --use_gpu True \
 --batch_size 20 \
 --num_epochs 30 \
 --save_model_name DenseNet121_cut_30epoch.pkl \
 --result_file ./result/test_cut_30epoch.xlsx \
 --cuda_device 0 \
 > out_cut_30epoch.log &
 
 方案三：提取肺实质图像
  nohup python train.py \
 --data_root_path /data/zengnanrong/LUNG_SEG/ \
 --cut True \
 --use_gpu True \
 --batch_size 20 \
 --num_epochs 30 \
 --save_model_name DenseNet121_seg_cut_30epoch.pkl \
 --result_file ./result/test_seg_cut_30epoch.xlsx \
 --cuda_device 1 \
 > out_seg_cut_30epoch.log &
"""
