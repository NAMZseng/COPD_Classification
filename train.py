import argparse
import os
import random
import sys
from datetime import datetime

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from global_settings import CHECKPOINT_PATH, LOG_DIR, TIME_NOW
from models import densenet_3d, swin_tranformer
from models.densenet import densenet121
from models.resnet import generate_model
from models.resnet18 import resnet18
from utils.dataset import load_2d_datapath_label, load_data, load_3d_datapath_label, load_1316_datapath_label, load_3d_npy_datapath_label
from utils.logger import log


def next_batch(batch_size, index_in_total, data, cut_pic_size, cut_pic_num, phase):
    start = index_in_total
    index_in_total += batch_size
    total_num = len(data)

    # 最后一个batch
    if total_num < index_in_total < total_num + batch_size:
        index_in_total = total_num

    end = index_in_total

    batch_images = []
    batch_lrf = []  # Lung radiomics features
    batch_labels = []
    batch_dirs = []

    for i in range(start, end):
        if i < total_num:
            image, lrf = load_data(data[i], cut_pic_size, cut_pic_num, phase)
            batch_lrf.append(lrf)
            batch_images.append(image)

            label = data[i]['label']
            batch_labels.append(label)

            if phase == 'test':
                batch_dirs.append(data[i]['dir'])

    return batch_images, batch_lrf, batch_labels, batch_dirs, index_in_total


def train(net, net_name, use_gpu, train_data, valid_data, cut_pic_size, cut_pic_num, batch_size, num_epochs, optimizer,
          scheduler, criterion, save_model_name, scale_num):
    prev_time = datetime.now()

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, net_name, TIME_NOW))

    phase = 'train_valid'
    global_max_valid_acc = 0.0

    checkpoint_dir = os.path.join(CHECKPOINT_PATH, net_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 采用drop_last方式
    # 每个scale的数据量都是相同的，所以取第一个来计算长度即可
    batch_num_train = int(len(train_data[0]) / batch_size)
    batch_num_valid = int(len(valid_data[0]) / batch_size)

    for epoch in range(num_epochs):
        net.train()
        for scale in range(scale_num):
            random.shuffle(train_data[scale])
            random.shuffle(valid_data[scale])

        train_loss = 0.0
        train_acc = 0
        index_in_trainset = [0] * scale_num

        scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))

        for batch in range(batch_num_train):
            for scale in range(scale_num):
                batch_images, batch_lrf, batch_labels, _, index_in_trainset[scale] = next_batch(batch_size, index_in_trainset[scale],
                                                                                                train_data[scale], cut_pic_size,
                                                                                                cut_pic_num, phase)
                batch_images = torch.tensor(batch_images, dtype=torch.float)
                batch_lrf = torch.tensor(batch_lrf, dtype=torch.float)

                if use_gpu:
                    batch_images = Variable(batch_images.cuda())
                    batch_lrf = Variable(batch_lrf.cuda())
                    batch_labels = Variable(torch.tensor(batch_labels).cuda())
                else:
                    batch_images = Variable(batch_images)
                    batch_lrf = Variable(batch_lrf)
                    batch_labels = Variable(torch.tensor(batch_labels))

                optimizer.zero_grad()  # 清除上一个batch计算的梯度,因为pytorch默认会累积梯度
                output = net(batch_images, batch_lrf)
                loss = criterion(output, batch_labels)  # 计算损失
                loss = loss.requires_grad_()
                loss.backward()  # 计算梯度
                optimizer.step()  # 梯度更新

                train_loss += loss.data
                _, pred_label = output.max(1)
                num_correct = pred_label.eq(batch_labels).sum()
                train_acc += num_correct

        # 评估
        net.eval()
        with torch.no_grad():
            max_valid_acc = 0.0
            min_valid_loss = 10000.0

            if net_name == 'densenet_3D':
                valid_epoch = 5  # 由于densenet 3D采取的是从一个scan中随机抽取N张slices,为了减低偶然性，每次多评估几轮
            else:
                valid_epoch = 1
            for i in range(valid_epoch):
                valid_loss = 0
                valid_acc = 0
                index_in_validset = [0] * scale_num
                for batch in range(batch_num_valid):
                    for scale in range(scale_num):
                        batch_images, batch_lrf, batch_labels, _, index_in_validset[scale] = next_batch(batch_size,
                                                                                                        index_in_validset[scale],
                                                                                                        valid_data[scale], cut_pic_size,
                                                                                                        cut_pic_num, phase)
                        batch_images = torch.tensor(batch_images, dtype=torch.float)
                        batch_lrf = torch.tensor(batch_lrf, dtype=torch.float)

                        if use_gpu:
                            batch_images = Variable(batch_images.cuda())
                            batch_lrf = Variable(batch_lrf.cuda())
                            batch_labels = Variable(torch.tensor(batch_labels).cuda())
                        else:
                            batch_images = Variable(batch_images)
                            batch_lrf = Variable(batch_lrf)
                            batch_labels = Variable(torch.tensor(batch_labels))

                        output = net(batch_images, batch_lrf)
                        loss = criterion(output, batch_labels)
                        valid_loss += loss.data
                        _, pred_label = output.max(1)
                        num_correct = pred_label.eq(batch_labels).sum()
                        valid_acc += num_correct

                if valid_acc / (len(valid_data[0]) * scale_num) > max_valid_acc:
                    max_valid_acc = valid_acc / (len(valid_data[0]) * scale_num)
                if valid_loss / (len(valid_data[0]) * scale_num) < min_valid_loss:
                    min_valid_loss = valid_loss / (len(valid_data[0]) * scale_num)

        train_loss = train_loss / (len(train_data[0]) * scale_num)
        train_acc = train_acc / (len(train_data[0]) * scale_num)
        epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                     % (epoch + 1, train_loss, train_acc, min_valid_loss, max_valid_acc))

        writer.add_scalars('Loss', {'Train': train_loss, 'Valid': min_valid_loss}, epoch + 1)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Valid': max_valid_acc}, epoch + 1)

        if max_valid_acc > global_max_valid_acc:
            global_max_valid_acc = max_valid_acc
            torch.save({'ecpoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(CHECKPOINT_PATH, net_name, save_model_name))

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time
        print(epoch_str + time_str)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', type=str, default='resnet_3D',
                        choices=['densenet_2D', 'densenet_3D', 'swin_base', 'resnet18', 'resnet_3D'], help='使用的网络')
    parser.add_argument('--data_root_path', type=str, default='/data/zengnanrong/lung_seg_normal_resize', help='输入数据的根路径')
    parser.add_argument('--cut_pic_size', type=bool, default=True, help='是否将图片裁剪压缩')
    parser.add_argument('--cut_pic_num', type=str, choices=['remain', 'precise', 'rough'], default='precise',
                        help='是否只截去不含肺区域的图像，remain:不截，保留原始图像的个数，precise:精筛，rough:粗筛，直接截去上下各1/6的图像数量')
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否只使用GPU')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size, 2d:20, 3d:2')
    parser.add_argument('--num_epochs', type=int, default=100, help='num of epochs')
    parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate')
    parser.add_argument('--save_model_name', type=str, default='resnet10_img_multi_scale_finetune.pth', help='model save name')
    parser.add_argument('--cuda_device', type=str, choices=['0', '1'], default=['1'], help='使用哪块GPU')

    args_in = sys.argv[1:]
    args = parser.parse_args(args_in)

    # if args.use_gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    #     torch.cuda.empty_cache()

    channels = 1
    num_classes = 4  # 4分类
    scale_num = 4

    train_valid_label_path = '/data/zengnanrong/label_match_ct_4_range_del1524V2_train_valid.xlsx'
    train_valid_data_root_path = os.path.join(args.data_root_path, 'train_valid')

    if args.net_name == 'densenet_2D':
        train_valid_datapath_label = load_2d_datapath_label(train_valid_data_root_path, train_valid_label_path, args.cut_pic_num)
        net = densenet121(channels, num_classes, args.use_gpu, args.drop_rate)
    elif args.net_name == 'densenet_3D':
        train_valid_datapath_label = load_3d_datapath_label(train_valid_data_root_path, train_valid_label_path)
        net = densenet_3d.generate_model(121, args.use_gpu, n_input_channels=channels, num_classes=num_classes, drop_rate=args.drop_rate)
    elif args.net_name == 'resnet_3D':
        train_valid_datapath_label = load_3d_npy_datapath_label(args.data_root_path, train_valid_label_path)
        # net = resnet_3d.generate_model(10, args.use_gpu, n_input_channels=channels, n_classes=num_classes)
        net, parameters = generate_model(model_depth=10, use_gpu=args.use_gpu, gpu_id=args.cuda_device)
        params = [
            {'params': parameters['base_parameters'], 'lr': args.learning_rate},
            {'params': parameters['new_parameters'], 'lr': args.learning_rate * 100}
        ]
        optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif args.net_name == 'swin_base':
        train_valid_datapath_label = load_3d_datapath_label(train_valid_data_root_path, train_valid_label_path)
        net = swin_tranformer.generate_model(args.use_gpu, channels, num_classes)
    elif args.net_name == 'resnet18':
        train_valid_datapath_label = load_1316_datapath_label('/data/zengnanrong/dataset1316/train')
        net = resnet18(channels, num_classes, args.use_gpu)

    train_data = [[], [], [], []]
    valid_data = [[], [], [], []]

    for label in range(4):
        # 每个标签的数据按 训练集：验证集：测试集 6:1:3
        train_index = int(len(train_valid_datapath_label[0][label]) * 6 / 7)
        while train_valid_datapath_label[0][label][train_index]['dir'] == train_valid_datapath_label[0][label][train_index + 1]['dir']:
            train_index = train_index + 1
        train_index = train_index + 1
        for scale in range(4):
            train_data[scale].extend(train_valid_datapath_label[scale][label][:train_index])
            valid_data[scale].extend(train_valid_datapath_label[scale][label][train_index:])

    # optimizer = torch.optim.Adam(net.parameters(), lr=4e-3, weight_decay=0.0001)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    train(net, args.net_name, args.use_gpu, train_data, valid_data, args.cut_pic_size, args.cut_pic_num, args.batch_size, args.num_epochs,
          optimizer, scheduler, criterion, args.save_model_name, scale_num)
