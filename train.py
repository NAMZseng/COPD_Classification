import argparse
import os
import sys
from datetime import datetime

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from global_settings import CHECKPOINT_PATH, LOG_DIR, TIME_NOW
from models import resnet, tinynet
from utils.dataset import Dataset
from utils.logger import log


def train(net, net_name, use_gpu, train_loader, valid_loader, num_epochs, optimizer, scheduler, criterion, save_model_name):
    prev_time = datetime.now()

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, net_name, TIME_NOW))

    checkpoint_dir = os.path.join(CHECKPOINT_PATH, net_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    max_valid_acc = 0.0
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0
        net.train()
        for batch_images, batch_labels, _ in train_loader:
            if use_gpu:
                batch_images = Variable(batch_images.cuda())
                batch_labels = Variable(batch_labels.cuda())
            else:
                batch_images = Variable(batch_images)
                batch_labels = Variable(batch_labels)

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
        net.eval()
        valid_loss = 0.0
        valid_acc = 0
        for batch_images, batch_labels, _ in valid_loader:
            if use_gpu:
                batch_images = Variable(batch_images.cuda())
                batch_labels = Variable(batch_labels.cuda())
            else:
                batch_images = Variable(batch_images)
                batch_labels = Variable(batch_labels)

            output = net(batch_images)
            loss = criterion(output, batch_labels)
            valid_loss += loss.data
            _, pred_label = output.max(1)
            num_correct = pred_label.eq(batch_labels).sum()
            valid_acc += num_correct

        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)
        valid_acc = valid_acc / len(valid_loader)
        valid_loss = valid_loss / len(valid_loader)

        epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                     % (epoch + 1, train_loss, train_acc, valid_loss, valid_acc))

        writer.add_scalars('Loss', {'Train': train_loss, 'Valid': valid_loss}, epoch + 1)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Valid': valid_acc}, epoch + 1)

        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            torch.save({'ecpoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(CHECKPOINT_PATH, net_name, save_model_name))

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time
        print(epoch_str + time_str)

        scheduler.step()
        log.info('lr = {}'.format(scheduler.get_last_lr()))

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', type=str, default='tinynet', choices=['resnet_3D', 'tinynet'], help='net model to use')
    parser.add_argument('--data_root_path', type=str, default='/data/zengnanrong/lung_seg_normal_resize', help='input data path')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use GPU')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=200, help='num of epochs')
    parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate')
    parser.add_argument('--save_model_name', type=str, default='debug.pth', help='model save name')
    parser.add_argument('--cuda_device', type=str, choices=['0', '1'], default=['1'], help='which GPU(s) to use')

    args_in = sys.argv[1:]
    args = parser.parse_args(args_in)

    train_label_path = '/data/zengnanrong/label_match_ct_4_range_del1524V2_train.xlsx'
    train_dataset = Dataset(args.data_root_path, train_label_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=False)

    valid_label_path = '/data/zengnanrong/label_match_ct_4_range_del1524V2_valid.xlsx'
    valid_dataset = Dataset(args.data_root_path, valid_label_path)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=False)

    if args.net_name == 'resnet_3D':
        net, parameters = resnet.generate_model(model_depth=10, use_gpu=args.use_gpu, gpu_id=args.cuda_device)
        params = [
            {'params': parameters['base_parameters'], 'lr': args.learning_rate},
            {'params': parameters['new_parameters'], 'lr': args.learning_rate * 10}
        ]
    elif args.net_name == 'tinynet':
        net, parameters = tinynet.generate_model(use_gpu=args.use_gpu, gpu_id=args.cuda_device)
        params = [{'params': parameters, 'lr': args.learning_rate}]

    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
    # optimizer = torch.optim.Adam(params, weight_decay=1e-3)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.CrossEntropyLoss()

    train(net, args.net_name, args.use_gpu, train_loader, valid_loader, args.num_epochs, optimizer, scheduler, criterion,
          args.save_model_name)
