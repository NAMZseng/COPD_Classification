import argparse
import os
import sys

import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable

from global_settings import CHECKPOINT_PATH, RESULT_DIR
from models import densenet_3d, swin_tranformer
from models.densenet import densenet121
from models.resnet import generate_model
from models.resnet18 import resnet18
from train import next_batch
from utils.dataset import load_2d_datapath_label, load_3d_datapath_label, load_1316_datapath_label, load_3d_npy_datapath_label


def test(net, net_name, use_gpu, test_data, cut_pic_size, cut_pic_num, batch_size, result_file, scale_num):
    phase = 'test'
    result_dir = os.path.join(RESULT_DIR, net_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    softmax = nn.Softmax(dim=1)
    net.eval()

    with torch.no_grad():

        batch_num = int(len(test_data[0]) / batch_size)

        max_test_acc = 0.0

        if net_name == 'densenet_3D':
            test_epoch = 5
        else:
            test_epoch = 1
        for i in range(test_epoch):
            test_acc = 0
            index_in_testset = [0] * scale_num
            label_list = []
            probability_predicted_list = []
            label_predicted_list = []
            dirs_list = []
            for batch in range(batch_num):
                for scale in range(scale_num):
                    batch_images, batch_lrf, batch_labels, batch_dirs, index_in_testset[scale] = next_batch(batch_size,
                                                                                                            index_in_testset[scale],
                                                                                                            test_data[scale], cut_pic_size,
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
                    output = softmax(output)
                    _, pred_label = output.max(1)
                    num_correct = pred_label.eq(batch_labels).sum()
                    test_acc += num_correct

                    label_list.extend(batch_labels.cpu().numpy().tolist())
                    probability_predicted_list.extend(output.cpu().numpy().tolist())
                    label_predicted_list.extend(pred_label.cpu().numpy().tolist())
                    dirs_list.extend(batch_dirs)

            test_acc = test_acc / (len(test_data[0]) * scale_num)
            if test_acc > max_test_acc:
                print("Test Acc: %f" % test_acc)
                max_test_acc = test_acc
                df = pd.DataFrame(probability_predicted_list, columns=['p0', 'p1', 'p2', 'p3'])
                df.insert(df.shape[1], 'label-pre', label_predicted_list)
                df.insert(df.shape[1], 'label_gt', label_list)
                df.insert(df.shape[1], 'dirs', dirs_list)
                df.to_excel(os.path.join(RESULT_DIR, net_name, result_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', type=str, default='resnet_3D',
                        choices=['densenet_2D', 'densenet_3D', 'swin_base', 'resnet18', 'resnet_3D'], help='使用的网络')
    parser.add_argument('--data_root_path', type=str, default='/data/zengnanrong/lung_seg_normal_resize', help='输入数据的根路径')
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否只使用GPU')
    parser.add_argument('--cut_pic_size', type=bool, default=True, help='是否将图片裁剪压缩')
    parser.add_argument('--cut_pic_num', type=str, choices=['remain', 'precise', 'rough'], default='precise',
                        help='是否只截去不含肺区域的图像，remain:不截，保留原始图像的个数，precise:精筛，rough:粗筛，直接截去上下各1/6的图像数量')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size, 2d:20, 3d:2')
    parser.add_argument('--num_epochs', type=int, default=100, help='num of epochs')
    parser.add_argument('--save_model_name', type=str, default='resnet10_img_multi_scale_finetune.pth', help='checkpoint model name')
    parser.add_argument('--result_file', type=str, default='resnet10_img_multi_scale_finetune.xlsx',
                        help='test result filename')
    parser.add_argument('--cuda_device', type=str, choices=['0', '1'], default=['1'], help='使用哪块GPU')

    args_in = sys.argv[1:]
    args = parser.parse_args(args_in)

    # if args.use_gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    #     torch.cuda.empty_cache()

    channels = 1
    num_classes = 4  # 4分类
    scale_num = 4

    test_label_path = '/data/zengnanrong/label_match_ct_4_range_test.xlsx'
    test_data_root_path = os.path.join(args.data_root_path, 'test')

    if args.net_name == 'densenet_2D':
        test_datapath_label = load_2d_datapath_label(test_data_root_path, test_label_path, args.cut_pic_num)
        net = densenet121(channels, num_classes, args.use_gpu, args.drop_rate)
        net = torch.load(os.path.join(CHECKPOINT_PATH, args.net_name, args.save_model_name))
    elif args.net_name == 'densenet_3D':
        test_datapath_label = load_3d_datapath_label(test_data_root_path, test_label_path)
        net = densenet_3d.generate_model(121, args.use_gpu, n_input_channels=channels, num_classes=num_classes, drop_rate=args.drop_rate)
        net = torch.load(os.path.join(CHECKPOINT_PATH, args.net_name, args.save_model_name))
    elif args.net_name == 'resnet_3D':
        test_datapath_label = load_3d_npy_datapath_label(args.data_root_path, test_label_path)
        # net = resnet_3d.generate_model(10, args.use_gpu, n_input_channels=channels, n_classes=num_classes)
        # net = torch.load(os.path.join(CHECKPOINT_PATH, args.net_name, args.save_model_name))
        checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, args.net_name, args.save_model_name))
        net, _ = generate_model(model_depth=10, use_gpu=args.use_gpu, gpu_id=args.cuda_device, phase='test')
        net.load_state_dict(checkpoint['state_dict'])
    elif args.net_name == 'swin_base':
        test_datapath_label = load_3d_datapath_label(test_data_root_path, test_label_path)
        net = swin_tranformer.generate_model(args.use_gpu, channels, num_classes)
        net = torch.load(os.path.join(CHECKPOINT_PATH, args.net_name, args.save_model_name))
    elif args.net_name == 'resnet18':
        test_datapath_label = load_1316_datapath_label('/data/zengnanrong/dataset1316/test')
        net = resnet18(channels, num_classes, args.use_gpu)
        net = torch.load(os.path.join(CHECKPOINT_PATH, args.net_name, args.save_model_name))

    test_data = [[], [], [], []]

    for label in range(4):
        for scale in range(4):
            test_data[scale].extend(test_datapath_label[scale][label])

    test(net, args.net_name, args.use_gpu, test_data, args.cut_pic_size, args.cut_pic_num, args.batch_size, args.result_file, scale_num)