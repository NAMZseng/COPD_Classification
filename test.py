import argparse
import os
import sys

import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable

from global_settings import CHECKPOINT_PATH, RESULT_DIR
from models import tinynet, resnet
from utils.dataset import Dataset


def test(net, net_name, use_gpu, test_loader, result_file):
    result_dir = os.path.join(RESULT_DIR, net_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    net.eval()
    with torch.no_grad():
        test_acc = 0
        label_list = []
        probability_predicted_list = []
        label_predicted_list = []
        dirs_list = []
        for batch_images, batch_labels, batch_subjects in test_loader:
            if use_gpu:
                batch_images = Variable(batch_images.cuda())
                batch_labels = Variable(batch_labels.cuda())
            else:
                batch_images = Variable(batch_images)
                batch_labels = Variable(batch_labels)

            output = net(batch_images)
            softmax = nn.Softmax(dim=1)
            output = softmax(output)
            _, pred_label = output.max(1)
            num_correct = pred_label.eq(batch_labels).sum()
            test_acc += num_correct

            label_list.extend(batch_labels.cpu().numpy().tolist())
            probability_predicted_list.extend(output.cpu().numpy().tolist())
            label_predicted_list.extend(pred_label.cpu().numpy().tolist())
            dirs_list.extend(batch_subjects.cpu().numpy().tolist())

        test_acc = test_acc / len(test_loader)
        print("Test Acc: %f" % test_acc)
        df = pd.DataFrame(probability_predicted_list, columns=['p0', 'p1', 'p2', 'p3'])
        df.insert(df.shape[1], 'label-pre', label_predicted_list)
        df.insert(df.shape[1], 'label_gt', label_list)
        df.insert(df.shape[1], 'dirs', dirs_list)
        df.to_excel(os.path.join(RESULT_DIR, net_name, result_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', type=str, default='resnet_3D', choices=['resnet_3D', 'tinynet'], help='net model to use')
    parser.add_argument('--data_root_path', type=str, default='/data/zengnanrong/lung_seg_normal_resize', help='input data path')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use GPU')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--save_model_name', type=str, default='resnet10_img_multi_scale_finetune.pth', help='checkpoint model name')
    parser.add_argument('--result_file', type=str, default='resnet10_img_multi_scale_finetune.xlsx',
                        help='test result filename')
    parser.add_argument('--cuda_device', type=str, choices=['0', '1'], default=['0'], help='which GPU(s) to use')

    args_in = sys.argv[1:]
    args = parser.parse_args(args_in)

    test_label_path = '/data/zengnanrong/label_match_ct_4_range_test.xlsx'
    test_dataset = Dataset(args.data_root_path, test_label_path)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=False)

    if args.net_name == 'resnet_3D':
        net, _ = resnet.generate_model(model_depth=10, use_gpu=args.use_gpu, gpu_id=args.cuda_device, phase='test')
    elif args.net_name == 'tinynet':
        net, _ = tinynet.generate_model(use_gpu=args.use_gpu, gpu_id=args.cuda_device)

    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, args.net_name, args.save_model_name))
    net.load_state_dict(checkpoint['state_dict'])

    test(net, args.net_name, args.use_gpu, test_loader, args.result_file)
