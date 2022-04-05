import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyNet(nn.Module):
    def __init__(self, in_channel=1, num_classes=4):
        super(TinyNet, self).__init__()

        self.conv_layer1 = self._conv_layer_set(in_channel, 32, 3)
        self.conv_layer2 = self._conv_layer_set(32, 64, 3)
        self.conv_layer3 = self._conv_layer_set(64, 128, 3)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, num_classes)

        # 网络参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _conv_layer_set(self, in_c, out_c, kernel_size):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2)),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0))

        return conv_layer

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out


def generate_model(use_gpu=True, gpu_id=['1']):
    model = TinyNet()

    if use_gpu:
        if len(gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=gpu_id)
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id[0])
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)

    return model, model.parameters()


if __name__ == '__main__':
    model = TinyNet()
    x = torch.randn((4, 1, 224, 224, 224))
    out = model.forward(x)
    print(out)
