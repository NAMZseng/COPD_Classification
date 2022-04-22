import torch
import torch.nn as nn


class TinyNet(nn.Module):
    def __init__(self, in_channel=1, num_classes=4):
        super(TinyNet, self).__init__()

        self.conv_layer1 = self._basic_block(in_channel, 64)
        self.conv_layer2 = self._basic_block(64, 64)
        self.conv_layer3 = self._basic_block(64, 128)
        self.conv_layer4 = self._basic_block(128, 256)

        self.drop = nn.Dropout(0.5)
        self.nin_block = self._nin_block(256, num_classes, kernel_size=3, strides=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 网络参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _basic_block(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1))

        return conv_layer

    def _nin_block(self, in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.drop(x)
        x = self.nin_block(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


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
    x = torch.randn((4, 1, 128, 128, 128))
    out = model.forward(x)
    print(out)
