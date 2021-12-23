import torch
from torchvision import models

from dataset import load_data


def densenet121(channels, out_features, use_gpu, drop_rate):
    model = models.densenet121(drop_rate=drop_rate, num_classes=out_features)

    for parma in model.parameters():
        parma.requires_grad = False

    model.features.conv0 = torch.nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    if use_gpu:
        model = model.cuda()

    return model


if __name__ == '__main__':
    net = densenet121(1, 4, False, 0.5)
    print(net)
