from torch import nn
from torchvision import models

def resnet18(channels, out_features, use_gpu):
    model = models.resnet18(pretrained=False, num_classes=out_features)
    model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if use_gpu:
        model = model.cuda()

    return model


if __name__ == '__main__':
    net = resnet18(1, 4, False)
    print(net)
