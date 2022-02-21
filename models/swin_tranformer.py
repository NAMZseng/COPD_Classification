import torch
import torch.nn as nn
from torchvision import models
import timm


class SwinBase(nn.Module):
    def __init__(self, in_chans, num_classes):
        super().__init__()
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, in_chans=in_chans,
                                  num_classes=num_classes)
        layer_list = list(model.children())[:-2]

        self.pretrained_model = nn.Sequential(*layer_list)
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(1024, num_classes)
        # self.num_classes = num_classes

    def reshape_transform(tensor, height=7, width=7):
        result = tensor.reshape(tensor.size(0),
                                height, width, tensor.size(2))

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained_model(x)
        features = torch.transpose(features, 2, 1)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        # flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifier(pooled_features)
        return output


def generate_model(use_gpu, in_chans, num_classes):
    model = SwinBase(in_chans=in_chans, num_classes=num_classes)
    if use_gpu:
        model = model.to('cuda')

    return model


if __name__ == '__main__':
    use_gpu = False
    model = generate_model(use_gpu, in_chans=1, num_classes=4)
    print(model)
