import torch
import torch.nn as nn
import timm


class EfficientnetV2(nn.Module):
    def __init__(self, in_chans, n_classes):
        super().__init__()
        model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=False, in_chans=in_chans)
        layer_list = list(model.children())[:-2]
        self.pretrained_model = nn.Sequential(*layer_list)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(1280, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained_model(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifer(flattened_features)
        return output


def generate_model(use_gpu, in_chans, num_classes):
    model = EfficientnetV2(in_chans=in_chans, n_classes=num_classes)
    if use_gpu:
        model = model.to('cuda')

    return model


if __name__ == '__main__':
    use_gpu = False
    # model = generate_model(use_gpu, in_chans=1, num_classes=4)
    model = timm.create_model('tf_efficientnetv2_m_in21k', pretrained=False)

    print(model)
