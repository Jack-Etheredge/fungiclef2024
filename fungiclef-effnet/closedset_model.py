"""
modified from https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
"""
import timm
import torchvision.models as models
import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    """
    https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/7

    Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value you are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


def build_model(model_id='tf_efficientnetv2_s.in21k', pretrained=True, fine_tune=True, num_classes=10, dropout_rate=0.5,
                use_timm=True):
    """
    Unfortunately some early experiments were run without timm and the state dict keys don't match.
    """
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    if use_timm:
        model = timm.create_model(model_id, pretrained=pretrained)
    else:
        print(("WARNING: this is a legacy option for evaluation of some of the earliest trained models in this repo."
               "Make sure this is what you wanted to do."))
        model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        # model = models.efficientnet_v2_s(weights='DEFAULT' if pretrained else None)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # Change the final classification head.
    if use_timm:
        model.classifier = nn.Sequential(nn.Dropout(p=dropout_rate, inplace=True),
                                         nn.Linear(in_features=model.classifier.in_features, out_features=num_classes))
    else:
        model.classifier[0] = nn.Dropout(p=dropout_rate, inplace=True)
        model.classifier[1] = nn.Linear(in_features=model.classifier.in_features, out_features=num_classes)
    return model


def get_embedding_size(model_id='tf_efficientnetv2_s.in21k', use_timm=True):
    """
    Unfortunately some early experiments were run without timm and the state dict keys don't match.
    """
    if use_timm:
        model = timm.create_model(model_id, pretrained=False)
    else:
        print(("WARNING: this is a legacy option for evaluation of some of the earliest trained models in this repo."
               "Make sure this is what you wanted to do."))
        model = models.efficientnet_b0(weights=None)
    return model.classifier.in_features


def unfreeze_model(model):
    for params in model.parameters():
        params.requires_grad = True
    return model
