"""
modified from https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
"""

import torchvision.models as models
import torch.nn as nn


def build_model(pretrained=True, fine_tune=True, num_classes=10):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
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
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model


def unfreeze_model(model):
    for params in model.parameters():
        params.requires_grad = True
    return model
