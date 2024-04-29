"""
modified from https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
"""
from pathlib import Path

import timm
import torchvision.models as models
import torch
import torch.nn as nn
from models.MetaFG import *
from models.MetaFG_meta import *


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


def build_model(model_id='tf_efficientnetv2_s.in21k', pretrained=True, fine_tune=True, num_classes=10,
                dropout_rate=0.5,
                use_timm=True):
    """
    Unfortunately some early experiments were run without timm and the state dict keys don't match.
    """
    if pretrained:
        print('Loading pre-trained weights')
    else:
        print('Not loading pre-trained weights')
    if use_timm:
        if model_id in {"MetaFG_meta_0", "MetaFG_meta_1", "MetaFG_meta_2"}:
            model = timm.create_model(
                model_id,
                pretrained=False,
                num_classes=num_classes,
                drop_path_rate=0.1,  # from default
                img_size=384,  # model should be invariant to size (within reason)
                only_last_cls=False,  # from default
                extra_token_num=5,  # from model config
                meta_dims=[4, 34, 32, 31],  # temporal dims, country codes, substrates, habitats
                use_arcface=False,  # from default
                never_mask=False,  # from default
            )
            # print(f"built {model_id}. loading pretrained weights into model will need to be performed separately.")
            if pretrained:
                # TODO: consider moving this pathing to hydra config
                filenames = {"MetaFG_meta_0": "metafg_0_inat21_384.pth",
                             "MetaFG_meta_1": None,
                             "MetaFG_meta_2": "metafg_0_inat21_384.pth", }
                filename = filenames.get(model_id)
                if filename is None:
                    raise ValueError(f"no associated filename with model_id {model_id}")
                pretrained_model_path = f"~/pretrained_models/{filenames.get(model_id)}"
                load_pretrained_metaformer(model, pretrained_model_path)
        else:
            model = timm.create_model(model_id, pretrained=pretrained)
    else:
        print(("WARNING: this is a legacy option for evaluation of some of the earliest trained models in this repo."
               "Make sure this is what you wanted to do."))
        model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        # model = models.efficientnet_v2_s(weights='DEFAULT' if pretrained else None)
    if fine_tune:
        print('Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # Change the final classification head.
    if use_timm:
        try:
            model.classifier = nn.Sequential(nn.Dropout(p=dropout_rate, inplace=True),
                                             nn.Linear(in_features=model.classifier.in_features,
                                                       out_features=num_classes))
        except:
            model.head = nn.Sequential(nn.Dropout(p=dropout_rate, inplace=True),
                                       nn.Linear(in_features=model.head.in_features, out_features=num_classes))
    else:
        model.classifier[0] = nn.Dropout(p=dropout_rate, inplace=True)
        model.classifier[1] = nn.Linear(in_features=model.classifier.in_features, out_features=num_classes)
    return model


def update_dropout_rate(model, dropout_rate):
    try:
        model.classifier[0] = nn.Dropout(p=dropout_rate, inplace=True)
    except:
        model.head[0] = nn.Dropout(p=dropout_rate, inplace=True)
    return model


def get_embedding_size(model_id='tf_efficientnetv2_s.in21k', use_timm=True):
    """
    Unfortunately some early experiments were run without timm and the state dict keys don't match.
    """
    model = build_model(model_id=model_id, pretrained=False, use_timm=use_timm)
    try:
        embedding_size = model.classifier.in_features
    except:
        embedding_size = model.head.in_features
    return embedding_size


def unfreeze_model(model):
    for params in model.parameters():
        params.requires_grad = True
    return model


def load_pretrained_metaformer(model, pretrained_model_path, strict=False, drop_head=True, drop_meta=True,
                               image_size=384):
    print(f"==============> pretrain from {pretrained_model_path}....................")
    checkpoint = torch.load(pretrained_model_path, map_location='cpu')
    if 'model' not in checkpoint:
        if 'state_dict_ema' in checkpoint:
            checkpoint['model'] = checkpoint['state_dict_ema']
        else:
            checkpoint['model'] = checkpoint
    if drop_head:
        if 'head.weight' in checkpoint['model'] and 'head.bias' in checkpoint['model']:
            print(f"==============> drop head....................")
            del checkpoint['model']['head.weight']
            del checkpoint['model']['head.bias']
        if 'head.fc.weight' in checkpoint['model'] and 'head.fc.bias' in checkpoint['model']:
            print(f"==============> drop head....................")
            del checkpoint['model']['head.fc.weight']
            del checkpoint['model']['head.fc.bias']
    if drop_meta:
        print(f"==============> drop meta head....................")
        for k in list(checkpoint['model']):
            if 'meta' in k:
                del checkpoint['model'][k]

    checkpoint = relative_bias_interpolate(checkpoint, image_size)
    if 'point_coord' in checkpoint['model']:
        print(f"==============> drop point coord....................")
        del checkpoint['model']['point_coord']
    msg = model.load_state_dict(checkpoint['model'], strict=strict)
    print(msg)
    del checkpoint
    torch.cuda.empty_cache()


def relative_bias_interpolate(checkpoint, image_size):
    for k in list(checkpoint['model']):
        if 'relative_position_index' in k:
            del checkpoint['model'][k]
        if 'relative_position_bias_table' in k:
            relative_position_bias_table = checkpoint['model'][k]
            cls_bias = relative_position_bias_table[:1, :]
            relative_position_bias_table = relative_position_bias_table[1:, :]
            size = int(relative_position_bias_table.shape[0] ** 0.5)
            img_size = (size + 1) // 2
            if 'stage_3' in k:
                downsample_ratio = 16
            elif 'stage_4' in k:
                downsample_ratio = 32
            new_img_size = image_size // downsample_ratio
            new_size = 2 * new_img_size - 1
            if new_size == size:
                continue
            relative_position_bias_table = relative_position_bias_table.reshape(size, size, -1)
            relative_position_bias_table = relative_position_bias_table.unsqueeze(0).permute(0, 3, 1, 2)  # bs,nhead,h,w
            relative_position_bias_table = torch.nn.functional.interpolate(
                relative_position_bias_table, size=(new_size, new_size), mode='bicubic', align_corners=False)
            relative_position_bias_table = relative_position_bias_table.permute(0, 2, 3, 1)
            relative_position_bias_table = relative_position_bias_table.squeeze(0).reshape(new_size * new_size, -1)
            relative_position_bias_table = torch.cat((cls_bias, relative_position_bias_table), dim=0)
            checkpoint['model'][k] = relative_position_bias_table
    return checkpoint
