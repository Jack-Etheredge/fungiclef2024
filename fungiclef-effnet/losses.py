import torch
import torch.nn as nn
from focal_loss import FocalLoss
from torch.nn.functional import cross_entropy, one_hot, softmax, log_softmax

# df = pd.read_csv('./metadata/poison_status_list.csv')
# df[df['poisonous']==1]['class_id'].unique()
POISONOUS_CLASS_IDS = torch.Tensor([989, 1050, 413, 581, 320, 687, 309, 506, 252, 594, 992,
                                    586, 689, 52, 35, 688, 1079, 724, 726, 852, 44, 40,
                                    1332, 25, 286, 728, 42, 50, 43, 358, 263, 45, 445,
                                    1398, 1179, 843, 796, 281, 1281, 1246, 1311, 968, 1513, 229,
                                    1312, 211, 20, 307, 842])


class SeesawLoss(torch.nn.Module):
    """
    Implementation of seesaw loss.
    Refers to `Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)
    <https://arxiv.org/abs/2008.10032>
    Args:
        num_classes (int): The number of classes.
                Default to 1000 for the ImageNet dataset.
        p (float): The ``p`` in the mitigation factor.
                Defaults to 0.8.
        q (float): The ``q`` in the compensation factor.
                Defaults to 2.0.
        eps (float): The min divisor to smooth the computation of compensation factor.
                Default to 1e-2.
    """

    def __init__(self, num_classes=1000,
                 p=0.8, q=2.0, eps=1e-2, device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.q = q
        self.eps = eps
        self.device = device

        # cumulative samples for each category
        self.register_buffer('accumulate',
                             torch.zeros(self.num_classes, dtype=torch.float))
        self.accumulate = self.accumulate.to(device)

    def forward(self, output, target):
        # accumulate the samples for each category
        for unique in target.unique():
            self.accumulate[unique] += (target == unique.item()).sum()

        onehot_target = one_hot(target, self.num_classes)
        seesaw_weights = output.new_ones(onehot_target.size())

        # mitigation factor
        if self.p > 0:
            matrix = self.accumulate[None, :].clamp(min=1) / self.accumulate[:, None].clamp(min=1)
            index = (matrix < 1.0).float()
            sample_weights = matrix.pow(self.p) * index + (1 - index)  # M_{ij}
            mitigation_factor = sample_weights[target.long(), :]
            seesaw_weights = seesaw_weights * mitigation_factor

        # compensation factor
        if self.q > 0:
            scores = softmax(output.detach(), dim=1)
            self_scores = scores[torch.arange(0, len(scores)).to(scores.device).long(), target.long()]
            score_matrix = scores / self_scores[:, None].clamp(min=self.eps)
            index = (score_matrix > 1.0).float()
            compensation_factor = score_matrix.pow(self.q) * index + (1 - index)
            seesaw_weights = seesaw_weights * compensation_factor

        output = output + (seesaw_weights.log() * (1 - onehot_target))

        return cross_entropy(output, target, weight=None, reduction='none').mean()


class SupConLoss(nn.Module):
    """
    https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Author: Yonglong Tian (yonglong@mit.edu)
    Date: May 07, 2020
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    """

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class UnknownLoss(torch.nn.Module):

    def forward(self, model_outs):
        # modified from https://github.com/RenHuan1999/FungiCLEF2023-UstcAIGroup/blob/main/models/custom_loss.py
        labels_nov = torch.ones_like(model_outs) / model_outs.shape[1]
        loss_cls_classes_nov = - (labels_nov * log_softmax(model_outs, dim=1)).sum(dim=-1)
        return loss_cls_classes_nov.mean()


class CompositeLoss(torch.nn.Module):

    def __init__(self, multiclass_loss, use_poison_loss=True, poisonous_class_ids=POISONOUS_CLASS_IDS):
        super().__init__()
        self.multiclass_loss = multiclass_loss
        self.unknown_loss = UnknownLoss()
        # the penalty is 100x for classifying a poisonous (1) sample as edible (0) vs edible as poisonous
        # therefore we want to give a 100x weight to misclassifications of poisonous
        self.softmax = nn.Softmax(dim=-1)
        self.use_poison_loss = use_poison_loss
        self.poisonous_class_ids = poisonous_class_ids

    def forward(self, outputs, labels):
        open_outputs, closed_outputs, _, closed_labels, n_open, n_closed = self.split_open_closed(outputs, labels)
        if n_closed > 0:
            if isinstance(self.multiclass_loss, FocalLoss):
                loss = self.multiclass_loss(self.softmax(closed_outputs), closed_labels)
            else:
                loss = self.multiclass_loss(closed_outputs, closed_labels)
        else:
            loss = torch.tensor(0.)
        if n_open > 0:
            open_loss = self.unknown_loss(open_outputs)
        else:
            open_loss = torch.tensor(0.)
        total_samples = (n_closed + n_open)
        closed_proportion = n_closed / total_samples
        open_proportion = n_open / total_samples
        total_loss = loss * closed_proportion + open_loss * open_proportion
        if self.use_poison_loss:
            self.poisonous_class_ids = self.poisonous_class_ids.to(labels.device)
            softmax_outputs = self.softmax(outputs)
            poisonous_proba_sum = softmax_outputs[:, self.poisonous_class_ids.int()].sum(-1)
            edible_proba_sum = softmax_outputs.sum(-1) - poisonous_proba_sum
            poisonous_outputs = torch.concatenate([edible_proba_sum.unsqueeze(-1), poisonous_proba_sum.unsqueeze(-1)],
                                                  dim=1)
            poisonous_labels = torch.isin(labels, self.poisonous_class_ids)
            class_weight = torch.tensor([1., 100.]).to(labels.device)
            # TODO: also account for class balance in the class weights
            poison_loss = cross_entropy(poisonous_outputs, poisonous_labels.type(torch.uint8), weight=class_weight)
            total_loss = total_loss + poison_loss
        return total_loss

    @staticmethod
    def split_open_closed(outputs, labels):
        """
        get the open and closed labels and outputs from the labels and outputs to calculate losses separately
        """
        closed_indices = labels >= 0
        open_indices = labels == -1
        n_closed = closed_indices.sum()
        n_open = open_indices.sum()
        closed_labels = labels[closed_indices]
        open_labels = labels[open_indices]
        closed_outputs = outputs[closed_indices]
        open_outputs = outputs[open_indices]
        return open_outputs, closed_outputs, open_labels, closed_labels, n_open, n_closed
