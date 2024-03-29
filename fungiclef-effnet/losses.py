import torch
from torch.nn.functional import cross_entropy, one_hot, softmax


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
