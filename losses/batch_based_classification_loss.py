import torch
import torch.nn.functional as F

from trainers.abc import AbstractBaseMetricLoss


class BatchBasedClassificationLoss(AbstractBaseMetricLoss):
    def __init__(self):
        super().__init__()

    def forward(self, ref_features, tar_features):
        return self.cal_loss(ref_features, tar_features)

    @classmethod
    def cal_loss(cls, ref_features, tar_features):
        batch_size = ref_features.size(0)
        device = ref_features.device

        pred = ref_features.mm(tar_features.transpose(0, 1))
        labels = torch.arange(0, batch_size).long().to(device)
        return F.cross_entropy(pred, labels)

    @classmethod
    def code(cls):
        return 'batch_based_classification_loss'
