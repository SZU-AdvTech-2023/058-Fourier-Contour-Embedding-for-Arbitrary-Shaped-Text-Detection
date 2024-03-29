# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmocr_.registry import MODELS


@MODELS.register_module()
class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross entropy loss."""
