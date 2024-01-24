# Copyright (c) OpenMMLab. All rights reserved.
from .textdet_data_sample import TextDetDataSample
from .textrecog_data_sample import TextRecogDataSample
from .kie_data_sample import KIEDataSample
from .textspotting_data_sample import TextSpottingDataSample

__all__ = [
    'TextDetDataSample', 'TextRecogDataSample', 'KIEDataSample',
    'TextSpottingDataSample'
]