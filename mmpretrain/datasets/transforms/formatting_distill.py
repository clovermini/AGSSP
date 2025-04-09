# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import defaultdict
from collections.abc import Sequence

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from mmcv.transforms import BaseTransform
from mmengine.utils import is_str
from PIL import Image

from mmpretrain.registry import TRANSFORMS
from mmengine.structures import PixelData
from mmpretrain.structures import DataAnomalySample


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


@TRANSFORMS.register_module()
class PackAnomalyInputs(BaseTransform):
    """Pack the inputs data.

    **Required Keys:**

    - ``input_key``
    - ``*algorithm_keys``
    - ``*meta_keys``

    **Deleted Keys:**

    All other keys in the dict.

    **Added Keys:**

    - inputs (:obj:`torch.Tensor`): The forward data of models.
    - data_samples (:obj:`~mmpretrain.structures.DataSample`): The
      annotation info of the sample.

    Args:
        input_key (str): The key of element to feed into the model forwarding.
            Defaults to 'img'.
        algorithm_keys (Sequence[str]): The keys of custom elements to be used
            in the algorithm. Defaults to an empty tuple.
        meta_keys (Sequence[str]): The keys of meta information to be saved in
            the data sample. Defaults to :attr:`PackInputs.DEFAULT_META_KEYS`.

    .. admonition:: Default algorithm keys

        Besides the specified ``algorithm_keys``, we will set some default keys
        into the output data sample and do some formatting. Therefore, you
        don't need to set these keys in the ``algorithm_keys``.

        - ``gt_label``: The ground-truth label. The value will be converted
          into a 1-D tensor.
        - ``gt_score``: The ground-truth score. The value will be converted
          into a 1-D tensor.
        - ``mask``: The mask for some self-supervise tasks. The value will
          be converted into a tensor.

    .. admonition:: Default meta keys

        - ``sample_idx``: The id of the image sample.
        - ``img_path``: The path to the image file.
        - ``ori_shape``: The original shape of the image as a tuple (H, W).
        - ``img_shape``: The shape of the image after the pipeline as a
          tuple (H, W).
        - ``scale_factor``: The scale factor between the resized image and
          the original image.
        - ``flip``: A boolean indicating if image flip transform was used.
        - ``flip_direction``: The flipping direction.
    """

    DEFAULT_META_KEYS = ('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'flip', 'flip_direction')

    def __init__(self,
                 input_key='img',
                 algorithm_keys=(),
                 meta_keys=DEFAULT_META_KEYS):
        self.input_key = input_key
        self.algorithm_keys = algorithm_keys
        self.meta_keys = meta_keys

    @staticmethod
    def format_input(input_):
        if isinstance(input_, list):
            return [PackAnomalyInputs.format_input(item) for item in input_]
        elif isinstance(input_, np.ndarray):
            if input_.ndim == 2:  # For grayscale image.
                input_ = np.expand_dims(input_, -1)
            if input_.ndim == 3 and not input_.flags.c_contiguous:
                input_ = np.ascontiguousarray(input_.transpose(2, 0, 1))
                input_ = to_tensor(input_)
            elif input_.ndim == 3:
                # convert to tensor first to accelerate, see
                # https://github.com/open-mmlab/mmdetection/pull/9533
                input_ = to_tensor(input_).permute(2, 0, 1).contiguous()
            else:
                # convert input with other shape to tensor without permute,
                # like video input (num_crops, C, T, H, W).
                input_ = to_tensor(input_)
        elif isinstance(input_, Image.Image):
            input_ = F.pil_to_tensor(input_)
        elif not isinstance(input_, torch.Tensor):
            raise TypeError(f'Unsupported input type {type(input_)}.')

        return input_

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""

        packed_results = dict()
        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results['inputs'] = self.format_input(input_)

        data_sample = DataAnomalySample()

        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...])
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'])
            gt_sem_seg_data = dict(data=data)
            #print('PackAnomalyInputs gt_seg_map ', torch.max(data), ' -- min ', torch.min(data))
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        # Set default keys
        if 'gt_label' in results:
            data_sample.set_gt_label(results['gt_label'])
        if 'gt_score' in results:
            data_sample.set_gt_score(results['gt_score'])
        if 'mask' in results:
            data_sample.set_mask(results['mask'])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type='metainfo')

        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_key='{self.input_key}', "
        repr_str += f'algorithm_keys={self.algorithm_keys}, '
        repr_str += f'meta_keys={self.meta_keys})'
        return repr_str