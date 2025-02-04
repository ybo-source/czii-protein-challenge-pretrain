import torch
# from torch.utils.data import Dataset
from torch_em.util import ensure_tensor_with_channels
import numpy as np

import zarr
import os

from data_processing.create_heatmap import create_heatmap
from .. import load_data


class HeatmapDataset(torch.utils.data.Dataset):
    max_sampling_attempts = 500

    @staticmethod
    def compute_len(shape, patch_shape):
        if patch_shape is None:
            return 1
        else:
            n_samples = int(np.prod([float(sh / csh) for sh, csh in zip(shape, patch_shape)]))
            return n_samples

    def __init__(
        self,
        raw_image_paths: Union[List[Any], str, os.PathLike],
        label_paths: Union[List[Any], str, os.PathLike],
        patch_shape: Tuple[int, ...],
        raw_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        transfor: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.float32,
        n_samples: Optional[int] = None,
        sampler: Optional[Callable] = None,
        eps: Optional[float] = 10**-8,
        sigma: Optional[float] = None,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None
    ):
        self.raw_images = load_data(raw_image_paths) #TODO check if this is right
        self.label_paths = label_paths
        self._ndim = 3

        assert len(patch_shape) == self._ndim
        self.patch_shape = patch_shape

        self.raw_transform = raw_transform
        self.label_transform = label_transform
        self.transform = transform
        self.sampler = sampler

        self.dtype = dtype
        self.label_dtype = label_dtype

        # Julias arguments!
        self.eps = eps
        self.sigma = sigma
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # TODO: check if this is right?
        self.raw_images = np.array(self.raw_images) if not isinstance(self.raw_images, np.ndarray) else self.raw_images
        self.shape = self.raw_images[1:].shape
        self._len = self.compute_len(self.shape, self.patch_shape) if n_samples is None else n_samples

        #TODO do i need the sample random index stuff???
        '''self.sample_random_index = False if n_samples is None else True'''

    def __len__(self):
        return self._len

    @property
    def ndim(self):
        return self._ndim

    def _sample_bounding_box(self, shape):
        if any(sh < psh for sh, psh in zip(shape, self.patch_shape)):
            raise NotImplementedError(
                f"Image padding is not supported yet. Data shape {shape}, patch shape {self.patch_shape}"
            )
        bb_start = [
            np.random.randint(0, sh - psh) if sh - psh > 0 else 0
            for sh, psh in zip(shape, self.patch_shape)
        ]
        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, self.patch_shape))

    def _get_sample(self, index):
        #TODO do i need the sample random index stuff???
        '''if self.sample_random_index:
            index = np.random.randint(0, len(self.raw_images))'''
        raw, label_path = self.raw_images[index], self.label_paths[index]

        have_raw_channels = raw.ndim == 4  # 3D with channels
        
        prefix_box = tuple()
        if have_raw_channels:
            if shape[-1] < 16:
                shape = shape[:-1]
            else:
                shape = shape[1:]
                prefix_box = (slice(None), )

        bb = self._sample_bounding_box(shape)
        
        raw_patch = np.array(raw[prefix_box + bb])
        # sigma is not really used in my create_heatmap ... TODO ?
        label_patch = create_heatmap(
            label_path, raw.shape, eps=self.eps, sigma=self.sigma,
            lower_bound=self.lower_bound, upper_bound=self.upper_bound, bb=bb
        )

        have_label_channels = label_patch.ndim == 4
        if have_label_channels:
            raise NotImplementedError("Multi-channel labels are not supported.")


        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw_patch, label_patch):
                bb = self._sample_bounding_box(shape)
                raw_patch = np.array(raw[prefix_box + bb])
                # sigma is not really used in my create_heatmap ... TODO ?
                label_patch = create_heatmap(
                    label_path, raw.shape, eps=self.eps, sigma=self.sigma,
                    lower_bound=self.lower_bound, upper_bound=self.upper_bound, bb=bb
                )
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(f"Could not sample a valid batch in {self.max_sampling_attempts} attempts")

        if have_raw_channels and len(prefix_box) == 0:
            raw_patch = raw_patch.transpose((3, 0, 1, 2))  # Channels, Depth, Height, Width

        return raw_patch, label_patch

    def __getitem__(self, index):
        raw, labels = self._get_sample(index)
        # initial_label_dtype = labels.dtype

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.label_transform is not None:
            labels = self.label_transform(labels)

        if self.transform is not None:
            raw, labels = self.transform(raw, labels)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        return raw, labels
