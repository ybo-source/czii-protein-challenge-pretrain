import torch
# from torch.utils.data import Dataset
from torch_em.util import ensure_tensor_with_channels
import numpy as np

import zarr
import os

from data_processing.create_heatmap import process_tomogram


class HeatmapDataset(torch.utils.data.Dataset):
    max_sampling_attempts = 500

    def __init__(
        self,
        raw_image_paths,
        label_paths,
        patch_shape,
        raw_transform=None,
        label_transform=None,
        transform=None,
        dtype=torch.float32,
        label_dtype=torch.float32,
        n_samples=None,
        sampler=None,
        eps=10**-8,
        sigma=None,
        lower_bound=None,
        upper_bound=None
    ):
        self.raw_images = raw_image_paths
        self.label_images = label_paths
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

        if n_samples is None:
            self._len = len(self.raw_images)
            self.sample_random_index = False
        else:
            self._len = n_samples
            self.sample_random_index = True

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
        if self.sample_random_index:
            index = np.random.randint(0, len(self.raw_images))
        raw, label = self.raw_images[index], self.label_images[index]

        # TODO this is specific for challenge zarr files now, maybe need to generalize in the future
        # Yes ;). We can discuss this soon.

        # zarr_file = zarr.open(f"{raw}/VoxelSpacing10.000/denoised.zarr/0", mode='r')
        zarr_file = zarr.open(os.path.join(raw, "VoxelSpacing10.000", "denoised.zarr", "0"), mode='r')

        # This was very inefficient!
        # You first load the full data from zarr and then later load the bounding box.
        # raw = zarr_file[:]
        # Instead, you can just load the bounding box from the zarr
        raw = zarr_file

        # This is also quite inefficient.
        # You compute he labels for the full tomogram, and then sub-sample.
        # sigma is not really used in my process_tomogram ... TODO ?
        label = process_tomogram(
            label, raw.shape, eps=self.eps, sigma=self.sigma,
            lower_bound=self.lower_bound, upper_bound=self.upper_bound
        )

        have_raw_channels = raw.ndim == 4  # 3D with channels
        have_label_channels = label.ndim == 4
        if have_label_channels:
            raise NotImplementedError("Multi-channel labels are not supported.")

        shape = raw.shape
        prefix_box = tuple()
        if have_raw_channels:
            if shape[-1] < 16:
                shape = shape[:-1]
            else:
                shape = shape[1:]
                prefix_box = (slice(None), )

        bb = self._sample_bounding_box(shape)
        raw_patch = np.array(raw[prefix_box + bb])
        label_patch = np.array(label[bb])

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw_patch, label_patch):
                bb = self._sample_bounding_box(shape)
                raw_patch = np.array(raw[prefix_box + bb])
                label_patch = np.array(label[bb])
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
