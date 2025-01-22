import torch
from torch.utils.data import Dataset
from torch_em.util import ensure_spatial_array, ensure_tensor_with_channels

from data_processing.create_heatmap import process_tomogram, get_tomo_shape

class HeatmapLoader(torch.utils.data.Dataset):
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
        depth, height, width = shape
        z_start = np.random.randint(0, depth - self.patch_size[0] + 1)
        y_start = np.random.randint(0, height - self.patch_size[1] + 1)
        x_start = np.random.randint(0, width - self.patch_size[2] + 1)
        return (slice(z_start, z_start + self.patch_size[0]),
                slice(y_start, y_start + self.patch_size[1]),
                slice(x_start, x_start + self.patch_size[2]))

    def _get_sample(self, index):
        if self.sample_random_index:
        index = np.random.randint(0, len(self.raw_images))
        raw, label = self.raw_images[index], self.label_images[index]

        raw = #TODO read zarr file
        self.tomo_shape = get_tomo_shape(self.raw_image_paths) #TODO decide which form does the raw_image_path have and adjust the function accordingly
        label = process_tomogram(self.label_paths, self.tomo_shape)

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
        initial_label_dtype = labels.dtype

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.label_transform is not None:
            labels = self.label_transform(labels)

        if self.transform is not None:
            raw, labels = self.transform(raw, labels)


        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        return raw, labels