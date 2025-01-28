from torch.utils.data import DataLoader
from .heatmap_loader import HeatmapLoader


def create_data_loader(
    train_images, train_labels,
    val_images, val_labels,
    test_images,
    test_labels,
    raw_transform, transform,
    patch_shape, num_workers, batch_size,
    eps=0.00001, sigma=None,
    lower_bound=None, upper_bound=None
):
    train_set = HeatmapLoader(
        train_images, train_labels, patch_shape,
        raw_transform=raw_transform, transform=transform, eps=eps, sigma=sigma,
        lower_bound=lower_bound, upper_bound=upper_bound
    )
    val_set = HeatmapLoader(
        val_images, val_labels, patch_shape,
        raw_transform=raw_transform, transform=transform, eps=eps, sigma=sigma,
        lower_bound=lower_bound, upper_bound=upper_bound)
    test_set = HeatmapLoader(
        test_images, test_labels, patch_shape,
        raw_transform=raw_transform, transform=transform, eps=eps, sigma=sigma,
        lower_bound=lower_bound, upper_bound=upper_bound
    )

    # put into DataLoader
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers)

    train_dataloader.shuffle = True
    val_dataloader.shuffle = True
    test_dataloader.shuffle = True

    return train_dataloader, val_dataloader, test_dataloader
