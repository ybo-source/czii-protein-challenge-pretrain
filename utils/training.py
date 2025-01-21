from typing import Optional, Tuple

import torch
import torch_em
from torch_em.model import AnisotropicUNet
import torch.nn as nn

from .data_loader import CreateDataLoader

'''#Julias code ... don't know yet if I need to chage it ...
def get_in_channels(image_path):
    # Load the first image to determine the number of channels
    image = np.asarray(imread(image_path))

    # Check if the first image is grayscale or RGB
    if len(image.shape) == 2:
        in_channels = 1
        # print(f"About to process grayscale images")
    elif image.shape[-1] == 4:
        in_channels = 3
        # print(f"About to process RGB images")
    else:
        in_channels = image.shape[-1]
        # print(f"About to process images of dimensions = {image.shape}")

    return in_channels'''


def get_3d_model(
    in_channels: int,
    out_channels: int,
    initial_features: int = 32,
    final_activation: str = "Sigmoid",
) -> torch.nn.Module:
    """Get the 3D U-Net model.

    Args:
        in_channels: The number of input channels of the network.
        out_channels: The number of output channels of the network.
        scale_factors: The downscaling factors for each level of the U-Net encoder.
        initial_features: The number of features in the first level of the U-Net.
            The number of features increases by a factor of two in each level.
        final_activation: The activation applied to the last output layer.
    Returns:
        The U-Net.
    """
    model = AnisotropicUNet(
        scale_factors=scale_factors,
        in_channels=in_channels,
        out_channels=out_channels,
        initial_features=initial_features,
        gain=2,
        final_activation=final_activation,
    )
    return model

def supervised_training(
    name: str,
    train_paths: Tuple[str],
    val_paths: Tuple[str],
    patch_shape: Tuple[int, int, int],
    save_root: Optional[str] = None,
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: int = int(1e5),
    check: bool = False,
    out_channels: int = 2,
    **loader_kwargs,
):
    """
    Run supervised training

    Args:
        name: The name for the checkpoint to be trained.
        train_paths: Filepaths to the files for the training data.
        val_paths: Filepaths to the files for the validation data.
        patch_shape: The patch shape used for a training example..
        save_root: Folder where the checkpoint will be saved.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The number of iterations to train for.
        check: Whether to check the training and validation loaders instead of running training.
        out_channels: The number of output channels of the UNet.
        loader_kwargs: Additional keyword arguments for the dataloader.
    """
    #TODO Im missing a few arguments here
    train_loader, val_loader, _ = CreateDataLoader(train_images, train_labels, val_images, val_labels, test_images, test_labels, 
                                                    raw_transform=raw_transform, transform=transform, 
                                                    patch_shape=patch_shape, num_workers=num_workers, batch_size=batch_size, eps=epsilon, sigma=sigma, lower_bound=lower_bound, upper_bound=upper_bound)


    if check:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, n_samples=4)
        check_loader(val_loader, n_samples=4)
        return
    
    #TODO train_images: also used in CreateDataLoader, ACTUALLY ONLY USING 1 CHANNEL?
    in_channels = 1 #get_in_channels(train_images[0])

    model = get_3d_model(in_channels=in_channels,out_channels=out_channels)

    loss, metric = None, None

    loss = nn.MSELoss(reduction="mean")
    metric = loss


    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
        save_root=save_root,
        loss=loss,
        metric=metric,
    )
    trainer.fit(n_iterations)
