import os
import zarr
import numpy as np
from typing import Union, Sequence
from numpy.typing import ArrayLike

def load_image(image_path):
    """@private
    """
    #TODO make flexible for other data types
    return zarr.open(os.path.join(image_path, "VoxelSpacing10.000", "denoised.zarr", "0"), mode='r')

    
def load_data(
    path: Union[str, Sequence[str]],
) -> ArrayLike:
    """Load data from a file or multiple files.

    Args:
        path: The file path or paths to the data.

    Returns:
        The loaded data.
    """
    #TODO can expand to also read in a key (check torch-em/torch_em/util/image.py)

    have_single_file = isinstance(path, str)

    if have_single_file:
        return load_image(path)
    else:
        return np.stack([load_image(p) for p in path])
