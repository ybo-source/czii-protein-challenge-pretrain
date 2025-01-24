import numpy as np
from scipy.ndimage import gaussian_laplace, label, find_objects, center_of_mass
from scipy.ndimage.measurements import variance

def protein_detection(heatmap): #TODO do this properly
    """
    Detects local maxima and estimates sizes of Gaussians in a 3D heatmap.

    Parameters:
        heatmap (np.array): A 3D numpy array representing the heatmap.

    Returns:
        list of dict: A list of dictionaries with keys:
            - 'coordinates': Tuple of (z, y, x) for the local maxima
            - 'size': Estimated size of the Gaussian (sigma equivalent)
    """
    # Apply Laplacian of Gaussian (LoG) filter to enhance Gaussian-like structures
    log_filtered = -gaussian_laplace(heatmap, sigma=1)

    # Find local maxima
    labeled, num_features = label(log_filtered > np.mean(log_filtered))  # Binary threshold

    # Get bounding boxes and compute centers of mass
    regions = find_objects(labeled)
    detections = []

    for i, region in enumerate(regions):
        if region is None:
            continue

        # Extract subregion
        subregion = labeled[region]
        sub_heatmap = heatmap[region]

        # Mask specific to the current label
        mask = (subregion == (i + 1))

        # Compute center of mass as the coordinates of the local maximum
        com = center_of_mass(sub_heatmap, labels=mask, index=1)

        # Compute size: estimate the variance of the Gaussian
        size = np.sqrt(variance(sub_heatmap, labels=mask, index=1))

        # Adjust coordinates to global
        starts = [r.start for r in region]  # Dynamically handle dimensions
        com_global = tuple(com[i] + starts[i] for i in range(len(starts)))


        detections.append({
            'coordinates': com_global,
            'size': size
        })

    return detections