import numpy as np
from skimage.feature import blob_log

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
    detections = []

    data_path=#TODO pass the val data paths
    threshold = gridsearch(data_path, model) 
    #smalles protein structure: "beta-amylase": 33.27
    #bigges protein structure: "ribosome": 109.02,
    #0.3 is the factor to match the PDB size to the experimental data size
    adj_factor=0.3 #TODO implement this as an argument, also when creating heatmap
    pred_coords = blob_log(preds, min_sigma=33.27*adj_factor, max_sigma=109.02*adj_factor, threshold_abs=threshold) 


    detections.append({
        'coordinates': coordinates,
        'size': size
    })

    return detections