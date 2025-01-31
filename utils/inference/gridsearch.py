import pandas
from tqdm import tqdm
from skimage.feature import blob_log
from ..evaluation.evaluation_metrics import metric_coords, get_distance_threshold_from_gridsearch 
    

def gridsearch():
    data = []
    for i in tqdm(range(len(idx))):
        image_path = grid_search_image_paths[i]
        label_path = grid_search_label_paths[i]
        
        model, device = load_model(config, in_channels)
        preds = predict_image(image_path, model, tuple(config["patch_shape"]), device=device)
        label_coords = get_center_coordinates(label_path)


        for thresh in threshes:
            #smalles protein structure: "beta-amylase": 33.27
            #bigges protein structure: "ribosome": 109.02,
            #0.3 is the factor to match the PDB size to the experimental data size
            adj_factor=0.3 #TODO implement this as an argument, also when creating heatmap                
            pred_coords = blob_log(preds, min_sigma=33.27*adj_factor, max_sigma=109.02*adj_factor, threshold_abs=thresh) 


            _, _, f1, _, _, _ = metric_coords(label_coords, pred_coords)
            data.append([f1, dist, thresh])

    df = pandas.DataFrame(data=data, columns=["f1", "Distance", "Threshold"])
    dist, thresh = get_distance_threshold_from_gridsearch(df, distances, threshes)
    return dist, thresh, grid_search_numbers