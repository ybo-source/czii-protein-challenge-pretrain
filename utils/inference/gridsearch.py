import pandas
from tqdm import tqdm
import zarr
import json
from skimage.feature import blob_log
from ..evaluation.evaluation_metrics import metric_coords, get_threshold_from_gridsearch 
from utils import get_prediction_torch_em, parse_tiling

#TODO Do I want to make this more flexible??
TRAIN_ROOT = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/train/static/"
LABEL_ROOT = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/train/overlay/"


def get_volume(input_path):
    zarr_file = zarr.open(os.path.join(input_path, "VoxelSpacing10.000", "denoised.zarr", "0"), mode='r')
    input_volume = zarr_file[:]

    return input_volume

def get_full_image_path(json_val_path, val_path):
    file_name = os.path.basename(json_val_path)
    # Remove the prefix "split-" and the suffix ".json"
    experiment_name = file_name[len("split-"):-len(".json")]

    image_path = os.path.join(TRAIN_ROOT, experiment_name,val_path)

    return image_path
    
def get_full_label_path(json_val_path, val_path):
    file_name = os.path.basename(json_val_path)
    # Remove the prefix "split-" and the suffix ".json"
    experiment_name = file_name[len("split-"):-len(".json")]

    label_path = os.path.join(LABEL_ROOT, experiment_name, val_path)

    return label_path

def gridsearch(json_val_path, model_path):
    print("starting grid search")

    data = []

    # Load JSON from the file
    with open(json_val_path, "r") as file:
        data = json.load(file)

    # Extract the 'val' list
    val_list = data["val"]

    for val_path in val_list:

        image_path = get_full_image_path(json_val_path, val_path)
        label_path = get_full_label_path(json_val_path, val_path)
        
        tiling = parse_tiling(tile_shape=None, halo=None) #TODO implement tiling and halo choices
        print(f"using tiling {tiling} for gridsearch")

        input_volume = get_volume(image_path)
        pred = get_prediction_torch_em(input_volume=input_volume, tiling=tiling, model_path=model_path, verbose=True)
    
        label_coords = #TODO function to get the label coordinates from label_path


        for thresh in threshes:
            #smalles protein structure: "beta-amylase": 33.27
            #bigges protein structure: "ribosome": 109.02,
            #0.3 is the factor to match the PDB size to the experimental data size
            adj_factor=0.3 #TODO implement this as an argument, also when creating heatmap                
            pred_coords = blob_log(pred, min_sigma=33.27*adj_factor, max_sigma=109.02*adj_factor, threshold_abs=thresh) 


            _, _, f1, _, _, _ = metric_coords(label_coords, pred_coords) #TODO
            data.append([f1, thresh])

    df = pandas.DataFrame(data=data, columns=["f1", "Threshold"])
    thresh = get_threshold_from_gridsearch(df, threshes)#TODO

    return thresh