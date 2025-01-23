import os
from glob import glob
import argparse
import sys
sys.path.append("/user/muth9/u12095/czii-protein-challenge")

from utils.dataset_splits import get_paths
from utils.training import supervised_training

TRAIN_ROOT = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/train/static/" 
LABEL_ROOT = "/scratch-grete/projects/nim00007/cryo-et/challenge-data/train/overlay/" 
OUTPUT_ROOT = "/mnt/lustre-emmy-hdd/usr/u12095/cryo-et/czii_challenge/training"

def train(key, ignore_label = None, training_2D = False, testset = True, extension="zarr"):

    datasets = [
    "ExperimentRuns"
]
    model_name="protein_detection_czii_v1"

    output_path = os.path.join(OUTPUT_ROOT, model_name) 
    os.makedirs(output_path, exist_ok=True)

    train_paths, train_label_paths = get_paths("train", datasets=datasets, train_root=TRAIN_ROOT, output_root=output_path, testset=testset, label_root=LABEL_ROOT)
    val_paths, val_label_paths = get_paths("val", datasets=datasets, train_root=TRAIN_ROOT, output_root=output_path, testset=testset, label_root=LABEL_ROOT)

    if testset:
        test_paths, test_label_paths = get_paths("test", datasets=datasets, train_root=TRAIN_ROOT, output_root=output_path, testset=testset, label_root=LABEL_ROOT)
    else:
        test_paths, test_label_paths = None, None

    print("Start training with:")
    print(len(train_paths), "tomograms for training")
    print(len(val_paths), "tomograms for validation")

    patch_shape = [48, 256, 256]

    batch_size = 2 
    check = False

    #TODO do we want n_samples_train and n_samples_val in the supervised training?
    supervised_training(
        name=model_name,
        train_paths=train_paths,
        train_label_paths = train_label_paths,
        val_paths=val_paths,
        val_label_paths = val_label_paths,
        patch_shape=patch_shape, batch_size=batch_size,
        check=check,
        lr=1e-4,
        n_iterations=1e5,
        out_channels=1,
        augmentations=None,
        eps=1e-5, 
        sigma=None,  
        lower_bound=None,
        upper_bound=None,
        test_paths=test_paths,
        test_label_paths=test_label_paths,
        save_root="/mnt/lustre-emmy-hdd/usr/u12095/synapse_net/models_v2",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testset", action='store_false', help="Set to False if no testset should be created")
    args = parser.parse_args()
    train(args.testset)


if __name__ == "__main__":
    main()
