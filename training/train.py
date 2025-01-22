import os
from glob import glob
import argparse

from utils.dataset_splits import get_paths
from utils.training import supervised_training

TRAIN_ROOT = ""
LABEL_ROOT = ""
OUTPUT_ROOT = ""

def train(key, ignore_label = None, training_2D = False, testset = True, extension="h5"):

    datasets = [
    ""
]
    train_paths, train_label_paths = get_paths("train", datasets=datasets, train_root=TRAIN_ROOT, output_root=OUTPUT_ROOT, testset=testset, extension, label_root=LABEL_ROOT)
    val_paths, val_label_paths = get_paths("val", datasets=datasets, train_root=TRAIN_ROOT, output_root=OUTPUT_ROOT, testset=testset, extension, label_root=LABEL_ROOT)

    if testset:
        test_paths, test_label_paths = get_paths("test", datasets=datasets, train_root=TRAIN_ROOT, output_root=OUTPUT_ROOT, testset=testset, extension, label_root=LABEL_ROOT)
    else:
        test_paths, test_label_paths = None, None

    print("Start training with:")
    print(len(train_paths), "tomograms for training")
    print(len(val_paths), "tomograms for validation")

    patch_shape = [, , ] #TODO
    model_name="" #TODO

    batch_size = 4 #TODO
    check = False

    #TODO do we want n_samples_train and n_samples_val in the supervised training?
    supervised_training(
        name=model_name,
        train_paths=train_paths,
        train_label_paths = train_label_paths,
        val_paths=val_paths,
        val_label_paths = val_label_paths,
        test_paths=test_paths,
        test_label_paths=test_label_paths,
        patch_shape=patch_shape, batch_size=batch_size,
        check=check,
        save_root="/mnt/lustre-emmy-hdd/usr/u12095/synapse_net/models_v2",
        lr=,#TODO
        n_iterations=,#TODO
        out_channels=,#TODO 
        augmentations=,#TODO 
        eps=, #TODO 
        sigma=, #TODO 
        lower_bound=, #TODO
        upper_bound=,#TODO 
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testset", action='store_false', help="Set to False if no testset should be created")
    args = parser.parse_args()
    train(args.testset)


if __name__ == "__main__":
    main()
