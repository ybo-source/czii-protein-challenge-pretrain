import os
from glob import glob
import argparse

from utils.dataset_splits import get_paths

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

    #TODO call supervised training from utils.training



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--testset", action='store_false', help="Set to False if no testset should be created")
    args = parser.parse_args()
    train(args.testset)


if __name__ == "__main__":
    main()
