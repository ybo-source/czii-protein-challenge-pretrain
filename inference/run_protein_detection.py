import argparse
import os
import zarr
from tqdm import tqdm

from utils.prediction import get_prediction_torch_em
from utils.protein_detection import protein_detection

def get_volume(input_path):
    zarr_file = zarr.open(os.path.join(input_path, "VoxelSpacing10.000", "denoised.zarr", "0"), mode='r')
    input_volume = zarr_file[:]

    return input_volume

def run_protein_detection(input_path, output_path, model_path):
    data = get_volume(input_path)

    pred = get_prediction_torch_em()#TODO
    detection = #TODO protein detection

    #TODO saving


def process_folder(args):
    input_files = []
    input_files = [os.path.join(args.input_path, name) for name in os.listdir(args.input_path)
                   if os.path.isdir(os.path.join(args.input_path, name))]

    pbar = tqdm(input_files, desc="Run protein detection")
    for input_path in pbar:

        run_protein_detection(
            input_path, args.output_path, args.model_path
        )

def main():
    parser = argparse.ArgumentParser(description="Segment vesicles in EM tomograms.")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to the mrc file or the directory containing the tomogram data."
    )
    parser.add_argument(
        "--file", "-f", action="store_true",
        help="Input path is a single file. The protein detection only needs to run once. By default, multiple files are expected."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmentations will be saved."
    )
    parser.add_argument(
        "--model_path", "-m", required=True, help="The filepath to the vesicle model."
    )
    args = parser.parse_args()

    file = args.file
    
    if file:
        run_protein_detection(args.input_path, args.output_path, args.model_path)
    else:
        process_folder(args)
        

    print("Finished segmenting!")

if __name__ == "__main__":
    main()