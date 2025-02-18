import os
import csv
import argparse
from tqdm import tqdm

from utils.evaluation.evaluation_metrics import metric_coords
from data_processing.create_heatmap import parse_json_files

def evaluate(pred_coords, label_path, model_name, input_name):
    json_files = [os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith('.json')]
    label_coords, protein_types = parse_json_files(json_files)

    precision, recall, f1, dev_percentage, sMAPE, mae = metric_coords(label_coords, pred_coords)

    results_folder = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_folder, exist_ok=True)
    csv_file = os.path.join(results_folder, f"evaluation_{model_name}.csv")

    # Check if the file exists to determine whether to write the header
    write_header = not os.path.exists(csv_file)

    # Save the metrics to the CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["input_name", "precision", "recall", "f1", "dev_percentage", "sMAPE", "mae"])
        writer.writerow([input_name, precision, recall, f1, dev_percentage, sMAPE, mae])

    print(f"Metrics saved to {csv_file}")

def process_folder(args):
    # Get all prediction files
    input_files = [os.path.join(args.pred_coords, name) for name in os.listdir(args.pred_coords)
                   if name.endswith("_protein_detections.json") and os.path.isfile(os.path.join(args.pred_coords, name))]

    # Get all label subfolders
    label_subfolders = {name: os.path.join(args.label_path, name) for name in os.listdir(args.label_path)
                        if os.path.isdir(os.path.join(args.label_path, name))}

    pbar = tqdm(input_files, desc="Evaluate files")

    for pred_coords in pbar:
        # Extract input_name from the filename
        input_name = os.path.basename(pred_coords).replace("_protein_detections.json", "")

        # Match with label subfolder
        label_folder = label_subfolders.get(input_name)
        label_folder = os.path.join(label_folder,"Picks")

        if label_folder:
            evaluate(pred_coords, label_folder, args.model_name, input_name)
        else:
            print(f"Warning: No matching label folder found for {pred_coords}")


def main():
    parser = argparse.ArgumentParser(description="Segment vesicles in EM tomograms.")
    parser.add_argument(
        "--pred_coords", "-p", required=True,
        help="The filepath to the json file of the protein location predictions."
    )
    parser.add_argument(
        "--label_path", "-l", required=True,
        help="The filepath to directory where the picked proteins are stored in a json file (gt)."
    )
    parser.add_argument(
        "--model_name", "-m", required=True, help="Name of the model."
    )

    args = parser.parse_args()

    if os.path.isfile(args.pred_coords) and args.pred_coords.endswith(".json"):
        # Extract input_name from the filename
        input_name = os.path.basename(args.pred_coords).replace("_protein_detections.json", "")
        evaluate(args.pred_coords, args.label_path, args.model_name, input_name)
    elif os.path.isdir(args.pred_coords) and any(f.endswith(".json") for f in os.listdir(args.pred_coords)):
        process_folder(args)
    else:
        print("Invalid input")

    print("Finished segmenting!")

if __name__ == "__main__":
    main()
