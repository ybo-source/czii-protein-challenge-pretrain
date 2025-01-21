import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian
import json
import argparse
import os
import zarr

def width_to_sigma(width, eps, lower_bound, upper_bound):
    # shrink needs to be between 0 and 1
    sigma = np.sqrt(-(width**2) / (2*np.log(eps)))
    #### bounding ####
    if lower_bound and upper_bound:
        if sigma < lower_bound:
            sigma = lower_bound
        elif sigma > upper_bound:
            sigma = upper_bound

    return int(sigma)

def create_gaussian_stamp(width, eps, lower_bound, upper_bound):
    """
    Creates a 3D Gaussian stamp (cube) with size width x width x width.
    If width is even, set width = width - 1.
    """
    if width % 2 == 0:
        width = width - 1
    
    sigma = width_to_sigma(width, eps, lower_bound, upper_bound)
    
    # Create a 3D matrix (stamp)
    stamp = np.zeros((width, width, width))
    center = width // 2
    
    # Set the center point to 1 (Gaussian peak)
    stamp[center, center, center] = 1

    # Apply 3D Gaussian filter to create the Gaussian distribution in 3D space
    #stamp = gaussian_filter(stamp, sigma=sigma)
    #Julia:
    stamp = gaussian(stamp, sigma=sigma, truncate=10.0, mode='constant')
    
    # Threshold the values based on epsilon and apply the scaling factor
    stamp[stamp < eps] = 0
    stamp = stamp * 8 * np.pi * sigma**3

    return stamp

def create_heatmap(image_shape, coordinates, protein_types, width_dict):
    """
    Create a 3D heatmap based on given coordinates and protein types.

    Parameters:
        image_shape (tuple): Shape of the 3D image (z, y, x).
        coordinates (list of tuples): List of (z, y, x) coordinates.
        protein_types (list): List of protein types corresponding to each coordinate.
        sigma_dict (dict): Dictionary mapping protein types to Gaussian widths (sigmas).

    Returns:
        np.ndarray: A 3D heatmap with Gaussian distributions at specified coordinates.
    """
    heatmap = np.zeros(image_shape)

    for coord, protein in zip(coordinates, protein_types):
        z, y, x = map(int, coord)  # Convert coordinates to integers for indexing
        width = width_dict.get(protein, 1.0)  # Default sigma is 1.0 if protein type is not found
        # Create a sparse 3D array with a single point at the coordinate
        point = np.zeros(image_shape)
        point[z, y, x] = 1

        '''# Apply Gaussian filter with constant boundary condition
        # mode='constant' ensures that values outside the image bounds are treated as zero
        gaussian = gaussian_filter(point, sigma=sigma, mode='constant')'''
        gaussian = create_gaussian_stamp(int(width * 0.3), eps=0.00001, lower_bound=None, upper_bound=None )
        '''# Update the heatmap using the maximum of the current heatmap and the new Gaussian (takes care of overlapping gaussians)
        heatmap = np.maximum(heatmap, gaussian)'''

        # Define the bounds for placement of the Gaussian in the heatmap
        z_min = max(0, z - gaussian.shape[0] // 2)
        z_max = min(image_shape[0], z + gaussian.shape[0] // 2 + 1)
        y_min = max(0, y - gaussian.shape[1] // 2)
        y_max = min(image_shape[1], y + gaussian.shape[1] // 2 + 1)
        x_min = max(0, x - gaussian.shape[2] // 2)
        x_max = min(image_shape[2], x + gaussian.shape[2] // 2 + 1)

        # Slice the heatmap and add the Gaussian stamp (handling boundaries)
        heatmap[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
            heatmap[z_min:z_max, y_min:y_max, x_min:x_max], gaussian[
                (z_min - (z - gaussian.shape[0] // 2)):(z_max - (z - gaussian.shape[0] // 2)),
                (y_min - (y - gaussian.shape[1] // 2)):(y_max - (y - gaussian.shape[1] // 2)),
                (x_min - (x - gaussian.shape[2] // 2)):(x_max - (x - gaussian.shape[2] // 2))
            ]
        )

    return heatmap
    
def parse_json_files(json_files):
    """
    Parse multiple JSON files to extract coordinates and protein names.

    Parameters:
        json_files (list of str): List of JSON file paths.

    Returns:
        list of tuples, list: Extracted coordinates and corresponding protein types.
    """
    coordinates = []
    protein_types = []

    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)

            points = data.get("points", [])
            for point in points:
                location = point.get("location", {})
                x = location.get("x")
                y = location.get("y")
                z = location.get("z")
                if x is not None and y is not None and z is not None:
                    coordinates.append((z / 10, y / 10, x / 10))  # Scale coordinates as needed
                    protein_types.append(data.get("pickable_object_name", "unknown"))

    return coordinates, protein_types

def create_width_dict():
    """
    Create a dictionary mapping protein names to widths.

    Returns:
        dict: Protein name to width mapping.
    """
    return {
        "apo-ferritin": 79.07,
        "beta-amylase": 33.27,
        "beta-galactosidase": 57.44,
        "ribosome": 109.02,
        "thyroglobulin": 76.47,
        "virus-like-particle": 79.07
    }

def process_tomogram(json_folder, image_shape):
    """
    Process a tomogram by creating a heatmap based on protein data from JSON files.

    Parameters:
        json_folder (str): Path to the folder containing JSON files.
        image_shape (tuple): Shape of the 3D image (z, y, x).

    Returns:
        np.ndarray: Generated 3D heatmap.
    """
    json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith('.json')]
    coordinates, protein_types = parse_json_files(json_files)
    width_dict = create_width_dict()
    return create_heatmap(image_shape, coordinates, protein_types, width_dict)

def get_tomo_shape(zarr_folder):
    """
    Get the shape of the tomogram stored under the '0' key in a Zarr folder.

    Parameters:
        zarr_folder (str): Path to the Zarr folder.

    Returns:
        tuple: Shape of the '0' tomogram array, or None if not found.
    """
    # Open the Zarr store
    zarr_store = zarr.open(zarr_folder, mode='r')

    # Iterate over groups and arrays in the Zarr store
    def traverse_zarr(group, parent_key=""):
        for key, item in group.items():
            current_key = f"{parent_key}/{key}" if parent_key else key
            if current_key != "0":
                continue

            if isinstance(item, zarr.core.Array):
                # Add array to Napari viewer
                data = np.array(item)
                return data.shape

            elif isinstance(item, zarr.hierarchy.Group):
                # Recursively traverse subgroup
                traverse_zarr(item, current_key)

    
    return traverse_zarr(zarr_store)

def main():
    """
    Main function to parse arguments and create a heatmap from a tomogram.
    """
    parser = argparse.ArgumentParser(description="Generate a 3D heatmap from a tomogram and JSON files.")
    parser.add_argument("--zarr_path", "-z", type=str, required=True, help="Path to zarr file.")
    parser.add_argument("--json_folder", "-j", type=str, required=True, help="Path to the folder containing the 6 corresponding JSON files.")
    parser.add_argument("--output_folder", "-o", type=str, required=True, help="Path to save the generated heatmap as a .npy file.")

    args = parser.parse_args()

    tomogram_shape = get_tomo_shape(args.zarr_path)
    if tomogram_shape is not None:
        json_folder = args.json_folder
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)

        output_file =  os.path.join(output_folder, "heatmap.npy")

        heatmap = process_tomogram(json_folder, tomogram_shape)

        # Save the heatmap to a file
        np.save(output_file, heatmap)
        print(f"Heatmap saved to {output_file}")

        '''import napari
        loaded_array = np.load(output_file)
        v = napari.Viewer()
        v.add_image(loaded_array, name= f"heatmap")
        napari.run()'''

    else:
        print("No 0 zarr was found")

if __name__ == "__main__":
    main()
