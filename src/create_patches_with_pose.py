import os
import json
from PIL import Image
import numpy as np
import yaml
import sys
from multiprocessing.pool import ThreadPool

def patch_image(
        iterator_w_frame,
        input_image_dir, output_image_dir,
        patch_width, patch_height,
        focal_length_x_original, focal_length_y_original,
        cx, cy,
        x_zero_patches, y_zero_patches,
        num_patches_per_image, num_patches_y,
        bad_image_idxes,
):
    frame_idx, frame = iterator_w_frame
    image_name = frame['file_path']

    # Get image idx from the image name
    image_idx = int(image_name.split('_')[-1].split('.')[0])

    if image_idx in bad_image_idxes:
        return []

    # Load the image
    image_path = os.path.join(input_image_dir, image_name)
    image = Image.open(image_path)

    image_patch_transforms = []

    # Generate patches
    for x_idx, x in enumerate(x_zero_patches):
        for y_idx, y in enumerate(y_zero_patches):
            # Image index
            patched_idx = image_idx * num_patches_per_image + x_idx * num_patches_y + y_idx
            # Calculate patch boundaries
            patch_bounds = (
                x, 
                y, 
                x + patch_width, 
                y + patch_height,
            )

            # Create a patch
            patch = image.crop(patch_bounds)
            patch_filename = f"image_{patched_idx:05d}.jpg"
            patch.save(os.path.join(output_image_dir, patch_filename))

            # Make intrinsic matrix for the patch
            patch_intrinsic_matrix = np.array([
                [focal_length_x_original, 0, cx - x],
                [0, focal_length_y_original, cy - y],
                [0, 0, 1],
            ])
            
            # Append new transform entry
            new_entry = {
                'file_path': patch_filename,
                'transform_matrix': frame['transform_matrix'],
                'intrinsic_matrix': patch_intrinsic_matrix.tolist(),
            }
            image_patch_transforms.append(new_entry)
    
    return image_patch_transforms

def main():
    params = yaml.safe_load(open("params.yaml"))["create_patches_with_pose"]

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython create_patches_with_pose.py input_images_folder patched_images_folder\n")
        sys.exit(1)

    # Test data set split ratio
    patch_width = params["patch_width"]
    patch_height = params["patch_height"]
    bad_image_idxes = params["bad_image_idxes"]

    input_image_dir = sys.argv[1]
    output_image_dir = sys.argv[2]

    os.makedirs(output_image_dir, exist_ok=True)

    # Load the transforms file
    with open(os.path.join(input_image_dir, "transforms.json"), 'r') as f:
        transforms_data = json.load(f)

    width_original = transforms_data['w']
    height_original = transforms_data['h']
    focal_length_x_original = transforms_data['fl_x']
    focal_length_y_original = transforms_data['fl_y']
    cx, cy = transforms_data['cx'], transforms_data['cy']
    new_transforms = {
        "camera_model": transforms_data['camera_model'],
        "w": patch_width,
        "h": patch_height,
        "frames": [],
        "applied_transform": transforms_data['applied_transform'],
    }
    num_patches_x = width_original // patch_width + 1
    num_patches_y = height_original // patch_height + 1
    x_zero_patches = np.linspace(0, width_original-patch_width, num_patches_x).astype(int)
    y_zero_patches = np.linspace(0, height_original-patch_height, num_patches_y).astype(int)
    num_patches_per_image = num_patches_x * num_patches_y

    all_patched_frames = []

    n_cores = int(os.cpu_count()) - 2
    with ThreadPool(n_cores) as pool:
        all_patched_frames += pool.starmap(
            patch_image, 
            [
                (
                    iterator_w_frame,
                    input_image_dir, output_image_dir,
                    patch_width, patch_height,
                    focal_length_x_original, focal_length_y_original,
                    cx, cy,
                    x_zero_patches, y_zero_patches,
                    num_patches_per_image, num_patches_y,
                    bad_image_idxes,
                ) for iterator_w_frame in enumerate(transforms_data['frames'])
            ]
        )

    # Flatten the list
    all_patched_frames = [item for sublist in all_patched_frames for item in sublist]

    new_transforms['frames'] = all_patched_frames

    # Save updated transforms
    with open(os.path.join(output_image_dir, "transforms.json"), 'w') as f:
        json.dump(new_transforms, f, indent=4)


if __name__ == "__main__":
    main()