
import numpy as np
import torch
from gsplat.rendering import rasterization
import plyfile as plyf
import yaml
import sys
import os
import cv2
import json

# Function to create a transformation matrix
def create_transformation_matrix(translate, rotate, scale):
    tx, ty, tz = translate
    sx, sy, sz = scale
    rx, ry, rz = rotate

    # Translation matrix
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]

    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx

    # Combine rotation and scale into a single matrix
    T[:3, :3] = R @ np.diag([sx, sy, sz])

    # Final transformation matrix
    return T

def main():
    # Load the tree bounding boxes
    tree_bboxes = yaml.safe_load(open("params.yaml"))["tree_bounding_boxes"]

    # Load the parameters
    params = yaml.safe_load(open("params.yaml"))["render_real_depth"]
    sh_degree = params["sh_degrees"]
    render_mode = params["depth_render_mode"]
    eps2d = params["eps2d"]
    packed = params["packed"]
    sparse_grad = params["sparse_grad"]
    rasterize_mode = params["rasterize_mode"]
    max_background_depth = params["background_depth"]
    colored_fraction = params["colored_fraction"]
    render_diff_threshold = params["render_diff_threshold"]

    # Parameters
    if len(sys.argv) != 6:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython render_real_depth.py all-trees-splat-folder tree-splats-folder images-folder masked-images-folder real-depth-folder\n")
        sys.exit(1)

    all_trees_splat_folder = sys.argv[1]
    tree_splats_folder = sys.argv[2]
    input_image_folder = sys.argv[3]
    generated_images_folder = sys.argv[4]
    generated_depth_folder = sys.argv[5]

    if not os.path.exists(generated_images_folder):
        os.makedirs(generated_images_folder)
    if not os.path.exists(generated_depth_folder):
        os.makedirs(generated_depth_folder)

    # Set random seed
    torch.manual_seed(42)
    device = "cuda"

    #### Load the all trees splat file
    all_trees_gs_ply = plyf.PlyData.read(os.path.join(all_trees_splat_folder, "trees.ply"))

    # Convert to tensors
    all_trees_means = torch.tensor(np.transpose(np.asarray([
        all_trees_gs_ply["vertex"]["x"], 
        all_trees_gs_ply["vertex"]["y"],
        all_trees_gs_ply["vertex"]["z"],
        ]), (1, 0))).to(device).float()
    all_trees_quats = torch.tensor(np.transpose(np.asarray([
        all_trees_gs_ply["vertex"]["rot_0"], 
        all_trees_gs_ply["vertex"]["rot_1"],
        all_trees_gs_ply["vertex"]["rot_2"],
        all_trees_gs_ply["vertex"]["rot_3"],
        ]), (1, 0))).to(device).float()
    all_trees_scales = torch.tensor(np.transpose(np.exp(np.asarray([
        all_trees_gs_ply["vertex"]["scale_0"],
        all_trees_gs_ply["vertex"]["scale_1"],
        all_trees_gs_ply["vertex"]["scale_2"],
        ])), (1, 0))).to(device).float()
    all_trees_opacities = torch.tensor(
        all_trees_gs_ply["vertex"]["opacity"]
        ).to(device).float()
    all_trees_opacities = (1 / (1 + torch.exp(-all_trees_opacities))).type(torch.float)
    # Spherical harmonics are represented with first 15 R values, then 15 G values, then 15 B values
    all_trees_colors = torch.tensor(np.transpose(np.asarray([
        [
            all_trees_gs_ply["vertex"]["f_dc_0"],
            all_trees_gs_ply["vertex"]["f_dc_1"],
            all_trees_gs_ply["vertex"]["f_dc_2"],
        ],[
            all_trees_gs_ply["vertex"]["f_rest_0"],
            all_trees_gs_ply["vertex"]["f_rest_15"],
            all_trees_gs_ply["vertex"]["f_rest_30"],
        ],[
            all_trees_gs_ply["vertex"]["f_rest_1"],
            all_trees_gs_ply["vertex"]["f_rest_16"],
            all_trees_gs_ply["vertex"]["f_rest_31"],
        ],[
            all_trees_gs_ply["vertex"]["f_rest_2"],
            all_trees_gs_ply["vertex"]["f_rest_17"],
            all_trees_gs_ply["vertex"]["f_rest_32"],
        ],[
            all_trees_gs_ply["vertex"]["f_rest_3"],
            all_trees_gs_ply["vertex"]["f_rest_18"],
            all_trees_gs_ply["vertex"]["f_rest_33"],
        ],[
            all_trees_gs_ply["vertex"]["f_rest_4"],
            all_trees_gs_ply["vertex"]["f_rest_19"],
            all_trees_gs_ply["vertex"]["f_rest_34"],
        ],[
            all_trees_gs_ply["vertex"]["f_rest_5"],
            all_trees_gs_ply["vertex"]["f_rest_20"],
            all_trees_gs_ply["vertex"]["f_rest_35"],
        ],[
            all_trees_gs_ply["vertex"]["f_rest_6"],
            all_trees_gs_ply["vertex"]["f_rest_21"],
            all_trees_gs_ply["vertex"]["f_rest_36"],
        ],[
            all_trees_gs_ply["vertex"]["f_rest_7"],
            all_trees_gs_ply["vertex"]["f_rest_22"],
            all_trees_gs_ply["vertex"]["f_rest_37"],
        ],[
            all_trees_gs_ply["vertex"]["f_rest_8"],
            all_trees_gs_ply["vertex"]["f_rest_23"],
            all_trees_gs_ply["vertex"]["f_rest_38"],
        ],[
            all_trees_gs_ply["vertex"]["f_rest_9"],
            all_trees_gs_ply["vertex"]["f_rest_24"],
            all_trees_gs_ply["vertex"]["f_rest_39"],
        ],[
            all_trees_gs_ply["vertex"]["f_rest_10"],
            all_trees_gs_ply["vertex"]["f_rest_25"],
            all_trees_gs_ply["vertex"]["f_rest_40"],
        ],[
            all_trees_gs_ply["vertex"]["f_rest_11"],
            all_trees_gs_ply["vertex"]["f_rest_26"],
            all_trees_gs_ply["vertex"]["f_rest_41"],
        ],[
            all_trees_gs_ply["vertex"]["f_rest_12"],
            all_trees_gs_ply["vertex"]["f_rest_27"],
            all_trees_gs_ply["vertex"]["f_rest_42"],
        ],[
            all_trees_gs_ply["vertex"]["f_rest_13"],
            all_trees_gs_ply["vertex"]["f_rest_28"],
            all_trees_gs_ply["vertex"]["f_rest_43"],
        ],[
            all_trees_gs_ply["vertex"]["f_rest_14"],
            all_trees_gs_ply["vertex"]["f_rest_29"],
            all_trees_gs_ply["vertex"]["f_rest_44"],
        ]]), (2, 0, 1))
        ).to(device).float()

    # Load the transforms file
    with open(os.path.join(input_image_folder, "transforms.json"), 'r') as f:
        transforms_data = json.load(f)

    
    # Camera intrinsic parameters from transforms.json
    w, h = transforms_data["w"], transforms_data["h"]


    # Get applied transform
    applied_transform = torch.eye(4).to(device).float()
    applied_transform[:3] = torch.tensor(transforms_data["applied_transform"]).to(device).float()

    # Load the viewmats
    all_viewmats = torch.zeros(len(transforms_data["frames"]), 4, 4).to(device)
    all_Ks = torch.zeros(len(transforms_data["frames"]), 3, 3).to(device)
    frame_nums = torch.zeros(len(transforms_data["frames"])).to(device).int()
    for frame_idx, frame in enumerate(transforms_data["frames"]):
            
        image_transform = torch.tensor(frame["transform_matrix"]).to(device).float()

        # Transform with applied transform
        image_transform_rotated = torch.matmul(torch.tensor(
                [
                    [ 1.0,  0.0,  0.0,  0.0],
                    [ 0.0, -1.0,  0.0,  0.0],
                    [ 0.0,  0.0, -1.0,  0.0],
                    [ 0.0,  0.0,  0.0,  1.0]
                ]
            ).to(device).float(),
            torch.matmul(applied_transform,torch.matmul(image_transform, torch.tensor(
                [
                    [ 1.0,  0.0,  0.0,  0.0],
                    [ 0.0, -1.0,  0.0,  0.0],
                    [ 0.0,  0.0, -1.0,  0.0],
                    [ 0.0,  0.0,  0.0,  1.0]
                ]
            ).to(device).float())))

        all_viewmats[frame_idx] = image_transform_rotated.inverse()

        # Get the intrinsic matrix
        all_Ks[frame_idx] = torch.tensor(frame["intrinsic_matrix"]).to(device).float()

        # Get the frame number
        frame_nums[frame_idx] = int(frame["file_path"].split(".")[0].split("_")[1])

    for tree in tree_bboxes.keys():
        # Make tree folders
        if not os.path.exists(os.path.join(generated_images_folder, tree)):
            os.makedirs(os.path.join(generated_images_folder, tree))
        if not os.path.exists(os.path.join(generated_depth_folder, tree)):
            os.makedirs(os.path.join(generated_depth_folder, tree))

        # Load the ply file
        gs_ply = plyf.PlyData.read(os.path.join(tree_splats_folder, tree + ".ply"))

        # Convert to tensors
        means = torch.tensor(np.transpose(np.asarray([
            gs_ply["vertex"]["x"], 
            gs_ply["vertex"]["y"],
            gs_ply["vertex"]["z"],
            ]), (1, 0))).to(device).float()
        quats = torch.tensor(np.transpose(np.asarray([
            gs_ply["vertex"]["rot_0"], 
            gs_ply["vertex"]["rot_1"],
            gs_ply["vertex"]["rot_2"],
            gs_ply["vertex"]["rot_3"],
            ]), (1, 0))).to(device).float()
        scales = torch.tensor(np.transpose(np.exp(np.asarray([
            gs_ply["vertex"]["scale_0"],
            gs_ply["vertex"]["scale_1"],
            gs_ply["vertex"]["scale_2"],
            ])), (1, 0))).to(device).float()
        opacities = torch.tensor(
            gs_ply["vertex"]["opacity"]
            ).to(device).float()
        opacities = (1 / (1 + torch.exp(-opacities))).type(torch.float)
        # Spherical harmonics are represented with first 15 R values, then 15 G values, then 15 B values
        colors = torch.tensor(np.transpose(np.asarray([
            [
                gs_ply["vertex"]["f_dc_0"],
                gs_ply["vertex"]["f_dc_1"],
                gs_ply["vertex"]["f_dc_2"],
            ],[
                gs_ply["vertex"]["f_rest_0"],
                gs_ply["vertex"]["f_rest_15"],
                gs_ply["vertex"]["f_rest_30"],
            ],[
                gs_ply["vertex"]["f_rest_1"],
                gs_ply["vertex"]["f_rest_16"],
                gs_ply["vertex"]["f_rest_31"],
            ],[
                gs_ply["vertex"]["f_rest_2"],
                gs_ply["vertex"]["f_rest_17"],
                gs_ply["vertex"]["f_rest_32"],
            ],[
                gs_ply["vertex"]["f_rest_3"],
                gs_ply["vertex"]["f_rest_18"],
                gs_ply["vertex"]["f_rest_33"],
            ],[
                gs_ply["vertex"]["f_rest_4"],
                gs_ply["vertex"]["f_rest_19"],
                gs_ply["vertex"]["f_rest_34"],
            ],[
                gs_ply["vertex"]["f_rest_5"],
                gs_ply["vertex"]["f_rest_20"],
                gs_ply["vertex"]["f_rest_35"],
            ],[
                gs_ply["vertex"]["f_rest_6"],
                gs_ply["vertex"]["f_rest_21"],
                gs_ply["vertex"]["f_rest_36"],
            ],[
                gs_ply["vertex"]["f_rest_7"],
                gs_ply["vertex"]["f_rest_22"],
                gs_ply["vertex"]["f_rest_37"],
            ],[
                gs_ply["vertex"]["f_rest_8"],
                gs_ply["vertex"]["f_rest_23"],
                gs_ply["vertex"]["f_rest_38"],
            ],[
                gs_ply["vertex"]["f_rest_9"],
                gs_ply["vertex"]["f_rest_24"],
                gs_ply["vertex"]["f_rest_39"],
            ],[
                gs_ply["vertex"]["f_rest_10"],
                gs_ply["vertex"]["f_rest_25"],
                gs_ply["vertex"]["f_rest_40"],
            ],[
                gs_ply["vertex"]["f_rest_11"],
                gs_ply["vertex"]["f_rest_26"],
                gs_ply["vertex"]["f_rest_41"],
            ],[
                gs_ply["vertex"]["f_rest_12"],
                gs_ply["vertex"]["f_rest_27"],
                gs_ply["vertex"]["f_rest_42"],
            ],[
                gs_ply["vertex"]["f_rest_13"],
                gs_ply["vertex"]["f_rest_28"],
                gs_ply["vertex"]["f_rest_43"],
            ],[
                gs_ply["vertex"]["f_rest_14"],
                gs_ply["vertex"]["f_rest_29"],
                gs_ply["vertex"]["f_rest_44"],
            ]]), (2, 0, 1))
            ).to(device).float()

        # Write to viewmats and intrinsics to transforms.json
        transforms_dict = {
            "w": w,
            "h": h,
        }
        frames = []
        depth_frames = []

        # Make batches of max 24 viewmats
        viewmat_batch_size = 24
        for viewmat_i in range(0, len(all_viewmats), viewmat_batch_size):
            try:
                viewmats = all_viewmats[viewmat_i:viewmat_i+viewmat_batch_size]
                Ks = all_Ks[viewmat_i:viewmat_i+viewmat_batch_size]
            except:
                viewmats = all_viewmats[viewmat_i:]
                Ks = all_Ks[viewmat_i:]

            C = len(viewmats)

            # batched render
            render_depths, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors, # [N, K, 3], with K = (sh_degree + 1) ** 2
                viewmats,  # [C, 4, 4]
                Ks,  # [C, 3, 3]
                w,
                h,
                sh_degree=sh_degree,
                render_mode=render_mode,
                eps2d=eps2d,
                packed=packed,
                sparse_grad=sparse_grad,
                rasterize_mode=rasterize_mode,
            )
            del render_alphas
            del meta
            assert render_depths.shape == (C, h, w, 1)

            render_depths = render_depths / max_background_depth

            # Batched render of all trees splat
            render_depths_all_trees, render_alphas_all_trees, meta_all_trees = rasterization(
                all_trees_means,  # [N, 3]
                all_trees_quats,  # [N, 4]
                all_trees_scales,  # [N, 3]
                all_trees_opacities,  # [N]
                all_trees_colors, # [N, K, 3], with K = (sh_degree + 1) ** 2
                viewmats,  # [C, 4, 4]
                Ks,  # [C, 3, 3]
                w,
                h,
                sh_degree=sh_degree,
                render_mode=render_mode,
                eps2d=eps2d,
                packed=packed,
                sparse_grad=sparse_grad,
                rasterize_mode=rasterize_mode,
            )
            del render_alphas_all_trees
            del meta_all_trees
            assert render_depths_all_trees.shape == (C, h, w, 1)

            render_depths_all_trees = render_depths_all_trees / max_background_depth

            # Mask with false where all_trees_depths is closer
            render_depths = render_depths * (render_depths_all_trees > (render_depths - render_diff_threshold)).float()

            del render_depths_all_trees
            torch.cuda.empty_cache()

            # Load the original image
            original_rgbs = torch.zeros(C, h, w, 3).to(device)
            render_counter = 0
            for frame_idx in range(viewmat_i, viewmat_i+C):
                orig_img = cv2.imread(os.path.join(input_image_folder, transforms_data["frames"][frame_idx]["file_path"]))
                original_rgbs[render_counter] = torch.tensor(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)).to(device).float() / 255.0
                render_counter += 1

            render_rgbs = torch.zeros_like(original_rgbs)
            render_rgbs = original_rgbs * (render_depths != 0.0).float()
            
            # Save rgb and depth images
            for i in range(C):
                depth = render_depths[i].cpu().numpy()

                # Check how many non-black pixels are in the depth image
                count_non_black_pixels = np.sum(depth != 0.0)
                if count_non_black_pixels < colored_fraction * w * h:
                    del depth
                    torch.cuda.empty_cache()
                    continue

                rgb = render_rgbs[i].cpu().numpy()
                
                # Save rgb image as jpg using OpenCV
                cv2.imwrite(os.path.join(generated_images_folder, tree, f"{frame_nums[viewmat_i+i]:06d}.jpg"), cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                # Save depth image as 16 bit png using OpenCV
                cv2.imwrite(os.path.join(generated_depth_folder, tree, f"{frame_nums[viewmat_i+i]:06d}.png"), (depth * 65535).astype(np.uint16))

                # Append info to frames
                frames.append({
                    "file_path": f"{frame_nums[viewmat_i+i]:06d}.jpg",
                    "transform_matrix": np.linalg.inv(viewmats[i].cpu().numpy()).tolist(),
                    "intrinsic_matrix": Ks[i].cpu().numpy().tolist(),
                })
                depth_frames.append({
                    "file_path": f"{frame_nums[viewmat_i+i]:06d}.png",
                    "transform_matrix": np.linalg.inv(viewmats[i].cpu().numpy()).tolist(),
                    "intrinsic_matrix": Ks[i].cpu().numpy().tolist(),
                })

                del rgb
                del depth
                torch.cuda.empty_cache()

            # Clear render from memory
            del render_rgbs
            del render_depths
            torch.cuda.empty_cache()

         # Copy transforms_dict for depth_frames
        depth_transforms_dict = transforms_dict.copy()
        depth_transforms_dict["frames"] = depth_frames
        with open(os.path.join(generated_depth_folder, tree, "transforms.json"), "a") as f:
            json.dump(depth_transforms_dict, f, indent=4)

        # Save to transforms.json
        transforms_dict["frames"] = frames
        with open(os.path.join(generated_images_folder, tree, "transforms.json"), "a") as f:
            json.dump(transforms_dict, f, indent=4)


if __name__ == "__main__":
    main()
