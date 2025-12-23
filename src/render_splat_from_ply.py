
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
    params = yaml.safe_load(open("params.yaml"))["render_splat_from_ply"]
    sh_degree = params["sh_degrees"]
    image_width = params["image_width"]
    image_height = params["image_height"]
    focal_length_x = params["focal_length_x"]
    focal_length_y = params["focal_length_y"]
    principal_point_x = params["principal_point_x"]
    principal_point_y = params["principal_point_y"]
    render_mode = "RGB+" + params["depth_render_mode"]
    eps2d = params["eps2d"]
    packed = params["packed"]
    sparse_grad = params["sparse_grad"]
    rasterize_mode = params["rasterize_mode"]
    center_distances = params["center_distance"]
    height_steps = params["height_steps"]
    yaw_steps = params["yaw_steps"]
    pitch_steps = params["pitch_steps"]
    pitch_extreme = params["pitch_extreme"]
    roll_steps = params["roll_steps"]
    max_background_depth = params["background_depth"]

    # Parameters
    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython render_splat_from_ply.py tree-splats-folder generated-images-folder generated-depth-folder\n")
        sys.exit(1)

    tree_splats_folder = sys.argv[1]
    generated_images_folder = sys.argv[2]
    generated_depth_folder = sys.argv[3]

    if not os.path.exists(generated_images_folder):
        os.makedirs(generated_images_folder)
    if not os.path.exists(generated_depth_folder):
        os.makedirs(generated_depth_folder)

    # Set random seed
    torch.manual_seed(42)
    device = "cuda"

    # Create initial viewmats around the origin
    y_min = tree_bboxes["tree_1"]["translation"][1] - tree_bboxes["tree_1"]["scale"][1]
    y_max = tree_bboxes["tree_1"]["translation"][1] + tree_bboxes["tree_1"]["scale"][1]
    y_components = np.linspace(y_min, y_max, height_steps+2)[1:-1]
    yaw_components = np.linspace(0, 2*np.pi, yaw_steps, endpoint=False)
    pitch_components = np.linspace(-pitch_extreme*np.pi, pitch_extreme*np.pi, pitch_steps+2)[1:-1]
    roll_components = np.linspace(-0.25*np.pi, 0.25*np.pi, roll_steps+2)[1:-1]

    global_viewmats = np.zeros((len(y_components)*len(center_distances)*len(yaw_components)*len(pitch_components)*len(roll_components), 4, 4))
    index = 0
    for y in y_components:
        for center_distance in center_distances:
            for yaw in yaw_components:
                x, z = 0, -center_distance # center_distance * -np.sin(yaw), center_distance * -np.cos(yaw)
                for pitch in pitch_components:
                    for roll in roll_components:
                        # Calculate rotation matrix from euler angles
                        height_yaw_matrix = create_transformation_matrix(
                            translate=[0.0, y, 0.0],
                            rotate=[0.0, yaw, 0.0],
                            scale=[1, 1, 1]
                        )
                        roll_pitch_matrix = create_transformation_matrix(
                            translate=[0.0, 0.0, 0.0],
                            rotate=[roll, 0.0, pitch],
                            scale=[1, 1, 1]
                        )
                        dist_matrix = create_transformation_matrix(
                            translate=[x, 0.0, z],
                            rotate=[0.0, 0.0, 0.0],
                            scale=[1, 1, 1]
                        )
                        global_viewmats[index] = height_yaw_matrix @ roll_pitch_matrix @ dist_matrix
                        index += 1

    global_viewmats = torch.tensor(global_viewmats).to(device).float()

    for tree in tree_bboxes.keys():
        # Make tree folders
        if not os.path.exists(os.path.join(generated_images_folder, tree)):
            os.makedirs(os.path.join(generated_images_folder, tree))
        if not os.path.exists(os.path.join(generated_depth_folder, tree)):
            os.makedirs(os.path.join(generated_depth_folder, tree))

        # Get the tree transformation matrix
        tree_tf_matrix = torch.tensor(
            create_transformation_matrix(
            translate=np.asarray(tree_bboxes[tree]["translation"])*np.asarray([1.0, -1.0, -1.0]),
            rotate=[0, np.pi/4, 0],
            scale=[1, 1, 1])
        ).to(device).float()


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

        # Transform the global viewmats to the tree's coordinate system
        all_viewmats = torch.matmul(tree_tf_matrix, global_viewmats).inverse()

        # Write to viewmats and intrinsics to transforms.json
        transforms_dict = {
            "w": image_width,
            "h": image_height,
            "fl_x": focal_length_x,
            "fl_y": focal_length_y,
            "cx": principal_point_x,
            "cy": principal_point_y,
        }
        frames = []
        depth_frames = []

        # Make batches of max 24 viewmats
        viewmat_batch_size = 24
        for viewmat_i in range(0, len(all_viewmats), viewmat_batch_size):
            try:
                viewmats = all_viewmats[viewmat_i:viewmat_i+viewmat_batch_size]
            except:
                viewmats = all_viewmats[viewmat_i:]

            # Intrinsic matrix
            intrinsic_matrix = np.array([
                [focal_length_x, 0, principal_point_x],
                [0, focal_length_y, principal_point_y],
                [0, 0, 1]
            ])
            Ks = torch.tensor(np.asarray([
                intrinsic_matrix for _ in range(len(viewmats))
                ])).to(device).float()


            C = len(viewmats)

            # batched render
            render_colors, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors, # [N, K, 3], with K = (sh_degree + 1) ** 2
                viewmats,  # [C, 4, 4]
                Ks,  # [C, 3, 3]
                image_width,
                image_height,
                sh_degree=sh_degree,
                render_mode=render_mode,
                eps2d=eps2d,
                packed=packed,
                sparse_grad=sparse_grad,
                rasterize_mode=rasterize_mode,
            )
            assert render_colors.shape == (C, image_height, image_width, 4)
            assert render_alphas.shape == (C, image_height, image_width, 1)

            render_rgbs = render_colors[..., 0:3]
            render_depths = render_colors[..., 3:4]
            render_depths = render_depths / max_background_depth
            
            

            # Save rgb and depth images
            for i in range(C):
                rgb = render_rgbs[i].cpu().numpy()
                depth = render_depths[i].cpu().numpy()
                
                # Save rgb image as jpg using OpenCV
                cv2.imwrite(os.path.join(generated_images_folder, tree, f"{viewmat_i+i:06d}.jpg"), cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                # Save depth image as 16 bit png using OpenCV
                cv2.imwrite(os.path.join(generated_depth_folder, tree, f"{viewmat_i+i:06d}.png"), (depth * 65535).astype(np.uint16))

                # Append info to frames
                frames.append({
                    "file_path": f"{viewmat_i+i:06d}.jpg",
                    "transform_matrix": np.linalg.inv(viewmats[i].cpu().numpy()).tolist()
                })
                depth_frames.append({
                    "file_path": f"{viewmat_i+i:06d}.png",
                    "transform_matrix": np.linalg.inv(viewmats[i].cpu().numpy()).tolist()
                })

                del rgb
                del depth
                torch.cuda.empty_cache()

            # Clear render from memory
            del render_rgbs
            del render_colors
            del render_alphas
            del render_depths
            del meta
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
