#!/usr/bin/env python3
import os
import numpy as np
import struct 
import collections
import sys

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name, width=width, height=height, params=np.array(params)
            )
        assert len(cameras) == num_cameras
    return cameras

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = image_name.decode("utf-8")
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images

def read_points3d_binary(path_to_model_file):
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8*track_length, format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs
            )
    return points3D

def images_dict_to_poses(images):
    """
    Convert a dictionary of images (returned by read_images_binary) into a
    NumPy array of shape (4,4,N) where each column is a 4x4 camera-to-world pose matrix.
    The conversion uses the qvec and tvec from each image:
      - First, compute the rotation matrix from qvec.
      - Then, compute the camera-to-world transformation as [R^T, -R^T*t; 0, 1].
    Images are sorted by their image id.
    """
    keys = sorted(images.keys())
    poses_list = []
    for k in keys:
        image = images[k]
        R = qvec2rotmat(image.qvec)
        t = image.tvec.reshape(3, 1)
        # Compute camera-to-world: 
        # Note: COLMAP's image pose is stored as the inverse of the world-to-camera pose.
        R_c2w = R.T
        t_c2w = -R.T @ t
        pose = np.eye(4)
        pose[:3, :3] = R_c2w
        pose[:3, 3:4] = t_c2w
        poses_list.append(pose)
    poses = np.stack(poses_list, axis=-1)  # shape (4,4,N)
    return poses

def save_poses(basedir, poses, pts3d, perm):
    """
    Given:
      - basedir: directory to save poses_bounds.npy.
      - poses: a NumPy array of shape (4,4,N) with camera-to-world poses.
      - pts3d: a dictionary of 3D points (from COLMAP).
      - perm: a permutation (list or array) that gives the order of cameras to process.
      
    For each camera, computes:
      - near bound (0.1th percentile) of depths of visible 3D points.
      - far bound (99.9th percentile).
      
    Concatenates each camera’s flattened pose with its near and far bounds, and saves
    the result as poses_bounds.npy.
    """
    pts_arr = []
    vis_arr = []
    # Build arrays from 3D points and visibility.
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind:
                print('ERROR: the correct camera pose for current point cannot be accessed')
                return
            cams[ind - 1] = 1
        vis_arr.append(cams)
    pts_arr = np.array(pts_arr)   # (num_points, 3)
    vis_arr = np.array(vis_arr)   # (num_points, num_cameras)
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)
    
    # Compute depth values: for each camera, depth = dot( (pt - t), view_dir )
    # Here, t is the camera translation and view_dir is the third column of the rotation matrix.
    # We compute this for every 3D point and every camera.
    zvals = np.sum( -(pts_arr[:, np.newaxis, :].transpose([2, 0, 1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0 )
    
    valid_z = zvals[vis_arr == 1]
    print('Depth stats: min =', valid_z.min(), 'max =', valid_z.max(), 'mean =', valid_z.mean())
    
    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis == 1]
        close_depth = np.percentile(zs, 0.1)
        inf_depth   = np.percentile(zs, 99.9)
        pose_flat = poses[..., i].ravel()
        save_arr.append(np.concatenate([pose_flat, np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)
    
    save_dir = os.path.join(basedir, 'poses_bounds.npy')
    # print(save_dir)
    np.save(save_dir, save_arr)
    print("Saved poses_bounds.npy with shape:", save_arr.shape)

def make_poses_files(basedir, images, pts3d):
    # Convert the images dictionary into a (4,4,N) pose array.
    poses = images_dict_to_poses(images)

    # Create a permutation for processing cameras in sorted order.
    perm = sorted(range(poses.shape[-1]))

    save_poses(basedir, poses, pts3d, perm)

def convert_binary_image_names(images):

    print(type(images))
    # print(type(pts3d))
    #print(images.keys())
    # print(pts3d.keys())

    print(images[1].name)
    # print(pts3d[80488])
    for image_id, image in images.items():
        
        if image.name.startswith("base_"):
            print(f"{image.name} starts with base")
            prefix = "base"
        elif image.name.startswith("added"):
            print(f"{image.name} starts with added")
            prefix = "added"
        else:
            print("PROBLEM")
            prefix = "ERORR"
            continue
        new_name = os.path.join(prefix, image.name)
        print(f"{new_name=}")
        images[image_id] = image._replace(name=new_name)

    return images

def save_new_images(out_file, images):
    """
    Write the images dictionary to a binary file in the COLMAP format.
    This includes:
      - Writing the number of images (unsigned long long).
      - For each image:
          - Header record: image_id, qvec (4 doubles), tvec (3 doubles), camera_id.
          - Null-terminated image name.
          - Number of 2D points (unsigned long long).
          - For each 2D observation: x (double), y (double), and point3D_id (long long).
    """
    with open(out_file, "wb") as fid:
        # Write number of images.
        num_images = len(images)
        fid.write(struct.pack("<Q", num_images))
        # Write each image in sorted order by image id.
        for image_id in sorted(images.keys()):
            image = images[image_id]
            # Write header record.
            fid.write(struct.pack("<idddddddi",
                image.id,
                float(image.qvec[0]),
                float(image.qvec[1]),
                float(image.qvec[2]),
                float(image.qvec[3]),
                float(image.tvec[0]),
                float(image.tvec[1]),
                float(image.tvec[2]),
                image.camera_id,
            ))
            # Write the image name as a null-terminated string.
            fid.write(image.name.encode("utf-8") + b"\x00")
            # Write the number of 2D points.
            num_points2D = image.xys.shape[0]
            fid.write(struct.pack("<Q", num_points2D))
            # Write each 2D observation.
            for i in range(num_points2D):
                x = float(image.xys[i, 0])
                y = float(image.xys[i, 1])
                point3D_id = int(image.point3D_ids[i])
                fid.write(struct.pack("<ddq", x, y, point3D_id))
    print(f"Updated binary file written to {out_file}")


if __name__ == "__main__":
    # basedir = "/home/nicag/clnerf_thesis/data/counter_sm/counter_sm_merged/sparse/1/"
    basedir = "/mnt/nas_drive/nangelidis/drz"
    binaries_dir = os.path.join(basedir, "sparse/1/")
    # binaries_dir = "/mnt/nas_drive/nangelidis/breville"

    images = read_images_binary(os.path.join(binaries_dir, "images.bin"))
    points3d = read_points3d_binary(os.path.join(binaries_dir, "points3D.bin"))
    # cameras = read_cameras_binary(os.path.join(binaries_dir, "cameras.bin"))


    # images = read_images_binary(os.path.join(binaries_dir, "sparse/0/images_converted.bin"))

    # Functions for converting filenames from 
    # base_frame_00001.png to base/base_frame_00001.png
    # added_frame_00001.png to added/added_frame_00001.png respectively 
    # new_images = convert_binary_image_names(images)
    # new_images_filename = os.path.join(binaries_dir, "sparse/0/images_converted.bin")
    # save_new_images(new_images_filename, new_images)

    # DISPLAYING images
    # print_names = []
    for image_id, data in images.items():
        # print(data.name.split("/")[0])
        print(f"name: {data.name}, id: {data.id}")
        # if data.name.split("/")[0] not in print_names:
        #     print(data)
        #     print_names.append(data.name.split("/")[0])
    
    print(len(images))
    # print(len(points3d))
    # print(len(cameras))
    
    # Compute and save poses_bounds.npy
    make_poses_files(basedir, images, points3d)


    # for cam_id, data in cameras.items():
        # print(data)
    
    # print(type(cameras))
    # print(len(cameras))
    # print(cameras[1])
    # print(cameras[1].params[0])