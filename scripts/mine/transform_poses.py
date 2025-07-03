import numpy as np
import ast

poses = []
names = []

poses_path = '/home/nangelidis/CLNeRF/results/NGPGv2/CLNerf/colmap_ngpa_CLNerf/drz_custom_pose_s8.0_lr1e-2_dima48_dimg16_r5_e1_b4096_d1.0_gpu1/rep/poses.txt'

with open(poses_path) as f:
    for line in f:
        name, vec = line.split('\t', 1)
        # ast.literal_eval will turn the string "[x, y, z, …]" into a Python list
        row = ast.literal_eval(vec)
        poses.append(row)
        names.append(name.replace('.jpg', ''))  # remove .jpg from the name

poses = np.array(poses)  # shape will be (N, 12) if each list has 12 elements

# print(poses.shape)
final_poses = []
for name, pose in zip(names, poses):
    if name in ['0025', '0145', '0092']:
        print(f"{name} : {pose}")
        final_poses.append(pose)
final_poses = np.array(final_poses)  # shape will be (M, 12) where M is the number of selected poses

# Save the poses to a new file
homo = np.zeros((final_poses.shape[0], 4, 4), dtype=np.float32)
homo[:, :3, :4] = final_poses.reshape(-1, 3, 4)
homo[:, 3, 3] = 1

print(f"{homo}")

# np.save('poses.npy', homo)                     # shape (N,4,4)


