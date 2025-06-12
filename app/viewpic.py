    
import argparse
import os
import pathlib
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import util
import trimesh
import matplotlib.image as mpimg # mpimg 用于读取图片

import nvdiffrast.torch as dr

import imageio.v2 as imageio
import numpy as np

model_path        = "mesh/objects/"
model_name        = "bunny.obj"
mesh = trimesh.load_mesh(model_path+model_name)

pos = mesh.vertices
pos_idx = mesh.faces

normals = mesh.vertex_normals
if mesh.visual.uv is not None:
    uv = mesh.visual.uv  # UV坐标 (N, 2)
    uv_idx = mesh.faces  # 每个面对应的UV索引 (M, 3)，M为面数

if pos.shape[1] == 4: pos = pos[:, 0:3]
print("---------------------------")
print("pos.shape: ", pos.shape)
print("pos_idx.shape", pos_idx.shape)
print("normals.shape", normals.shape)
print("uv.shape", uv.shape)
print("uv_idx.shape", uv_idx.shape)
unique_pos, inverse_indices = np.unique(mesh.vertices, axis=0, return_inverse=True)
print(f"Original vertices: {mesh.vertices.shape[0]}, Unique vertices: {unique_pos.shape[0]}")
print("---------------------------")
print(uv_idx)

# 1. 获取每个角点的位置坐标
corner_positions = mesh.vertices[mesh.faces.reshape(-1)]  # shape: [N_faces*3, 3]

# 2. 用 np.unique 去重，看看每个 face 角上的坐标数 vs 实际 unique 数
unique_corners = np.unique(corner_positions, axis=0)

print(f"角点总数: {corner_positions.shape[0]}, 唯一位置数: {unique_corners.shape[0]}")