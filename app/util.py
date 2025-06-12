import numpy as np
import torch
import pyvista as pv
import pymeshlab
import math
from IPython.display import display
import matplotlib.image as mpimg # mpimg 用于读取图片
import os
import cv2
import trimesh
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import csv
import torch
import torchvision.transforms.functional as TF
import skimage.metrics
from dataclasses import dataclass


@dataclass
class ModelData:
    albedo: torch.Tensor
    pos_idx: torch.Tensor  # [faces_num, 3], dtype=int32, cuda
    vtx_pos: torch.Tensor  # [vertex_num, 3], dtype=float32, cuda
    uv_idx: torch.Tensor   # [faces_num, 3], dtype=int32, cuda
    vtx_uv: torch.Tensor   # [uv_num, 2], dtype=float32, cuda
    normals: torch.Tensor  # [vertex_num, 3], dtype=float32, cuda
    roughness_var: torch.Tensor  # scalar, float32, cuda
    metallic_var: torch.Tensor   # scalar, float32, cuda
    roughness: torch.Tensor 
    face_info: np.ndarray
    mtl: str
    material_name: str

#----------------------------------------------------------------------------
# Projection and transformation matrix helpers.
#----------------------------------------------------------------------------

def D_GGX(n, h, roughness):
    alpha = roughness ** 2
    NdotH = torch.sum(n * h, dim=-1, keepdim=True).clamp(min=0.0)
    denom = NdotH ** 2 * (alpha ** 2 - 1.0) + 1.0
    return alpha ** 2 / (math.pi * denom ** 2 + 1e-8)

# 几何遮蔽函数
def G_Smith(n, v, l, roughness):
    def G1(w):
        NdotW = torch.sum(n * w, dim=-1, keepdim=True).clamp(min=0.0)
        k = (roughness + 1.0) ** 2 / 8.0
        return NdotW / (NdotW * (1.0 - k) + k + 1e-8)
    return G1(v) * G1(l)

# 菲涅尔项
def F_Schlick(cos_theta, F0):
    return F0 + (1.0 - F0) * (1.0 - cos_theta) ** 5

def load_obj(filepath):
    vertices = []
    normals = []
    uvs = []
    faces = []
    uv_idxs = []
    normal_idxs = []
    face_materials = []
    current_material = None
    materials = {}
    textures = {}
    original_faces_raw = []
    material_name = None

    mtl_path = None
    mtl = None
    obj_dir = os.path.dirname(filepath)

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:4])))
            elif line.startswith('vn '):
                normals.append(list(map(float, line.split()[1:4])))
            elif line.startswith('vt '):
                uvs.append(list(map(float, line.split()[1:3])))
            elif line.startswith('f '):
                original_faces_raw.append(line)  # 保存原始面数
                face = []
                uv_idx = []
                for v in line.split()[1:]:
                    idx = v.split('/')
                    # 支持 v / vt / vn 全三种
                    vi = int(idx[0]) - 1
                    vti = int(idx[1]) - 1 if len(idx) > 1 and idx[1] else -1
                    vni = int(idx[2]) - 1 if len(idx) > 2 and idx[2] else -1
                    face.append(vi)
                    uv_idx.append(vti)
                uv_idxs.append(uv_idx)
                faces.append(face)
                face_materials.append(current_material)
            elif line.startswith('mtllib '):
                mtl =  line.split()[1]
                mtl_path = os.path.join(obj_dir, line.split()[1])
            elif line.startswith('usemtl '):
                current_material = line.split()[1]

    # 加载材质文件
    if mtl_path and os.path.exists(mtl_path):
        with open(mtl_path, 'r') as mtl_file:
            material_name = None
            for line in mtl_file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('newmtl '):
                    material_name = line.split()[1]
                    materials[material_name] = {}
                elif line.startswith('map_Kd ') and material_name:
                    texture_file = line.split()[1]
                    texture_path = os.path.join(obj_dir, texture_file)
                    if os.path.exists(texture_path):
                        materials[material_name]['diffuse_map'] = texture_path
                        img = mpimg.imread(texture_path)
                        img = np.flipud(img)
                        albedo_map_ = img.astype(np.float32)/255.0
                elif line.startswith('Ns '):
                    Ns = float(line.split()[1])
                    roughness_var_ = math.sqrt(2.0/(Ns+2.0))

    # 转为 numpy
    vertices_ = np.array(vertices, dtype=np.float32)
    uvs_ = np.array(uvs, dtype=np.float32) if uvs else None
    faces_ = np.array(faces, dtype=np.int32)
    uv_idxs_ = np.array(uv_idxs, dtype=np.int32)
    if len(normals) == 0:
        normals = compute_vertex_normals(vertices_, faces_)
    normals_ = np.array(normals, dtype=np.float32)

    pos_idx = torch.from_numpy(faces_.astype(np.int32)).cuda() # faces [faces_num, 3] 
    vtx_pos = torch.from_numpy(vertices_.astype(np.float32)).cuda()   # vertices [vertexs_num, 3] 
    uv_idx  = torch.from_numpy(uv_idxs_.astype(np.int32)).cuda()  # faces每个点的uv index ↓  [faces_num, 3] 
    vtx_uv  = torch.from_numpy(uvs_.astype(np.float32)).cuda()    # uvs (0.37, 0.82) 单独的uv合集，许多顶点可以共同用uv  [uv_num, 2]
    normals = torch.as_tensor(normals_, dtype=torch.float32, device='cuda').contiguous()
    albedo_map = torch.from_numpy(albedo_map_.astype(np.float32)).cuda()
    metallic_var = torch.tensor(0, dtype=torch.float32, device='cuda')
    roughness = torch.full(normals.shape, roughness_var_, device='cuda').contiguous()
    roughness_var = torch.tensor(roughness_var_, dtype=torch.float32, device='cuda')

    model_data = ModelData(albedo = albedo_map, 
                            pos_idx = pos_idx,
                            vtx_pos = vtx_pos,
                            uv_idx = uv_idx,
                            vtx_uv = vtx_uv,
                            normals = normals,
                            roughness_var = roughness_var,
                            metallic_var = metallic_var,
                            roughness = roughness,
                            face_info = original_faces_raw,
                            mtl = mtl,
                            material_name = material_name
                            )
    
    return model_data

# 法线计算函数
def compute_vertex_normals(vertices, faces):
    normals = np.zeros_like(vertices, dtype=np.float32)
    for face in faces:
        idx = [face[0], face[1], face[2]]
        v0, v1, v2 = vertices[idx[0]], vertices[idx[1]], vertices[idx[2]]
        face_normal = np.cross(v1 - v0, v2 - v0)
        face_normal /= np.linalg.norm(face_normal) + 1e-8
        for i in idx:
            normals[i] += face_normal
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / (norms + 1e-8)


def simplify(model_path, LOD_level):
    # 生成新的文件名，添加 "_simplified" 后缀

    dir_name, base_name = os.path.split(model_path)
    name, ext = os.path.splitext(base_name)
    new_model_path = os.path.join(dir_name, f"{name}_LOD_{LOD_level}{ext}")
    
    if os.path.exists(new_model_path) == False:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(model_path)
        # 获取当前网格
        mesh = ms.current_mesh()

        # 获取面（faces）数量
        num_faces = mesh.face_number()
        print(f"模型面数: {num_faces}")

        target = num_faces
        for i in range(LOD_level):
            target /= 2
        print(f"减少至面数: {math.ceil(target)}")

        # 应用带拓扑/边界保护的简化
        ms.apply_filter(
            'meshing_decimation_quadric_edge_collapse',
            targetfacenum=math.ceil(target),                # 目标面数
            preservetopology=True,             # 保拓扑，不打洞
            preserveboundary=True,             # 保边界，不拆分
            preservenormal=True,               # 保法线
            optimalplacement=True,             # 顶点自动寻找最佳位置
            planarquadric=True,                # 平面区域适当保留三角面
            planarweight=0.001,                # 平面保留权重，越大越保留
            qualitythr=0.3                     # 面质量阈值，避免奇怪面片
        )
        
        # 保存简化后模型
        ms.save_current_mesh(new_model_path)

    return new_model_path

def export_to_obj(model_data: ModelData, save_path: str):
    """
    Export ModelData to a .obj file.
    """
    assert model_data.vtx_pos.ndim == 2 and model_data.vtx_pos.shape[1] == 3
    assert model_data.vtx_uv.ndim == 2 and model_data.vtx_uv.shape[1] == 2
    assert model_data.normals.ndim == 2 and model_data.normals.shape[1] == 3

    # 将 CUDA Tensor 转为 CPU，确保是 float32
    vtx_pos = model_data.vtx_pos.detach().cpu().numpy()
    vtx_uv = model_data.vtx_uv.detach().cpu().numpy()
    normals = model_data.normals.detach().cpu().numpy()
    vtx_pos = np.round(vtx_pos, 6)
    vtx_uv = np.round(vtx_uv, 6)
    normals = np.round(normals, 6)

    pos_idx = model_data.pos_idx.detach().cpu().numpy()
    uv_idx = model_data.uv_idx.detach().cpu().numpy()

    with open(save_path, 'w') as f:
        # 材质信息
        f.write(f"mtllib {model_data.mtl}\n")
        f.write(f"usemtl {model_data.material_name}\n")
        # 顶点位置
        for v in vtx_pos:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # UV 坐标
        for vt in vtx_uv:
            f.write(f"vt {vt[0]} {vt[1]}\n")  # OpenGL UV 是从底部开始的，OBJ 是从顶部
        # 法线
        for vn in normals:
            f.write(f"vn {vn[0]} {vn[1]} {vn[2]}\n")

        # 面索引，OBJ 索引是从 1 开始的
        for line in model_data.face_info:
            f.write(f"{line}\n")

    print(f"[✓] Exported optimized mesh to: {save_path}")

#----------------------------------------------------------------------------
# Projection and transformation matrix helpers.
#----------------------------------------------------------------------------

def projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0],
                     [  0,  n/x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]).astype(np.float32)

def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0],
                     [0,  c, s, 0],
                     [0, -s, c, 0],
                     [0,  0, 0, 1]]).astype(np.float32)

def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]]).astype(np.float32)

def scale(sx, sy, sz):
    return np.array([[sx, 0, 0, 0],
                     [0, sy, 0, 0],
                     [0, 0, sz, 0],
                     [0, 0, 0,  1]]).astype(np.float32)

def random_rotation_translation(t):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return m

def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def transform_normals_to_world(normal, mtx):

    t_mtx = torch.from_numpy(mtx).float().cuda() if isinstance(mtx, np.ndarray) else mtx
    M33 = t_mtx[:3, :3]  # 提取旋转缩放部分

    # normal 变换矩阵是逆转置
    normal_matrix = torch.inverse(M33).transpose(0, 1)

    normal_world = torch.matmul(normal, normal_matrix)  # (N, 3)
    normal_world = F.normalize(normal_world, dim=-1)
    return normal_world[None, ...]  # 加 batch 维变成 (1, N, 3)

#----------------------------------------------------------------------------
# Bilinear downsample by 2x.
#----------------------------------------------------------------------------

def bilinear_downsample(x):
    w = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=torch.float32, device=x.device) / 64.0
    w = w.expand(x.shape[-1], 1, 4, 4) 
    x = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, padding=1, stride=2, groups=x.shape[-1])
    return x.permute(0, 2, 3, 1)

#----------------------------------------------------------------------------
# Image display function using OpenGL.
#----------------------------------------------------------------------------

_glfw_window = None
def display_image(image, zoom=None, size=None, title=None): # HWC
    # Import OpenGL and glfw.
    import OpenGL.GL as gl
    import glfw

    # Zoom image if requested.
    image = np.asarray(image)
    if size is not None:
        assert zoom is None
        zoom = max(1, size // image.shape[0])
    if zoom is not None:
        image = image.repeat(zoom, axis=0).repeat(zoom, axis=1)
    height, width, channels = image.shape

    # Initialize window.
    if title is None:
        title = 'Debug window'
    global _glfw_window
    if _glfw_window is None:
        glfw.init()
        _glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(_glfw_window)
        glfw.show_window(_glfw_window)
        glfw.swap_interval(0)
    else:
        glfw.make_context_current(_glfw_window)
        glfw.set_window_title(_glfw_window, title)
        glfw.set_window_size(_glfw_window, width, height)

    # Update window.
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {'uint8': gl.GL_UNSIGNED_BYTE, 'float32': gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(_glfw_window)
    if glfw.window_should_close(_glfw_window):
        return False
    return True

#----------------------------------------------------------------------------
# Image save helper.
#----------------------------------------------------------------------------

def save_image(fn, x):
    import imageio
    x = np.rint(x * 255.0)
    x = np.clip(x, 0, 255).astype(np.uint8)
    imageio.imsave(fn, x)


def calculate_ssim_color(color, color_opt, C1=0.01**2, C2=0.03**2):
    """
    计算两个颜色张量的 SSIM（结构相似性）。
    :param color: 原图张量，形状 [C, H, W] 或 [H, W, C]
    :param color_opt: 经过优化后的图像张量，形状 [C, H, W] 或 [H, W, C]
    :param C1: SSIM 计算中的稳定因子（默认值 0.01^2）
    :param C2: SSIM 计算中的稳定因子（默认值 0.03^2）
    :return: SSIM 值（范围 0~1）
    """
    # 确保图像形状相同
    assert color.shape == color_opt.shape, "两个图像张量的形状必须相同！"

    # 计算均值
    mu_x = F.avg_pool2d(color, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(color_opt, kernel_size=3, stride=1, padding=1)

    # 计算方差和协方差
    sigma_x = F.avg_pool2d(color**2, kernel_size=3, stride=1, padding=1) - mu_x**2
    sigma_y = F.avg_pool2d(color_opt**2, kernel_size=3, stride=1, padding=1) - mu_y**2
    sigma_xy = F.avg_pool2d(color * color_opt, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

    # 计算 SSIM
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = numerator / denominator

    # 取全局 SSIM 均值
    return ssim_map.mean().item()

def save_csv(LOD_level, vtx_position, vtx_uv, normals, albedo, ssim_avg):
    csv_filename = "ssim_results.csv"

    if not os.path.exists(csv_filename):
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["LOD Level", "vtx_position_LR", "vtx_uv_LR", "normals_LR", "albedo_LR", "SSIM_avg"])
    
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([LOD_level, vtx_position, vtx_uv, normals, albedo, ssim_avg])

def save_txt(data, filename):
    with open(filename, 'w') as f:
        f.write(str(data))


def tensor_to_numpy(img_tensor):
    """将 PyTorch Tensor 转换为 NumPy 数组，确保范围在 [0,1]"""
    img_np = img_tensor.detach().cpu().numpy()  # 转换到 CPU 并变为 NumPy
    return np.clip(img_np, 0, 1)  # 确保数值在 [0,1] 之间

def compute_metrics(img1_tensor, img2_tensor):
    """计算 SSIM 和 PSNR"""
    img1 = tensor_to_numpy(img1_tensor)
    img2 = tensor_to_numpy(img2_tensor)

    ssim_val = skimage.metrics.structural_similarity(img1, img2, data_range=1, multichannel=True)
    psnr_val = skimage.metrics.peak_signal_noise_ratio(img1, img2, data_range=1)
    
    return ssim_val, psnr_val
#----------------------------------------------------------------------------
