import argparse
import os
import pathlib
import sys
import numpy as np
import trimesh
import torch
import matplotlib.image as mpimg # mpimg 用于读取图片
import util
import math
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class ModelData:
    albedo: torch.Tensor   # [res, res, 3], dtype=int32, cuda
    pos_idx: torch.Tensor  # [faces_num, 3], dtype=int32, cuda
    vtx_pos: torch.Tensor  # [vertex_num, 3], dtype=float32, cuda
    uv_idx: torch.Tensor   # [faces_num, 3], dtype=int32, cuda
    vtx_uv: torch.Tensor   # [uv_num, 2], dtype=float32, cuda
    normals: torch.Tensor  # [vertex_num, 3], dtype=float32, cuda
    roughness_var: torch.Tensor  # scalar, float32, cuda
    metallic_var: torch.Tensor   # scalar, float32, cuda

# loss_avg 
def plot_loss_curve(loss_avg):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_avg) + 1), loss_avg, marker='o', linestyle='-', color='b', label='Loss')
    plt.xlabel('Iteration')  # X 轴表示迭代次数
    plt.ylabel('Loss Value')  # Y 轴表示损失值
    plt.title('Loss Curve')  # 标题
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
    plt.show()

def render(glctx, mtx, lightdir, campos, d: ModelData, resolution, enable_mip, max_mip_level):

    # extract info
    pos_clip = util.transform_pos(mtx, d.vtx_pos)

    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, d.pos_idx, resolution=[resolution, resolution])

    lighting_color = render_lighting_albedo(lightdir, d, campos, enable_mip, max_mip_level, rast_out, rast_out_db)
    env_color = render_ambient(d, max_mip_level, 0.5, enable_mip, rast_out, rast_out_db)

    final_color = lighting_color + env_color

    background_color = torch.tensor([0.2, 0.2, 0.2], device=final_color.device)  # 定义灰色背景
    mask = torch.clamp(rast_out[..., -1:], 0, 1)  # 获取前景/背景 mask

    final_color = final_color * mask + background_color * (1 - mask)

    return final_color

def render_lighting(lightdir, pos, pos_idx, normals, campos, roughness_var, metallic_var, albedo_var, rast_out, rast_out_db):        
    # 插值对象：normal + viewDir + reflect
    # -------- 插值 --------
    # 顶点法向量
    normals_interp, _ = dr.interpolate(normals, rast_out, pos_idx, rast_db=rast_out_db, diff_attrs='all')
    normals_interp = normals_interp / (torch.sum(normals_interp ** 2, -1, keepdim=True) + 1e-8)**0.5  # 单位化法线
    
    # 视线向量
    viewvec = campos[np.newaxis, np.newaxis, :] - pos[..., :3]
    viewvec_interp, _ = dr.interpolate(viewvec, rast_out, pos_idx, rast_db=rast_out_db, diff_attrs='all')
    viewvec_interp = viewvec_interp / (torch.sum(viewvec_interp ** 2, -1, keepdim=True) + 1e-8)**0.5

    # 半程向量
    ldir = -lightdir.view(1, 1, 3)
    hvec = (viewvec_interp + ldir)
    hvec = hvec / (torch.sum(hvec ** 2, -1, keepdim=True) + 1e-8)**0.5

    # -------- 材质参数 --------
    roughness = roughness_var.clamp(0.02, 1.0)  # 手动保证范围
    metallic = metallic_var.clamp(0.0, 1.0)     # 金属度 0~1
    albedo = albedo_var.clamp(0.0, 1.0)         # 颜色 0~1

    F0 = 0.04 * (1.0 - metallic) + albedo * metallic  # 菲涅尔基准反射率

    # -------- Cook-Torrance BRDF --------
    D = util.D_GGX(normals_interp, hvec, roughness)
    G = util.G_Smith(normals_interp, viewvec_interp, ldir, roughness)
    F = util.F_Schlick(torch.sum(hvec * viewvec_interp, -1, keepdim=True).clamp(0.0, 1.0), F0)

    NdotV = torch.sum(normals_interp * viewvec_interp, -1, keepdim=True).clamp(1e-4, 1.0)
    NdotL = torch.sum(normals_interp * ldir, -1, keepdim=True).clamp(1e-4, 1.0)

    spec = (D * F * G) / (4.0 * NdotV * NdotL + 1e-8)  # 镜面反射

    # 漫反射部分
    k_d = (1.0 - F) * (1.0 - metallic)
    diffuse = k_d * albedo / math.pi

    # -------- 光照 --------
    brdf_color = (diffuse + spec) * NdotL  # BRDF 结合 N.L
    return brdf_color

def render_lighting_albedo(lightdir, d: ModelData, campos, enable_mip, max_mip_level, rast_out, rast_out_db):
    # 获取逐像素albedo
    if enable_mip:
        uv_interp, uvdb_interp = dr.interpolate(d.vtx_uv[None, ...], rast_out, d.uv_idx, rast_db=rast_out_db, diff_attrs='all')
        albedo_interp = dr.texture(d.albedo[None, ...], uv_interp, uvdb_interp, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        uv_interp, uvdb_interp = dr.interpolate(d.vtx_uv[None, ...], rast_out, d.uv_idx)
        albedo_interp = dr.texture(d.albedo[None, ...], uv_interp, filter_mode='linear')  
    
    # 获取逐像素顶点法向量
    normals_interp, _ = dr.interpolate(d.normals, rast_out, d.pos_idx, rast_db=rast_out_db, diff_attrs='all')
    normals_interp = normals_interp / (torch.sum(normals_interp ** 2, -1, keepdim=True) + 1e-8)**0.5  # 单位化法线
    
    # 视线向量
    viewvec = campos[np.newaxis, np.newaxis, :] - d.vtx_pos[..., :3]
    viewvec_interp, _ = dr.interpolate(viewvec, rast_out, d.pos_idx, rast_db=rast_out_db, diff_attrs='all')
    viewvec_interp = viewvec_interp / (torch.sum(viewvec_interp ** 2, -1, keepdim=True) + 1e-8)**0.5
    
    # 半程向量
    ldir = -lightdir.view(1, 1, 3)
    hvec = (viewvec_interp + ldir)
    hvec = hvec / (torch.sum(hvec ** 2, -1, keepdim=True) + 1e-8)**0.5

    # -------- 材质参数 --------
    roughness = d.roughness_var.clamp(0.02, 1.0)  # 手动保证范围
    metallic = d.metallic_var.clamp(0.0, 1.0)     # 金属度 0~1

    F0 = 0.04 * (1.0 - metallic) + albedo_interp * metallic  # 菲涅尔基准反射率

    # -------- Cook-Torrance BRDF --------
    D = util.D_GGX(normals_interp, hvec, roughness)
    G = util.G_Smith(normals_interp, viewvec_interp, ldir, roughness)
    F = util.F_Schlick(torch.sum(hvec * viewvec_interp, -1, keepdim=True).clamp(0.0, 1.0), F0)

    NdotV = torch.sum(normals_interp * viewvec_interp, -1, keepdim=True).clamp(1e-4, 1.0)
    NdotL = torch.sum(normals_interp * ldir, -1, keepdim=True).clamp(1e-4, 1.0)

    spec = (D * F * G) / (4.0 * NdotV * NdotL + 1e-8)  # 镜面反射

    # 漫反射部分
    k_d = (1.0 - F) * (1.0 - metallic)
    diffuse = k_d * albedo_interp / math.pi

    # -------- 光照 --------
    brdf_color = (diffuse + spec) * NdotL  # BRDF 结合 N.L

    return brdf_color

def render_env(envmap, d: ModelData, campos, intensity, rast_out, rast_out_db):
    viewvec = d.pos[..., :3] - campos[np.newaxis, np.newaxis, :]
    # 反射向量
    reflvec = viewvec - 2.0 * d.normals[np.newaxis, ...] * torch.sum(d.normals[np.newaxis, ...] * viewvec, -1, keepdim=True) # Reflection vectors at vertices.
    reflvec = reflvec / torch.sum(reflvec**2, -1, keepdim=True)**0.5 # Normalize.
    reflvec_interp, refld_interp = dr.interpolate(reflvec, rast_out, d.pos_idx, rast_db=rast_out_db, diff_attrs='all') # Interpolated reflection vectors.
    
    # target img
    env_color = dr.texture(envmap[np.newaxis, ...], reflvec_interp, uv_da=refld_interp, filter_mode='linear-mipmap-linear', boundary_mode='cube')

    return env_color*intensity

def render_ambient(d: ModelData, max_mip_level, intensity, enable_mip, rast_out, rast_out_db):
    if enable_mip:
        texc, texd = dr.interpolate(d.vtx_uv[None, ...], rast_out, d.uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(d.albedo[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(d.vtx_uv[None, ...], rast_out, d.uv_idx)
        color = dr.texture(d.albedo[None, ...], texc, filter_mode='linear')    
    return color * intensity

def load_model(path, texture_path=None, metallic_var=0.0, roughness_var=0.0):
    # loading origin model data, and pack into modelData
    mesh = trimesh.load_mesh(path)

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump()[0]  # Access the first mesh in the scene

    # 获取几何信息
    pos = mesh.vertices
    pos_idx = mesh.faces
    normals = mesh.vertex_normals
    if mesh.visual.uv is not None:
        uv = mesh.visual.uv  # UV坐标 (N, 2)
        uv_idx = mesh.faces  # 每个面对应的UV索引 (M, 3)，M为面数
    if pos.shape[1] == 4: pos = pos[:, 0:3]

    material = mesh.visual.material  # 获取材质信息
    
    if material != None:
        Kd = np.array(material.diffuse)
        Ka = material.ambient   # pbr无视
        Ks = material.specular  # pbr无视
        Ns = material.glossiness

        img = material.image  # 贴图

        if texture_path == None:
            if img == None:
                #如果全都没有的话，把贴图设置成全部都是Kd的颜色
                img = np.full((1024, 1024, 3), Kd, dtype=np.float32)  # 生成 1024x1024 纯色纹理
        else:
            img = mpimg.imread(texture_path)

        img = np.flipud(img)
        albedo_map = img.astype(np.float32)/255.0
        if Ns != None:
            roughness_var = math.sqrt(2.0/(Ns+2.0))
    
    # Create position/triangle index tensors
    pos_idx = torch.from_numpy(pos_idx.astype(np.int32)).cuda() # faces [faces_num, 3] 
    vtx_pos = torch.from_numpy(pos.astype(np.float32)).cuda()   # vertices [vertexs_num, 3] 
    uv_idx  = torch.from_numpy(uv_idx.astype(np.int32)).cuda()  # faces每个点的uv index ↓  [faces_num, 3] 
    vtx_uv  = torch.from_numpy(uv.astype(np.float32)).cuda()    # uvs (0.37, 0.82) 单独的uv合集，许多顶点可以共同用uv  [uv_num, 2]
    normals = torch.as_tensor(normals, dtype=torch.float32, device='cuda').contiguous()
    albedo_map = torch.from_numpy(albedo_map.astype(np.float32)).cuda()
    metallic_var = torch.tensor(metallic_var, dtype=torch.float32, device='cuda')
    roughness_var = torch.tensor(roughness_var, dtype=torch.float32, device='cuda')
    
    model_data = ModelData(albedo = albedo_map, 
                            pos_idx = pos_idx,
                            vtx_pos = vtx_pos,
                            uv_idx = uv_idx,
                            vtx_uv = vtx_uv,
                            normals = normals,
                            roughness_var = roughness_var,
                            metallic_var = metallic_var
                            )
    
    return model_data

#-------------------------------------Initial states--------------------------------------
max_iter          = 4000
log_interval      = 100
display_interval  = 10
display_res       = 1024
enable_mip        = True
res               = 2048
ref_res           = 2048
lr_base           = 1e-4
lr_ramp           = 0.1
out_dir           = f'./opt_result'
log_fn            = 'log.txt'
texsave_interval  = 1000
texsave_fn        = 'tex_%06d.png'
imgsave_interval  = 1000
imgsave_fn        = 'img_%06d.png'
use_opengl        = False

print (f'Saving results under {out_dir}')

log_file = None
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    if log_fn:
        log_file = open(out_dir + '/' + log_fn, 'wt')
else:
    imgsave_interval, texsave_interval = None, None
#------------------------------ Step.1 prepare model data ----------------------------
# Create LOD mesh
model_path = "mesh/objects/bunny.obj"
# model_path = "mesh/objects/buddha2.obj"
simplified_path = util.simplify(model_path, 3)

# class ModelData:
#     albedo: torch.Tensor
#     pos_idx: torch.Tensor  # [faces_num, 3], dtype=int32, cuda
#     vtx_pos: torch.Tensor  # [vertex_num, 3], dtype=float32, cuda
#     uv_idx: torch.Tensor   # [faces_num, 3], dtype=int32, cuda
#     vtx_uv: torch.Tensor   # [uv_num, 2], dtype=float32, cuda
#     normals: torch.Tensor  # [vertex_num, 3], dtype=float32, cuda
#     roughness_var: torch.Tensor  # scalar, float32, cuda
#     metallic_var: torch.Tensor   # scalar, float32, cuda
#     albedo_var: torch.Tensor     # [3], float32, cuda (RGB)

# load the two mesh
model_data = load_model(model_path)

max_mip_level = 9 # Texture is a 4x3 atlas of 512x512 maps.
# loading origin model data, and pack into modelData
low_model_data = load_model(simplified_path)

low_model_data.vtx_pos.requires_grad_()
low_model_data.vtx_uv.requires_grad_()
low_model_data.normals.requires_grad_()

low_model_data.albedo = model_data.albedo
low_model_data.roughness_var = model_data.roughness_var
low_model_data.metallic_var = model_data.metallic_var
    
glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

ang = 0.0

# Adam optimizer for texture with a learning rate ramp.
optimizer    = torch.optim.Adam([low_model_data.vtx_pos, low_model_data.vtx_uv, low_model_data.normals], lr=lr_base)
scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp**(float(x)/float(max_iter)))

# Render.
ang = 0.0
losses = []
loss_avg = []
for it in range(max_iter + 1):
    # Random rotation/translation matrix for optimization.
    r_rot = util.random_rotation_translation(0.25)

    # Smooth rotation for display.
    a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(ang))
    dist = np.random.uniform(0.0, 48.5)

    # Modelview and modelview + projection matrices.
    proj  = util.projection(x=0.4, n=1.0, f=200.0)
    r_mv  = np.matmul(util.translate(0, 0, -10-dist), r_rot)
    r_mv  = np.matmul(r_mv, util.scale(0.01, 0.01, 0.01)).astype(np.float32)
    r_mvp = np.matmul(proj, r_mv).astype(np.float32)
    a_mv  = np.matmul(util.translate(0, 0, -15), a_rot)
    a_mv  = np.matmul(a_mv, util.scale(0.01, 0.01, 0.01)).astype(np.float32)
    a_mvp = np.matmul(proj, a_mv).astype(np.float32)

    # Solve camera positions.
    a_campos = torch.as_tensor(np.linalg.inv(a_mv)[:3, 3], dtype=torch.float32, device='cuda')
    r_campos = torch.as_tensor(np.linalg.inv(r_mv)[:3, 3], dtype=torch.float32, device='cuda')

    lightdir = np.asarray([.8, -1., .5, 0.0])
    lightdir = np.matmul(a_mvp, lightdir)[:3]
    lightdir /= np.linalg.norm(lightdir)
    lightdir = torch.as_tensor(lightdir, dtype=torch.float32, device='cuda')

    # Render reference and optimized frames. Always enable mipmapping for reference.

    color     = render(glctx, r_mvp, lightdir, r_campos, model_data,     ref_res, True, max_mip_level)
    color_opt = render(glctx, r_mvp, lightdir, r_campos, low_model_data, ref_res, True, max_mip_level)
    # Reduce the reference to correct size.
    while color.shape[1] > res:
        color = util.bilinear_downsample(color)

    # Compute loss and perform a training step.
    loss = torch.mean((color - color_opt)**2) # L2 pixel loss.
    losses.append(float(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Print/save log.
    if log_interval and (it % log_interval == 0):
        log_loss_avg = np.mean(np.asarray(losses))
        loss_avg.append(float(log_loss_avg))
        s = "iter=%d,loss_avg=%f,loss=%f" % (it, log_loss_avg, loss)
        print(s)
        if log_file:
            log_file.write(s + '\n')
        if loss < 0.000001:
            break

    # Show/save image.
    display_image = display_interval and (it % display_interval == 0)
    save_image = imgsave_interval and (it % imgsave_interval == 0)
    save_texture = texsave_interval and (it % texsave_interval) == 0

    if it == 0:
        origin_img = render(glctx, a_mvp, lightdir, a_campos, model_data, ref_res, True, max_mip_level)[0].cpu().numpy()[::-1]
        util.save_image(out_dir + '/origin_' + (imgsave_fn % it), origin_img)

    if display_image or save_image:
        ang = ang + 0.1

        with torch.no_grad():
            result_image = color_opt = render(glctx, a_mvp, lightdir, a_campos, low_model_data, ref_res, True, max_mip_level)[0].cpu().numpy()[::-1]

            if display_image:
                util.display_image(result_image, size=display_res, title='%d / %d' % (it, max_iter))
            if save_image:
                util.save_image(out_dir + '/' + (imgsave_fn % it), result_image)

            if save_texture:
                texture = low_model_data.albedo.cpu().numpy()[::-1]
                util.save_image(out_dir + '/' + (texsave_fn % it), texture)


# Done.
plot_loss_curve(loss_avg)
if log_file:
    log_file.close()


#----------------------------------------------------------------------------
