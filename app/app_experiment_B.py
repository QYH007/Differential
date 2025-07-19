import os
import numpy as np
import torch
import torch.nn.functional as F
import util
import math
import nvdiffrast.torch as dr
import time

#-------------------------------------Initial states--------------------------------------
#-------------- Control -----------
model_path        = "mesh/objects/"
model_name        = "bunny.obj"     # "buddha.obj", "buddha2.obj", "cat.obj", "feline.obj", "tiger.obj", "zebra.obj"
LOD_level         = 4               # from 1 to 7
SSAO_mode         = False           # toggle 
texture_mode      = True
max_mip_level     = 9

#------------- render ------------
ssao_samples      = 16
ssao_radius       = 0.05
ssao_sharpness    = 5.0
res               = 1024
ref_res           = 1024

#------------- Iteration -----------
max_iter          = 1000
display_interval  = 1000
log_interval      = 10
imgsave_fn        = 'img_%06d.png'
imgsave_interval  = 1000
display_res       = 512
use_opengl        = True
enable_mip        = True

#------------- learning rate ---------------
vtx_pos_lr        = [1e-3,  0,      1e-3,   1e-3,   1e-3,   0,      1e-3, 1e-3]
normal_lr         = [0,     1e-3,   0,      1e-3,   1e-3,   0,      1e-3, 1e-3]
vtx_uv_lr         = [0,     1e-4,   1e-4,   0,      1e-4,   1e-4,   0   , 1e-4]
albedo_lr         = [0,     1e-3,   1e-3,   1e-3,   0,      1e-3,   0  ,  1e-3]
lr_ramp           = 0.1

#-------------- output----------
out_dir           = f'./opt_result'
texsave_fn        = f'tex_{LOD_level}.png'

def render(glctx, mtx, lightdir, campos, d: util.ModelData, resolution, max_mip_level, SSAO_mode, M = None, V = None, P = None):
    # Rasterization
    pos_clip = util.transform_pos(mtx, d.vtx_pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, d.pos_idx, resolution=[resolution, resolution])
    mask = torch.clamp(rast_out[..., -1:], 0, 1)

    # lighting computation: direct + indirect + SSAO
    lighting_color = render_lighting_albedo(lightdir, d, M, campos, max_mip_level, rast_out, rast_out_db)

    env_color = render_ambient(d, max_mip_level, 0.7, rast_out, rast_out_db)

    if(SSAO_mode):
        ao = render_SSAO(d, M, V, P, rast_out, rast_out_db, kernel, ssao_samples, ssao_radius, ssao_sharpness)
        final_color = ao*env_color + lighting_color
        final_color = ao
    else:
        final_color = env_color + lighting_color
    
    background_color = torch.tensor([0.2, 0.2, 0.2], device=final_color.device)  # background color

    final_color = final_color * mask + background_color * (1 - mask)

    return final_color

def render_lighting_albedo(lightdir, d: util.ModelData, M, campos, max_mip_level, rast_out, rast_out_db):
    # get texture color each pixel
    uv_interp, uvdb_interp = dr.interpolate(d.vtx_uv[None, ...], rast_out, d.uv_idx, rast_db=rast_out_db, diff_attrs='all')
    albedo_interp = dr.texture(d.albedo[None, ...], uv_interp, uvdb_interp, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    
    # get normal each pixel
    normals_world = util.transform_normals_to_world(d.normals, M)
    normals_interp, _ = dr.interpolate(normals_world, rast_out, d.pos_idx, rast_db=rast_out_db, diff_attrs='all')
    normals_interp = normals_interp / (torch.sum(normals_interp ** 2, -1, keepdim=True) + 1e-8)**0.5  # 单位化法线
    
    # get view direction each pixel
    viewvec = campos[np.newaxis, np.newaxis, :] - d.vtx_pos[..., :3]
    viewvec_interp, _ = dr.interpolate(viewvec, rast_out, d.pos_idx, rast_db=rast_out_db, diff_attrs='all')
    viewvec_interp = viewvec_interp / (torch.sum(viewvec_interp ** 2, -1, keepdim=True) + 1e-8)**0.5
    
    # half vector each pixel
    ldir = -lightdir.view(1, 1, 3)
    hvec = (viewvec_interp + ldir)
    hvec = hvec / (torch.sum(hvec ** 2, -1, keepdim=True) + 1e-8)**0.5

    # clamp variable
    roughness = d.roughness_var.clamp(0.02, 1.0)
    metallic = d.metallic_var.clamp(0.0, 1.0)

    F0 = 0.04 * (1.0 - metallic) + albedo_interp * metallic

    # -------- Cook-Torrance BRDF --------
    D = util.D_GGX(normals_interp, hvec, roughness)
    G = util.G_Smith(normals_interp, viewvec_interp, ldir, roughness)
    F = util.F_Schlick(torch.sum(hvec * viewvec_interp, -1, keepdim=True).clamp(0.0, 1.0), F0)

    NdotV = torch.sum(normals_interp * viewvec_interp, -1, keepdim=True).clamp(1e-4, 1.0)
    NdotL = torch.sum(normals_interp * ldir, -1, keepdim=True).clamp(1e-4, 1.0)

    spec = (D * F * G) / (4.0 * NdotV * NdotL + 1e-8)
    k_d = (1.0 - F) * (1.0 - metallic)
    diffuse = k_d * albedo_interp / math.pi

    brdf_color = (diffuse + spec) * NdotL

    return brdf_color

def render_ambient(d: util.ModelData, max_mip_level, intensity, rast_out, rast_out_db):
    texc, texd = dr.interpolate(d.vtx_uv[None, ...], rast_out, d.uv_idx, rast_db=rast_out_db, diff_attrs='all')
    color = dr.texture(d.albedo[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level) 
    return color * intensity
    
def render_SSAO(d: util.ModelData, M, V, P, rast_out, rast_out_db, kernel, num_samples=16, radius=0.02, sharpness=3.0, bias=0.01):
    # --- interpolate geometry info ---
    # get world position each pixel
    pos_world = util.transform_pos(M, d.vtx_pos)
    positions_interp, _ = dr.interpolate(pos_world, rast_out, d.pos_idx, rast_db=rast_out_db, diff_attrs='all')
    
    # get world normal each pixel
    normals_world = util.transform_normals_to_world(d.normals, M)
    normals_interp, _ = dr.interpolate(normals_world, rast_out, d.pos_idx, rast_db=rast_out_db, diff_attrs='all')
    normals_interp = normals_interp / (torch.sum(normals_interp ** 2, -1, keepdim=True) + 1e-8)**0.5  # 单位化法线

    normal_map = normals_interp.permute(0, 3, 1, 2)  # -> [B, 3, H, W]
    position_map = positions_interp.permute(0, 3, 1, 2)  # -> [B, 4, H, W]

    B, _, H, W = position_map.shape
    device = position_map.device
    valid_mask = (
        torch.clamp(rast_out[..., -1:], 0, 1)     # [B, H, W, 1]
        .permute(0, 3, 1, 2)                      # [B, 1, H, W]
        .unsqueeze(1)                             # [B, 1, 1, H, W]
        .expand(-1, num_samples, -1, -1, -1)      # [B, N, 1, H, W]
        .reshape(B * num_samples, 1, H, W)        # [B*N, 1, H, W]
    )

    # construct TBN basis by world normal per pixel
    up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    T = F.normalize(torch.cross(normal_map, up.expand_as(normal_map)), dim=1)
    B_ = torch.cross(normal_map, T)
    TBN = torch.stack([T, B_, normal_map], dim=-1)  # (B, 3, H, W, 3)

    # project samples direction to world coord.
    # align tensor lengths
    kernel_exp = kernel.view(1, num_samples, 1, 1, 3)  # (1, N, 1, 1, 3)
    TBN_t = TBN.permute(0, 2, 3, 1, 4)  # (B, H, W, 3, 3)
    TBN_exp = TBN_t.unsqueeze(1)  # (B, 1, H, W, 3, 3)
    kernel_exp = kernel_exp.expand(-1, -1, TBN_t.shape[1], TBN_t.shape[2], -1).unsqueeze(-1)
    sample_dirs_world = torch.matmul(TBN_exp, kernel_exp).squeeze(-1) # ([B, N, H, W, 3])

    # construct sample points in world positoin
    position_map = position_map[:, :3, :, :]
    position_exp = position_map.unsqueeze(1).permute(0, 1, 3, 4, 2)  # (B, 1, H, W, 3)
    sample_positions = position_exp + radius * sample_dirs_world  # (B, N, H, W, 3)

    # get sample points NDC position
    sample_positions_homo = torch.cat([sample_positions, torch.ones_like(sample_positions[..., :1])], dim=-1)  # (B, N, H, W, 4)

    # View-Projection Matrix
    vp = np.matmul(P, V).astype(np.float32) # 4,4
    vp = torch.from_numpy(vp).to(device=device, dtype=torch.float32)
    sample_clip = torch.matmul(sample_positions_homo, vp.t())

    # [x,y,z] / w -> NDC
    sample_ndc = sample_clip[..., :3] / (sample_clip[..., 3:4] + 1e-8)  # 防止除0

    # get sample points' [x, y] NDC position
    offset_grid = sample_ndc[..., :2]  # (B, N, H, W, 2)

    # perpare for grid sample
    offset_grid = offset_grid.reshape(B * num_samples, H, W, 2)
    offset_grid = torch.clamp(offset_grid, -1, 1)

    # grid_sample on origin position map
    position_map_expand = position_map.unsqueeze(1).expand(-1, num_samples, -1, -1, -1)  # (B, N, 3, H, W)
    position_map_expand = position_map_expand.reshape(B * num_samples, 3, H, W)

    sampled_pos = F.grid_sample(position_map_expand, offset_grid, mode='bilinear', align_corners=False)
    sampled_valid = F.grid_sample(valid_mask, offset_grid, mode='bilinear', align_corners=False)

    # thus get sampled position of each pixel and referrence positoin of each pixel
    sampled_pos = sampled_pos.view(B, num_samples, 3, H, W)
    sampled_pos *= sampled_valid
    ref_pos = position_map.unsqueeze(1).expand(-1, num_samples, -1, -1, -1)  # (B, N, 3, H, W)

    # compute SSAO by sigmoid
    delta = sampled_pos - ref_pos

    normal_map_expand = normal_map.unsqueeze(1).expand(-1, num_samples, -1, -1, -1)  # (B, N, 3, H, W)
 
    # compute projection of delta w.r.t. normal, dot(delta, normal)
    dp = (delta * normal_map_expand).sum(dim=2)  # (B, N, H, W)

    # Soft occlusion
    occ = torch.sigmoid((dp - bias) *sharpness) - 0.5

    valid_count = sampled_valid.squeeze(1)  
    valid_count = valid_count.view(B, num_samples, H, W).sum(dim=1) + 1e-6  # [B, H, W]

    occ = occ * sampled_valid.view(B, num_samples, H, W)
    occ_sum = occ.sum(dim=1)  # [B, H, W]
    ao = 1.0 - (occ_sum / valid_count)  # [B, H, W]
    ao = torch.clamp(ao, 0.0, 1.0)
    ao = ao.unsqueeze(-1)  # [1, 512, 512, 1]
    return ao**1.5
    
# ----------------------- Step.1  prepareing -------------------------
os.makedirs(out_dir, exist_ok=True)

for test in range(len(vtx_pos_lr)):
    ssim_accumulator = []
    for avg in range(3):
        #-------------------------------------Initial states--------------------------------------
        model_data = util.load_obj(model_path + model_name)
        # loading origin model data, and pack into util.ModelData
        low_res_path = util.simplify(model_path + model_name, LOD_level)
        low_model_data = util.load_obj(low_res_path)

        low_model_data.vtx_pos.requires_grad_()
        low_model_data.vtx_uv.requires_grad_()
        low_model_data.normals.requires_grad_()
        low_model_data.albedo = model_data.albedo.clone()
        low_model_data.roughness_var = model_data.roughness_var
        low_model_data.metallic_var = model_data.metallic_var
            
        glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
        ang = 0.0

        # Adam optimizer for texture with a learning rate ramp.
        if texture_mode:
            low_model_data.albedo.requires_grad_()
            optimizer = torch.optim.Adam([
                {'params': low_model_data.vtx_pos, 'lr': vtx_pos_lr[test]},  
                {'params': low_model_data.vtx_uv, 'lr': vtx_uv_lr[test]},  
                {'params': low_model_data.normals, 'lr': normal_lr[test]},
                {'params': low_model_data.albedo, 'lr': albedo_lr[test]}    
            ])
        else:
            optimizer = torch.optim.Adam([
                {'params': low_model_data.vtx_pos, 'lr': vtx_pos_lr[test]},  
                {'params': low_model_data.vtx_uv, 'lr': vtx_uv_lr[test]},  
                {'params': low_model_data.normals, 'lr': normal_lr[test]} 
            ])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp**(float(x)/float(max_iter)))

        # Render.
        ang = 0.0
        losses = [] 
        loss_avg = []
        SSIMs = []
        kernel = util.sample_hemisphere(ssao_samples, device='cuda')
        kernel = F.normalize(kernel, dim=-1)
        scales = torch.rand(ssao_samples, 1, device='cuda') ** 2  # bias towards center
        kernel = kernel * scales  # now in unit sphere, not unit sphere surface
        for it in range(max_iter + 1):
            torch.cuda.synchronize()
            start_time = time.time()
            # Random rotation/translation matrix for optimization.
            r_rot = util.random_rotation_translation(0.25)

            # Smooth rotation for display.
            a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(ang))
            dist = np.random.uniform(0.0, 30)

            P = util.projection(x=0.4, n=1.5, f=100.0).astype(np.float32)

            r_V = util.translate(0, 0, -10 - dist).astype(np.float32)
            r_M = np.matmul(r_rot, util.scale(0.01, 0.01, 0.01)).astype(np.float32)
            r_MV = np.matmul(r_V, r_M).astype(np.float32)
            r_MVP = np.matmul(P, r_MV).astype(np.float32)

            a_V = util.translate(0, 0, -15).astype(np.float32)
            a_M = np.matmul(a_rot, util.scale(0.01, 0.01, 0.01)).astype(np.float32)
            a_MV = np.matmul(a_V, a_M).astype(np.float32)
            a_MVP = np.matmul(P, a_MV).astype(np.float32)

            # Solve camera positions.
            a_campos = torch.as_tensor(np.linalg.inv(a_MV)[:3, 3], dtype=torch.float32, device='cuda')
            r_campos = torch.as_tensor(np.linalg.inv(r_MV)[:3, 3], dtype=torch.float32, device='cuda')

            lightdir = np.asarray([.8, -1., .5, 0.0])
            lightdir = np.matmul(a_MVP, lightdir)[:3]
            lightdir /= np.linalg.norm(lightdir)
            lightdir = torch.as_tensor(lightdir, dtype=torch.float32, device='cuda')
            
            # Render reference and optimized frames. Always enable mipmapping for reference.
            color     = render(glctx, r_MVP, lightdir, r_campos, model_data,     ref_res, max_mip_level, SSAO_mode, M = r_M, V= r_V, P = P)
            color_opt = render(glctx, r_MVP, lightdir, r_campos, low_model_data, ref_res, max_mip_level, SSAO_mode, M = r_M, V= r_V, P = P)
            # Reduce the reference to correct size.
            while color.shape[1] > res:
                color = util.bilinear_downsample(color)

            # Compute loss and perform a training step.
            L2 = torch.mean((color - color_opt)**2) # L2 pixel loss.
            ssim_map = util.calculate_ssim_color(color, color_opt)
            loss =  torch.mean(1.0 - ssim_map)
            losses.append(float(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Print/save log.
            if log_interval and (it % log_interval == 0):
                color_opt = render(glctx, a_MVP, lightdir, a_campos, low_model_data, ref_res, max_mip_level, SSAO_mode,  M = a_M, V= a_V, P = P)
                color = render(glctx, a_MVP, lightdir, a_campos, model_data, ref_res, max_mip_level, SSAO_mode,  M = a_M, V= a_V, P = P)
                ssim = util.calculate_ssim_color(color, color_opt).mean().item()
                SSIMs.append(ssim) # ssim
                s = "iter=%d, loss=%f, SSIM=%f, " % (it, loss, ssim)
                print(s)

            # Show/save image.
            display_image = display_interval and (it % display_interval == 0)

            if display_image:
                with torch.no_grad():
                    result_image = color_opt = render(glctx, a_MVP, lightdir, a_campos, low_model_data, ref_res, max_mip_level, SSAO_mode,  M = a_M, V= a_V, P = P)[0].cpu().numpy()[::-1]
                    if display_image:
                        util.display_image(result_image, size=display_res, title='%d / %d' % (it, max_iter))

        # Done.
        if avg == 0:
            ssim_accumulator = [0.0] * len(SSIMs)
        for i in range(len(SSIMs)):
            ssim_accumulator[i] += SSIMs[i]
        # plot_loss_curve(SSIMs)
    ssim_accumulator = [x / 3.0 for x in ssim_accumulator]
    log_path = "./LT_logs/"
    txt_filename =  log_path + "B"+str(test+1) +".txt"
    util.save_txt(ssim_accumulator, txt_filename)
#----------------------------------------------------------------------------
