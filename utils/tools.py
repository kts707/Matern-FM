import math
import torch

def batchify_tensor(tensor, batch_size):
    if len(tensor.shape) == 1:
        return tensor.unsqueeze(0).repeat(batch_size, 1)
    if len(tensor.shape) == 2:
        return tensor.unsqueeze(0).repeat(batch_size, 1, 1)
    else:
        raise NotImplementedError("currently only support 1D and 2D tensors")


def center_around_zeros_mass_weighted(tensor, vertex_mass, dimension=1):
    '''
    tensor: BxVxC
    vertex_mass: BxV
    '''
    
    assert dimension == 1, "Currently only support inputs of shape BxVxC"
    tensor_center = torch.sum(tensor * vertex_mass.unsqueeze(-1), dim=1, keepdim=True) / torch.sum(vertex_mass, dim=1, keepdim=True).unsqueeze(-1)

    return tensor - tensor_center


def MSE_loss(pred_v, tar_v, mass=None, mass_weighted=True):
    if mass_weighted:
        assert mass is not None, "Mass must be provided for mass-weighted loss"
        # L = ∑_v m_v * ‖v_tar - v_pred‖^2 / ∑_v m_v / num_channels
        return (
            (mass.unsqueeze(-1) * (pred_v - tar_v).pow(2)).sum(dim=(1, 2)) /
            (mass.sum(dim=1) * pred_v.shape[-1])
        ).mean()
    else:
        return ((pred_v - tar_v).pow(2)).mean()


def NJF_loss(pred_v, pred_grad, tar_v, G, v_mass, f_mass, mass_weighted=True):
    tar_grad = torch.bmm(G, tar_v)
    loss_v = MSE_loss(pred_v, tar_v, v_mass, mass_weighted)
    loss_g = MSE_loss(pred_grad, tar_grad, f_mass, mass_weighted)
    return loss_v, loss_g



def normalize_mesh(v, f, mode='surface_area', discretization_aware=True, return_scale_shift=False, target_surface_area=None):
    '''
    Normalizes mesh vertices using one of the following modes:
    - 'unit_sphere': centers the mesh and scales it to fit in unit sphere
    - 'surface_area': centers the mesh and scales it to have unit surface area

    `discretization_aware=True` optionally shifts the mesh using proper area-weighted centroid of the mesh
    '''
    import numpy as np
    import igl

    eps = 1e-8

    use_torch = torch.is_tensor(v)
    if use_torch:
        device, dtype = v.device, v.dtype
        v_np = v.detach().cpu().numpy()
        f_np = f.detach().cpu().numpy()
    else:
        v_np, f_np = v, f

    # compute (area-weighted) centroid in NumPy
    if discretization_aware or mode == 'surface_area':
        areas = igl.doublearea(v_np, f_np) * 0.5
        total_area = areas.sum()
        tri_centers = v_np[f_np].mean(axis=1)
        centroid_np = (areas[:,None] * tri_centers).sum(axis=0) / total_area
    else:
        centroid_np = v_np.mean(axis=0)

    v_centered_np = v_np - centroid_np

    if mode == 'unit_sphere':
        # max distance from origin
        dists = np.linalg.norm(v_centered_np, axis=1)
        scale_np = dists.max()
    elif mode == 'surface_area':
        scale_np = np.sqrt(total_area) + eps
        if target_surface_area is not None:
            scale_np = scale_np / np.sqrt(target_surface_area)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    v_out = v_centered_np / scale_np

    if use_torch:
        v_out = torch.from_numpy(v_out).to(device=device, dtype=dtype)
    
    if return_scale_shift:
        return v_out, scale_np, centroid_np
    
    return v_out