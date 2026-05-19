import numpy as np
import trimesh
import torch

import argparse

import cholespy
import potpourri3d as pp3d

def parse_args():
    parser = argparse.ArgumentParser(description='Sample Matern noise on a mesh')
    parser.add_argument('--mesh', type=str, default='path/to/your/mesh.obj', help='Path to input mesh file')
    parser.add_argument('--screening_term', type=float, default=100.0, help='Screening term')
    parser.add_argument('--sigma', type=float, default=5.0, help='Standard deviation of the sampled iid Gaussiannoise')
    args = parser.parse_args()
    return args

def construct_screened_solvers(verts_np, faces_np, screening_term=0, device='cuda'):
    '''
    Creates the following operators for a mesh. Uses PyTorch CUDA extension.
    - solver:       [B,]        list of Cholesky solvers for mesh's cotangent Laplacian

    Optionally set high_precision to True to use double precision in intermediate
    computations. Note the operators will be returned in float precision regardless.
    '''

    nV = verts_np.shape[0]

    L_scipy = pp3d.cotan_laplacian(verts_np, faces_np, denom_eps=1e-10)
    massvec_np = pp3d.vertex_areas(verts_np, faces_np)

    mass_vec = torch.tensor(massvec_np, dtype=torch.float32)

    L_coo = L_scipy.tocoo()
    indices = torch.tensor(np.vstack((L_coo.row, L_coo.col)), dtype=torch.long, device=device)
    values = torch.tensor(L_coo.data, dtype=torch.float32, device=device)
    L_pp3d = torch.sparse_coo_tensor(indices, values, size=L_scipy.shape, device=device, dtype=torch.float32)

    solvers = []

    # Create Cholesky solver for each Laplacian in batch
    # This could be block-diagonalized, but speedup seems marginal
    eps = 1e-6
    if screening_term == 0:
        print('screening term is set to zero!')
        sparse_eps_diag = torch.sparse.spdiags(eps * torch.ones(nV), torch.zeros(1, dtype=torch.long), (nV, nV)).to(device)
    else:
        print(f'screening term: {screening_term}')
        sparse_eps_diag = torch.sparse.spdiags(screening_term * torch.ones(nV) * mass_vec, torch.zeros(1, dtype=torch.long), (nV, nV)).to(device)
    L_pp3d = L_pp3d + sparse_eps_diag # (b, v, v)
    
    nretry = 0
    while nretry < 5:
        try:
            nrows = L_pp3d.shape[-1]
            ii = L_pp3d._indices()[0]
            jj = L_pp3d._indices()[1]
            x = L_pp3d._values()
            solver = cholespy.CholeskySolverF(
                n_rows=nrows, ii=ii, jj=jj,
                x=x, type=cholespy.MatrixType.COO,
                pin_memory=True
            )
            solvers.append(solver)
            break
        except Exception as e:
            print(f"Retrying Cholesky solver for mesh due to error: {e}")
            eps = eps * 10.0
            sparse_eps_diag = torch.sparse.spdiags(eps * torch.ones(nV), torch.zeros(1, dtype=torch.long), (nV, nV)).to(device)
            L_pp3d = L_pp3d + sparse_eps_diag # (b, v, v)
            nretry += 1
    if nretry >= 5:
        raise RuntimeError(f"Failed to create Cholesky solver for mesh")
    vert_mass = mass_vec.unsqueeze(0).to(device) # (1, v)
    return solvers, vert_mass

class PoissonSolver(torch.autograd.Function):
    # Interface with external cholesky solver (Cholespy)
    @staticmethod
    def forward(ctx, solver, rhs: torch.Tensor):
        # Solve Lx = rhs
        ctx.solver = solver
        x = torch.zeros_like(rhs, device=rhs.device, dtype=rhs.dtype)
        solver.solve(rhs, x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        f_grad = None
        if ctx.needs_input_grad[1]:
            grad_output = grad_output.contiguous()
            f_grad = torch.zeros_like(grad_output)
            ctx.solver.solve(grad_output, f_grad)
        del ctx.solver
        return None, f_grad

def poisson_solve(rhs, solvers, vertex_mass):
    # Solve Poisson equation Lu = rhs
    u = torch.empty_like(rhs)
    B = rhs.shape[0]
    for b in range(B):
        # solver only accepts 128 simultaneous solves, so split channels if needed:
        if rhs.shape[-1] > 128:
            for j in range(0, rhs.shape[-1], 128):
                u[b][:, j:j+128] = PoissonSolver.apply(solvers[b], rhs[b][:, j:j+128].contiguous())
        else:
            u[b] = PoissonSolver.apply(solvers[b], rhs[b].contiguous())

    u = u - torch.sum(u * vertex_mass.unsqueeze(-1), dim=1, keepdim=True) / torch.sum(vertex_mass, dim=1, keepdim=True).unsqueeze(-1)

    return u

def sample_matern_noise(vertex_mass, solver, signal_dim=1, x0_sigma=1.0, device='cuda'):
    num_verts = vertex_mass.shape[1]
    random_signals = x0_sigma * torch.randn((1, num_verts, signal_dim), device=device)
    random_signals_mass_weighted = random_signals * torch.sqrt(vertex_mass.unsqueeze(-1))

    x0 = poisson_solve(random_signals_mass_weighted, solver, vertex_mass)

    return x0

def main():

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    args = parse_args()
    mesh_path = args.mesh
    screening_term = args.screening_term
    sigma = args.sigma

    device = 'cuda'

    mesh = trimesh.load(mesh_path, process=False)
    verts_np = mesh.vertices.astype(np.float64)
    faces_np = mesh.faces.astype(np.int64)

    solver, vertex_mass = construct_screened_solvers(
        verts_np=verts_np,
        faces_np=faces_np,
        screening_term=screening_term
    )

    x0 = sample_matern_noise(
        vertex_mass,
        solver,
        signal_dim=1,
        x0_sigma=sigma,
        device=device
    )

    print("Sampled matern noise shape:", x0.shape) # (1, N, 1)
    print('Sampled matern noise min:', x0.min().item())
    print('Sampled matern noise max:', x0.max().item())



if __name__ == "__main__":
    main()