import trimesh
import torch

import argparse

import cholespy
import torch_mesh_ops as TMO

def parse_args():
    parser = argparse.ArgumentParser(description='Sample Matern noise on a mesh')
    parser.add_argument('--mesh', type=str, default='path/to/your/mesh.obj', help='Path to input mesh file')
    parser.add_argument('--screening_term', type=float, default=100.0, help='Screening term')
    parser.add_argument('--sigma', type=float, default=5.0, help='Standard deviation of the sampled iid Gaussiannoise')
    args = parser.parse_args()
    return args

def construct_screened_solvers(V, F, screening_term=0, high_precision=False):
    '''
    Creates the following operators for a mesh. Uses PyTorch CUDA extension.
    - solver:       [B,]        list of Cholesky solvers for mesh's cotangent Laplacian

    Optionally set high_precision to True to use double precision in intermediate
    computations. Note the operators will be returned in float precision regardless.
    '''
    # ensure contiguous memory layout before calling CUDA extension
    V = V.contiguous()
    F = F.contiguous()
    if V.ndim == 2:
        V = V.unsqueeze(0)
        F = F.unsqueeze(0)
            
    if high_precision:
        V = V.double()

    nB, nV, _ = V.shape
    device = V.device

    vert_mass       = TMO.vertex_mass_batched(V, F, 1e-8)                     # (b, v)

    vert_mass = vert_mass.float()

    mass_vec = vert_mass.cpu()

    solvers = []
    Ls = TMO.cotangent_laplacian_batched(V, F, 1e-10).float() # (b, v, v)

    # Create Cholesky solver for each Laplacian in batch
    # This could be block-diagonalized, but speedup seems marginal
    for bi in range(nB):
        L = Ls[bi]
        eps = 1e-6
        if screening_term == 0:
            print('screening term is set to zero!')
            sparse_eps_diag = torch.sparse.spdiags(eps * torch.ones(nV), torch.zeros(1, dtype=torch.long), (nV, nV)).to(device)
        else:
            print(f'screening term: {screening_term}')
            sparse_eps_diag = torch.sparse.spdiags(screening_term * torch.ones(nV) * mass_vec[bi], torch.zeros(1, dtype=torch.long), (nV, nV)).to(device)
        L = L + sparse_eps_diag # (b, v, v)
        
        nretry = 0
        while nretry < 5:
            try:
                nrows = L.shape[-1]
                ii = L._indices()[0]
                jj = L._indices()[1]
                x = L._values()
                solver = cholespy.CholeskySolverF(
                    n_rows=nrows, ii=ii, jj=jj,
                    x=x, type=cholespy.MatrixType.COO
                )
                solvers.append(solver)
                break
            except Exception as e:
                print(f"Retrying Cholesky solver for mesh {bi} in batch due to error: {e}")
                eps = eps * 10.0
                sparse_eps_diag = torch.sparse.spdiags(eps * torch.ones(nV), torch.zeros(1, dtype=torch.long), (nV, nV)).to(device)
                L = L + sparse_eps_diag
                nretry += 1
        if nretry >= 5:
            raise RuntimeError(f"Failed to create Cholesky solver for mesh {bi} in batch")
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
    verts_tensor = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces_tensor = torch.tensor(mesh.faces, dtype=torch.long, device=device)

    solver, vertex_mass = construct_screened_solvers(
        V=verts_tensor,
        F=faces_tensor,
        screening_term=screening_term,
        high_precision=True
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