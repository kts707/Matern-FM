import os

import torch
import trimesh
import pytorch_lightning as pl

from utils.tools import (batchify_tensor, 
                         center_around_zeros_mass_weighted,
                         NJF_loss,
                         MSE_loss)

from models import poissonnet
from models.poissonnet.operators import construct_mesh_operators, construct_screened_solvers
from models.poissonnet.common import poisson_solve

from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper


def load_model(config, source_mesh=None):
    if config.model.type == 'Matern_FM':
        return Matern_FM(config, source_mesh=source_mesh)
    elif config.model.type == 'Matern_FM_arbitrary_source':
        return Matern_FM_arbitrary_source(config, source_mesh=source_mesh)
    elif config.model.type == 'eigenvecs_predictor':
        return Eigenvec_Predictor(config, source_mesh=source_mesh)
    else:
        raise NotImplementedError

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model.forward_generate(x, t)


class Matern_FM(pl.LightningModule):
    def __init__(self, config, source_mesh=None, device='cuda'):
        super().__init__()

        model_type = config.model.model_type
        self.model_type = model_type

        self.outputs_at = config.model.outputs_at
        
        self.solver = None

        if model_type == 'PoissonNet':

            # handle extra features
            self.extra_features_dim = 0

            # compute extra features dimensions
            self.extra_features = config.model.extra_features
            for feature in self.extra_features:
                if feature == 'v_source':
                    self.extra_features_dim += 3
                elif feature == 't':
                    self.extra_features_dim += 1
                else:
                    raise NotImplementedError("extra feature %s not supported" % (feature))

            self.use_source_xyz_grad_in = config.model.get('use_source_xyz_grad', True)

            self.base_model = poissonnet.PoissonNet(
                                            C_in=config.model.in_dim,
                                            C_out=config.model.out_dim,
                                            C_width=config.model.hidden_dim,
                                            n_blocks=config.model.num_blocks,
                                            head=config.model.head_type,
                                            extra_features=self.extra_features_dim,
                                            outputs_at=self.outputs_at,
                                            last_activation=torch.nn.Identity(),
                                            config=config.model.extra_configs)
            
            self.head_type = config.model.head_type

        else:
            raise NotImplementedError("model type not supported")
        
        # input dimensions: 3 for standard 3D meshes, can also be in arbitrary dimensions, representing vertex signals
        self.signal_dim = config.model.in_dim
        self.base_model.train()

        B = config.train.batch_size

        self.center_target_data = config.train.get('center_target_data', True)

        # source mesh
        mesh = trimesh.load(source_mesh, process=False)
        source_v, source_f = mesh.vertices, mesh.faces

        self.source_vertices = torch.tensor(source_v, dtype=torch.float32)
        self.source_faces = torch.tensor(source_f, dtype=torch.int64)
        # batchify tensors and operators
        self.batchify_tensors_fixed_source(B, device=device)

        self.lr = config.train.lr
        self.total_iterations = config.train.iterations
        self.batch_size = config.train.batch_size

        # conditiona probability path
        self.path = CondOTProbPath()

        # other configs
        self.config = config
        self.loss_type = config.train.loss_type

        self.ode_solver_config = config.ode_solver

        if self.loss_type == 'NJF_loss':
            self.lambda_v = config.train.loss_weights.lambda_v
            self.lambda_g = config.train.loss_weights.lambda_g
            self.mass_weighted_mse = config.train.loss_weights.mass_weighted
        else:
            raise NotImplementedError("loss type is not supported")
        
        x0_dist_config = config.model.get('x0_dist', {'type': 'poisson_solve'})
        self.x0_dist_type = x0_dist_config['type']
        if self.x0_dist_type == 'poisson_solve':
            self.x0_sigma = x0_dist_config.get('sigma', 1.0)


        self.screened_solver_for_x0 = x0_dist_config.get('screened_solver_for_x0', False)
        if self.screened_solver_for_x0:
            self.screening_term = x0_dist_config.get('screening_term', 1.0)
            self.solver_screened = construct_screened_solvers(
                self.v_source, self.source_faces_batch,
                screening_term=self.screening_term,
                high_precision=True,
                to_cpu=False
            )

    def batchify_operators_poisson_net_fixed_source(self, B, device='cuda'):

        if self.v_source is None or self.v_source.shape[0] != B:
            self.v_source = batchify_tensor(self.source_vertices, B).to(device)

        self.source_faces_batch = batchify_tensor(self.source_faces, B).to(device)

        # convert tensors to batched versions
        self.vertex_mass, self.solver, self.G, self.M = construct_mesh_operators(
            self.v_source, self.source_faces_batch, high_precision=True
        )

    def batchify_tensors_fixed_source(self, B, device='cuda'):

        self.v_source = batchify_tensor(self.source_vertices, B).to(device)

        if self.model_type == 'PoissonNet':
            self.batchify_operators_poisson_net_fixed_source(B, device=device)
        else:
            raise NotImplementedError("model type not supported")

    def configure_optimizers(self):
        params_list = list(self.base_model.parameters())

        optimizer = torch.optim.Adam(params_list, lr=self.lr)
        if self.config.train.lr_scheduler.type == 'MultiStepLR':
            milestones = self.config.train.lr_scheduler.milestones
            gamma = self.config.train.lr_scheduler.gamma
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma),
                'name': 'scheduler',
                'interval': 'step',
                }
        else:
            raise NotImplementedError
        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler
                }


    def sample_path(self, deformed_v, device='cuda'):
        t = torch.rand(deformed_v.shape[0]).to(device)
        x_1 = deformed_v
        x_0 = self.sample_x0_verts(deformed_v.shape[0], device=device)

        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)

        return path_sample

    def sample_x0_verts(self, num_samples, device='cuda'):
        if self.x0_dist_type == 'poisson_solve':
            num_verts = self.vertex_mass.shape[1]
            random_signals = self.x0_sigma * torch.randn((num_samples, num_verts, self.signal_dim), device=device)
            random_signals_mass_weighted = random_signals * torch.sqrt(self.vertex_mass.unsqueeze(-1))

            if self.screened_solver_for_x0:
                x0 = poisson_solve(random_signals_mass_weighted, self.solver_screened, self.vertex_mass)
            else:
                x0 = poisson_solve(random_signals_mass_weighted, self.solver, self.vertex_mass)
        else:
            raise NotImplementedError("x0 distribution type %s not supported" % (self.x0_dist_type))

        # center the x0 around zeros
        x0 = center_around_zeros_mass_weighted(x0, self.vertex_mass, dimension=1)
        return x0

    def forward(self, x_t, t):
        
        num_verts = self.vertex_mass.shape[1]
        t = t.reshape(-1, 1).repeat(1, num_verts).unsqueeze(-1)
        if t.shape[0] != x_t.shape[0]:
            t = t.repeat(x_t.shape[0], 1, 1)

        x_in = x_t
        features_list = []
        for feature in self.extra_features:
            if feature == 'v_source':
                features_list.append(self.v_source)
            elif feature == 't':
                features_list.append(t)
            else:
                raise NotImplementedError("extra feature %s not supported" % (feature))
        
        extra_features = torch.cat(features_list, dim=-1)

        if self.head_type == 'njf':
            if self.use_source_xyz_grad_in:
                grad_in = torch.bmm(self.G, self.v_source)
            else:
                grad_in = None
            pred_v, pred_j = self.base_model(x_in=x_in,
                                            M=self.M, 
                                            G=self.G, 
                                            solver=self.solver, 
                                            faces=self.source_faces_batch, 
                                            vertex_mass=self.vertex_mass,
                                            extra_features=extra_features,
                                            grad_in=grad_in)
            return pred_v, pred_j
        else:
            raise NotImplementedError("head type %s not supported" % (self.head_type))


    def _save_checkpoint(self, path):
        self.trainer.save_checkpoint(path)

    def forward_generate(self, x_t, t):
        pred_v, _ = self.forward(x_t, t)

        return pred_v

    def training_step(self, batch, batch_idx):
        deformed_v = batch
        if deformed_v.shape[0] != self.v_source.shape[0]:
            self.batchify_tensors_fixed_source(deformed_v.shape[0], device=deformed_v.device)
        if self.center_target_data:
            # mass weighted centering
            deformed_v = center_around_zeros_mass_weighted(deformed_v, self.vertex_mass, dimension=1)

        path_sample = self.sample_path(deformed_v, device=deformed_v.device)

        pred_v, pred_j = self.forward(path_sample.x_t, path_sample.t)
        
        # loss functions
        if self.loss_type == 'NJF_loss':
            loss_v, loss_g = NJF_loss(pred_v, pred_j, path_sample.dx_t, self.G, self.vertex_mass, self.M, mass_weighted=self.mass_weighted_mse)
            loss = self.lambda_v * loss_v + self.lambda_g * loss_g
            self.log("vertex_loss", loss_v, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log("jacobian_loss", loss_g, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        else:
            raise NotImplementedError("loss type is not supported")

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        self.log("lr", self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)

        return loss


    def generate_fixed_source(self, saving_dir, num_samples=1, device='cuda', save_intermediates=False, evolution_dir=None, init_x0=None, repeat_idx=0):
        # set to eval mode
        self.base_model.eval()

        # batchify tensors and operators
        self.batchify_tensors_fixed_source(num_samples, device=device)

        if self.screened_solver_for_x0:
            self.solver_screened = construct_screened_solvers(
                self.v_source, self.source_faces_batch,
                screening_term=self.screening_term,
                high_precision=True,
                to_cpu=False
            )

        with torch.no_grad():
            self.base_model.to(device)
            wrapped_vf = WrappedModel(self)
            solver = ODESolver(velocity_model=wrapped_vf)

            method = self.ode_solver_config.method
            step_size = self.ode_solver_config.step_size

            if init_x0 is not None:
                x_init = init_x0
            else:
                x_init = self.sample_x0_verts(num_samples, device=device)

            if save_intermediates:
                assert evolution_dir is not None
                time_grid = torch.linspace(0.0, 1.0, int(1/step_size)+1, device=device)
                sol = solver.sample(time_grid=time_grid, x_init=x_init, method=method, step_size=step_size, return_intermediates=True)

                for i in range(num_samples):
                    os.makedirs(os.path.join(evolution_dir, 'deformed_mesh_%04d' % (i + repeat_idx*num_samples)), exist_ok=True)
                    for t_idx in range(time_grid.shape[0]):
                        mesh = trimesh.Trimesh(vertices=sol[t_idx][i].cpu().detach().numpy(), faces=self.source_faces.numpy(), process=False)
                        mesh.export(os.path.join(evolution_dir, 'deformed_mesh_%04d' % (i + repeat_idx*num_samples), 'time_%04d.obj' % (t_idx)))
                    mesh = trimesh.Trimesh(vertices=sol[-1][i].cpu().detach().numpy(), faces=self.source_faces.numpy(), process=False)
                    mesh.export(os.path.join(saving_dir, 'deformed_mesh_%04d.obj' % (i + repeat_idx*num_samples)))                

            else:
                time_grid = torch.tensor([0.0, 1.0], device=device)

                sol = solver.sample(time_grid=time_grid, x_init=x_init, method=method, step_size=step_size, return_intermediates=False)
                for i in range(num_samples):
                    mesh = trimesh.Trimesh(vertices=sol[i].cpu().detach().numpy(), faces=self.source_faces.numpy(), process=False)
                    mesh.export(os.path.join(saving_dir, 'deformed_mesh_%04d.obj' % (i + repeat_idx*num_samples)))

        self.base_model.train()

        # batchify tensors and operators
        self.batchify_tensors_fixed_source(self.batch_size, device=device)

        if self.screened_solver_for_x0:
            self.solver_screened = construct_screened_solvers(
                self.v_source, self.source_faces_batch,
                screening_term=self.screening_term,
                high_precision=True,
                to_cpu=False
            )



class Matern_FM_arbitrary_source(pl.LightningModule):
    def __init__(self, config, source_mesh=None, device='cuda'):
        super().__init__()

        model_type = config.model.model_type
        self.model_type = model_type

        self.outputs_at = config.model.outputs_at
        self.use_cached_solvers = config.dataset.get('cache_solvers', False)
        self.solver = None

        if model_type == 'PoissonNet':

            # handle extra features
            self.extra_features_dim = 0

            self.use_predicted_eigenvecs = config.model.get('use_predicted_eigenvecs', True)

            # compute extra features dimensions
            self.extra_features = config.model.extra_features
            for feature in self.extra_features:
                if feature == 'v_source':
                    self.extra_features_dim += 3
                elif feature == 't':
                    self.extra_features_dim += 1
                else:
                    raise NotImplementedError("extra feature %s not supported" % (feature))
            
            if self.use_predicted_eigenvecs:
                self.extra_features_dim += config.model.eigenvecs_predictor.out_dim

            self.use_source_xyz_grad_in = config.model.get('use_source_xyz_grad', True)

            self.base_model = poissonnet.PoissonNet(
                                            C_in=config.model.in_dim,
                                            C_out=config.model.out_dim,
                                            C_width=config.model.hidden_dim,
                                            n_blocks=config.model.num_blocks,
                                            head=config.model.head_type,
                                            extra_features=self.extra_features_dim,
                                            outputs_at=self.outputs_at,
                                            last_activation=torch.nn.Identity(),
                                            config=config.model.extra_configs)
            
            self.head_type = config.model.head_type

            if self.use_predicted_eigenvecs:
                eigenvecs_predictor_configs = config.model.get('eigenvecs_predictor', {})
                self.eigenvecs_predictor = poissonnet.PoissonNet(
                                            C_in=eigenvecs_predictor_configs.in_dim,
                                            C_out=eigenvecs_predictor_configs.out_dim,
                                            C_width=eigenvecs_predictor_configs.hidden_dim,
                                            n_blocks=eigenvecs_predictor_configs.num_blocks,
                                            head=eigenvecs_predictor_configs.head_type,
                                            extra_features=0,
                                            outputs_at=eigenvecs_predictor_configs.outputs_at,
                                            last_activation=torch.nn.Identity(),
                                            config=eigenvecs_predictor_configs.extra_configs)

                eigenvecs_predictor_ckpt_path = config.model.eigenvecs_predictor.get('checkpoint_path', None)
                assert eigenvecs_predictor_ckpt_path is not None, "eigenvecs predictor checkpoint path must be provided"
                checkpoint_state_dict = torch.load(eigenvecs_predictor_ckpt_path)['state_dict']
                prefix = 'base_model.'
                state_dict = {k[len(prefix):]: v for k, v in checkpoint_state_dict.items() if k.startswith(prefix)}
                self.eigenvecs_predictor.load_state_dict(state_dict)
                self.eigenvecs_predictor.eval()
                for param in self.eigenvecs_predictor.parameters():
                    param.requires_grad = False

        else:
            raise NotImplementedError("model type not supported")
        
        # input dimensions: 3 for standard 3D meshes, can also be in arbitrary dimensions, representing vertex signals
        self.signal_dim = config.model.in_dim
        self.base_model.train()

        B = config.train.batch_size

        self.center_target_data = config.train.get('center_target_data', True)

        # source mesh
        mesh = trimesh.load(source_mesh, process=False)
        source_f = mesh.faces

        self.source_faces = torch.tensor(source_f, dtype=torch.int64)
        self.source_faces_batch = batchify_tensor(self.source_faces, B).to(device)

        self.lr = config.train.lr
        self.total_iterations = config.train.iterations
        self.batch_size = config.train.batch_size

        # conditiona probability path
        self.path = CondOTProbPath()

        # other configs
        self.config = config
        self.loss_type = config.train.loss_type

        self.ode_solver_config = config.ode_solver

        if self.loss_type == 'NJF_loss':
            self.lambda_v = config.train.loss_weights.lambda_v
            self.lambda_g = config.train.loss_weights.lambda_g
            self.mass_weighted_mse = config.train.loss_weights.mass_weighted
        else:
            raise NotImplementedError("loss type is not supported")
        
        x0_dist_config = config.model.get('x0_dist', {'type': 'poisson_solve'})
        self.x0_dist_type = x0_dist_config['type']
        if self.x0_dist_type == 'poisson_solve':
            self.x0_sigma = x0_dist_config.get('sigma', 1.0)


        self.screened_solver_for_x0 = x0_dist_config.get('screened_solver_for_x0', False)
        if self.screened_solver_for_x0:
            self.screening_term = x0_dist_config.get('screening_term', 1.0)
            self.solver_screened = None

    def batchify_operators_poisson_net_arbitrary_source(self, B, source_v, apply_to_faces=True, device='cuda', use_cached_solvers=False):

        if apply_to_faces or self.source_faces_batch.shape[0] != B:
            self.source_faces_batch = batchify_tensor(self.source_faces, B).to(device)

        # convert tensors to batched versions
        self.vertex_mass, self.solver, self.G, self.M = construct_mesh_operators(
            source_v, self.source_faces_batch, high_precision=True, create_solvers=not use_cached_solvers
        )

    def batchify_tensors_arbitrary_source(self, B, source_v, apply_to_faces=True, device='cuda', use_cached_solvers=False):

        self.v_source = source_v

        # assuming source_v is already in batched form
        if self.model_type == 'PoissonNet':
            self.batchify_operators_poisson_net_arbitrary_source(B, source_v, apply_to_faces=apply_to_faces, device=device, use_cached_solvers=use_cached_solvers)
        else:
            raise NotImplementedError("model type not supported")

    def configure_optimizers(self):
        params_list = list(self.base_model.parameters())

        optimizer = torch.optim.Adam(params_list, lr=self.lr)
        if self.config.train.lr_scheduler.type == 'MultiStepLR':
            milestones = self.config.train.lr_scheduler.milestones
            gamma = self.config.train.lr_scheduler.gamma
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma),
                'name': 'scheduler',
                'interval': 'step',
                }
        else:
            raise NotImplementedError
        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler
                }


    def sample_path(self, deformed_v, device='cuda'):
        t = torch.rand(deformed_v.shape[0]).to(device)
        x_1 = deformed_v
        x_0 = self.sample_x0_verts(deformed_v.shape[0], device=device)

        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)

        return path_sample

    def sample_x0_verts(self, num_samples, device='cuda'):
        if self.x0_dist_type == 'poisson_solve':
            num_verts = self.vertex_mass.shape[1]
            random_signals = self.x0_sigma * torch.randn((num_samples, num_verts, self.signal_dim), device=device)
            random_signals_mass_weighted = random_signals * torch.sqrt(self.vertex_mass.unsqueeze(-1))

            if self.screened_solver_for_x0:
                x0 = poisson_solve(random_signals_mass_weighted, self.solver_screened, self.vertex_mass)
            else:
                x0 = poisson_solve(random_signals_mass_weighted, self.solver, self.vertex_mass)
        else:
            raise NotImplementedError("x0 distribution type %s not supported" % (self.x0_dist_type))

        # center the x0 around zeros
        x0 = center_around_zeros_mass_weighted(x0, self.vertex_mass, dimension=1)
        return x0

    def forward(self, x_t, t):
        
        num_verts = self.vertex_mass.shape[1]
        t = t.reshape(-1, 1).repeat(1, num_verts).unsqueeze(-1)
        if t.shape[0] != x_t.shape[0]:
            t = t.repeat(x_t.shape[0], 1, 1)

        x_in = x_t
        features_list = []
        for feature in self.extra_features:
            if feature == 'v_source':
                features_list.append(self.v_source)
            elif feature == 't':
                features_list.append(t)
            else:
                raise NotImplementedError("extra feature %s not supported" % (feature))
        
        if self.use_predicted_eigenvecs:
            # get the predicted eigenvectors
            with torch.no_grad():
                source_mesh_eigenvecs = self.eigenvecs_predictor(x_in=self.v_source,
                                                        M=self.M, 
                                                        G=self.G, 
                                                        solver=self.solver, 
                                                        faces=self.source_faces_batch, 
                                                        vertex_mass=self.vertex_mass,
                                                        extra_features=None,
                                                        grad_in=None)
                
                features_list.append(source_mesh_eigenvecs)

        extra_features = torch.cat(features_list, dim=-1)

        if self.head_type == 'njf':
            if self.use_source_xyz_grad_in:
                grad_in = torch.bmm(self.G, self.v_source)
            else:
                grad_in = None
            pred_v, pred_j = self.base_model(x_in=x_in,
                                            M=self.M, 
                                            G=self.G, 
                                            solver=self.solver, 
                                            faces=self.source_faces_batch, 
                                            vertex_mass=self.vertex_mass,
                                            extra_features=extra_features,
                                            grad_in=grad_in)
            return pred_v, pred_j
        else:
            raise NotImplementedError("head type %s not supported" % (self.head_type))


    def _save_checkpoint(self, path):
        self.trainer.save_checkpoint(path)

    def forward_generate(self, x_t, t):
        pred_v, _ = self.forward(x_t, t)

        return pred_v

    def training_step(self, batch, batch_idx):
        if self.use_cached_solvers:
            if self.screened_solver_for_x0:
                # move old solvers back to cpu
                if self.solver is not None:
                    for solver in self.solver:
                        solver.to_cpu()
                if self.solver_screened is not None:
                    for solver in self.solver_screened:
                        solver.to_cpu()
                (source_v, deformed_v, _,), solvers, solvers_screened = batch
                for i in range(len(solvers)):
                    solvers[i].to_gpu()
                for i in range(len(solvers_screened)):
                    solvers_screened[i].to_gpu()
                self.batchify_tensors_arbitrary_source(source_v.shape[0], source_v, apply_to_faces=False, device=deformed_v.device, use_cached_solvers=True)
                self.solver = solvers
                self.solver_screened = solvers_screened
            else:
                # move old solvers back to cpu
                if self.solver is not None:
                    for solver in self.solver:
                        solver.to_cpu()
                (source_v, deformed_v, _,), solvers = batch
                for i in range(len(solvers)):
                    solvers[i].to_gpu()
                self.batchify_tensors_arbitrary_source(source_v.shape[0], source_v, apply_to_faces=False, device=deformed_v.device, use_cached_solvers=True)
                self.solver = solvers
        else:
            source_v, deformed_v, _ = batch
            self.batchify_tensors_arbitrary_source(source_v.shape[0], source_v, apply_to_faces=False, device=deformed_v.device, use_cached_solvers=False)
            if self.screened_solver_for_x0:
                self.solver_screened = construct_screened_solvers(self.v_source, self.source_faces_batch, screening_term=self.screening_term, high_precision=True, to_cpu=False)        
        
        if self.center_target_data:
            # mass weighted centering
            deformed_v = center_around_zeros_mass_weighted(deformed_v, self.vertex_mass, dimension=1)

        path_sample = self.sample_path(deformed_v, device=deformed_v.device)

        pred_v, pred_j = self.forward(path_sample.x_t, path_sample.t)
        
        # loss functions
        if self.loss_type == 'NJF_loss':
            loss_v, loss_g = NJF_loss(pred_v, pred_j, path_sample.dx_t, self.G, self.vertex_mass, self.M, mass_weighted=self.mass_weighted_mse)
            loss = self.lambda_v * loss_v + self.lambda_g * loss_g
            self.log("vertex_loss", loss_v, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log("jacobian_loss", loss_g, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        else:
            raise NotImplementedError("loss type is not supported")

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        self.log("lr", self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)

        return loss


    def generate_arbitrary_source(self, source_v, saving_dir_deformed, device='cuda', save_intermediates=False, evolution_dir=None, init_x0=None, repeat_idx=0):
        # set to eval mode
        self.base_model.eval()

        # batchify tensors and operators
        self.batchify_tensors_arbitrary_source(source_v.shape[0], source_v, apply_to_faces=True, device=device, use_cached_solvers=False)

        if self.screened_solver_for_x0:
            self.solver_screened = construct_screened_solvers(self.v_source, self.source_faces_batch, screening_term=self.screening_term, high_precision=True, to_cpu=False)

        num_samples = source_v.shape[0]

        method = self.ode_solver_config.method
        step_size = self.ode_solver_config.step_size


        with torch.no_grad():
            self.base_model.to(device)
            if self.use_predicted_eigenvecs:
                self.eigenvecs_predictor.to(device)

            wrapped_vf = WrappedModel(self)
            solver = ODESolver(velocity_model=wrapped_vf)
                    

            if init_x0 is not None:
                x_init = init_x0
            else:
                x_init = self.sample_x0_verts(num_samples, device=device)

            if save_intermediates:
                assert evolution_dir is not None
                time_grid = torch.linspace(0.0, 1.0, int(1/step_size)+1, device=device)
                sol = solver.sample(time_grid=time_grid, x_init=x_init, method=method, step_size=step_size, return_intermediates=True)

                for i in range(num_samples):
                    os.makedirs(os.path.join(evolution_dir, 'deformed_mesh_%04d_%04d' % (i, repeat_idx)), exist_ok=True)
                    for t_idx in range(time_grid.shape[0]):
                        mesh = trimesh.Trimesh(vertices=sol[t_idx][i].cpu().detach().numpy(), faces=self.source_faces.numpy(), process=False)
                        mesh.export(os.path.join(evolution_dir, 'deformed_mesh_%04d_%04d' % (i, repeat_idx), 'time_%04d.obj' % (t_idx)))

                    mesh = trimesh.Trimesh(vertices=sol[-1][i].cpu().detach().numpy(), faces=self.source_faces.numpy(), process=False)
                    mesh.export(os.path.join(saving_dir_deformed, 'deformed_mesh_%04d_%04d.obj' % (i, repeat_idx)))

            else:
                time_grid = torch.tensor([0.0, 1.0], device=device)

                sol = solver.sample(time_grid=time_grid, x_init=x_init, method=method, step_size=step_size, return_intermediates=False)

                for i in range(num_samples):
                    mesh = trimesh.Trimesh(vertices=sol[i].cpu().detach().numpy(), faces=self.source_faces.numpy(), process=False)
                    mesh.export(os.path.join(saving_dir_deformed, 'deformed_mesh_%04d_%04d.obj' % (i, repeat_idx)))

        if self.screened_solver_for_x0:
            # move solvers back to cpu
            for s in self.solver_screened:
                s.to_cpu()

        self.base_model.train()

        self.source_faces_batch = batchify_tensor(self.source_faces, self.batch_size).to(device)


    def generate_arbitrary_source_test(self, source_v, source_f, saving_dir_deformed, device='cuda', save_intermediates=False, evolution_dir=None, init_x0=None, repeat_idx=0, mesh_filenames=None):
        # set to eval mode
        self.base_model.eval()

        method = self.ode_solver_config.method
        step_size = self.ode_solver_config.step_size

        num_samples = len(source_v)

        with torch.no_grad():
            self.base_model.to(device)
            if self.use_predicted_eigenvecs:
                self.eigenvecs_predictor.to(device)

            wrapped_vf = WrappedModel(self)
            solver = ODESolver(velocity_model=wrapped_vf)

            for sample_idx in range(num_samples):
                filename = mesh_filenames[sample_idx] if mesh_filenames is not None else sample_idx
                print('mesh idx', sample_idx, 'filename', filename)
                source_vertices = source_v[sample_idx].unsqueeze(0)
                self.source_faces = source_f[sample_idx]
                self.source_faces_batch = self.source_faces.unsqueeze(0)


                self.batchify_tensors_arbitrary_source(source_vertices.shape[0], source_vertices, device=device, use_cached_solvers=False)

                if self.solver == []:
                    print("Warning: solvers list is empty, skipping current source mesh")
                    continue

                if self.screened_solver_for_x0:
                    self.solver_screened = construct_screened_solvers(self.v_source, self.source_faces_batch, screening_term=self.screening_term, high_precision=True, to_cpu=False)


                if init_x0 is not None:
                    x_init = init_x0[sample_idx:sample_idx+1]
                else:
                    x_init = self.sample_x0_verts(1, device=device)

                if save_intermediates:
                    assert evolution_dir is not None
                    time_grid = torch.linspace(0.0, 1.0, int(1/step_size)+1, device=device)
                    sol = solver.sample(time_grid=time_grid, x_init=x_init, method=method, step_size=step_size, return_intermediates=True)

                    os.makedirs(os.path.join(evolution_dir, '%s_%04d_%04d' % (filename, sample_idx, repeat_idx)), exist_ok=True)
                    for t_idx in range(time_grid.shape[0]):
                        mesh = trimesh.Trimesh(vertices=sol[t_idx][0].cpu().detach().numpy(), faces=self.source_faces.cpu().numpy(), process=False)
                        mesh.export(os.path.join(evolution_dir, '%s_%04d_%04d' % (filename, sample_idx, repeat_idx), 'time_%04d.obj' % (t_idx)))


                    mesh = trimesh.Trimesh(vertices=sol[-1][0].cpu().detach().numpy(), faces=self.source_faces.cpu().numpy(), process=False)
                    mesh.export(os.path.join(saving_dir_deformed, '%s_%04d_%04d.obj' % (filename, sample_idx, repeat_idx)))

                else:
                    time_grid = torch.tensor([0.0, 1.0], device=device)

                    sol = solver.sample(time_grid=time_grid, x_init=x_init, method=method, step_size=step_size, return_intermediates=False)

                    mesh = trimesh.Trimesh(vertices=sol[0].cpu().detach().numpy(), faces=self.source_faces.cpu().numpy(), process=False)
                    mesh.export(os.path.join(saving_dir_deformed, '%s_%04d_%04d.obj' % (filename, sample_idx, repeat_idx)))

                if self.screened_solver_for_x0:
                    # move solvers back to cpu and delete them
                    for s in self.solver_screened:
                        s.to_cpu()



class Eigenvec_Predictor(pl.LightningModule):
    def __init__(self, config, single_source=False, source_mesh=None, device='cuda'):
        super().__init__()

        model_type = config.model.model_type
        self.model_type = model_type

        self.outputs_at = config.model.outputs_at

        self.target_centering = config.model.get('target_centering', False)

        if model_type == 'PoissonNet':

            self.base_model = poissonnet.PoissonNet(
                                            C_in=config.model.in_dim,
                                            C_out=config.model.out_dim,
                                            C_width=config.model.hidden_dim,
                                            n_blocks=config.model.num_blocks,
                                            head=config.model.head_type,
                                            extra_features=0,
                                            outputs_at=self.outputs_at,
                                            last_activation=torch.nn.Identity(),
                                            config=config.model.extra_configs)
            
            self.head_type = config.model.head_type

        else:
            raise NotImplementedError("model type not supported")
        
        # input dimensions: 3 for standard 3D meshes, can also be in arbitrary dimensions, representing vertex signals
        self.signal_dim = config.model.in_dim
        self.base_model.train()

        B = config.train.batch_size

        # source mesh
        mesh = trimesh.load(source_mesh, process=False)
        source_f = mesh.faces
        self.source_faces = torch.tensor(source_f, dtype=torch.int64)
        self.source_faces_batch = batchify_tensor(self.source_faces, B).to(device)

        self.lr = config.train.lr
        self.total_iterations = config.train.iterations
        self.batch_size = config.train.batch_size

        # other configs
        self.config = config

        self.use_cached_solvers = config.dataset.get('cache_solvers', False) and not single_source
        self.solver = None

        self.mass_weighted_mse = config.train.loss_weights.mass_weighted

    def batchify_operators_poisson_net_arbitrary_source(self, B, source_v, apply_to_faces=True, device='cuda', use_cached_solvers=False):

        if apply_to_faces or self.source_faces_batch.shape[0] != B:
            self.source_faces_batch = batchify_tensor(self.source_faces, B).to(device)

        # convert tensors to batched versions
        self.vertex_mass, self.solver, self.G, self.M = construct_mesh_operators(
            source_v, self.source_faces_batch, high_precision=True, create_solvers=not use_cached_solvers
        )

    def batchify_tensors_arbitrary_source(self, B, source_v, apply_to_faces=True, device='cuda', use_cached_solvers=False):

        self.v_source = source_v

        # assuming source_v is already in batched form
        if self.model_type == 'PoissonNet':
            self.batchify_operators_poisson_net_arbitrary_source(B, source_v, apply_to_faces=apply_to_faces, device=device, use_cached_solvers=use_cached_solvers)
        else:
            raise NotImplementedError("model type not supported")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=self.lr)

        if self.config.train.lr_scheduler.type == 'MultiStepLR':
            milestones = self.config.train.lr_scheduler.milestones
            gamma = self.config.train.lr_scheduler.gamma
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma),
                'name': 'scheduler',
                'interval': 'step',
                }
        else:
            raise NotImplementedError
        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler
                }


    def forward(self, x):
        if self.head_type == 'linear' or self.head_type == 'mlp':
            grad_in = None
            pred = self.base_model(x_in=x,
                                    M=self.M, 
                                    G=self.G, 
                                    solver=self.solver, 
                                    faces=self.source_faces_batch, 
                                    vertex_mass=self.vertex_mass,
                                    extra_features=None,
                                    grad_in=grad_in)
        else:
            raise NotImplementedError("head type %s not supported" % (self.head_type))
        
        return pred


    def training_step(self, batch, batch_idx):
        if self.use_cached_solvers:
            # move old solvers back to cpu
            if self.solver is not None:
                for solver in self.solver:
                    solver.to_cpu()
            (source_v, deformed_v, _), solvers = batch
            for i in range(len(solvers)):
                solvers[i].to_gpu()
            self.batchify_tensors_arbitrary_source(source_v.shape[0], source_v, apply_to_faces=False, device=deformed_v.device, use_cached_solvers=True)
            self.solver = solvers
        else:
            source_v, deformed_v, _ = batch
            self.batchify_tensors_arbitrary_source(source_v.shape[0], source_v, apply_to_faces=False, device=deformed_v.device, use_cached_solvers=False)
        if self.target_centering:
            deformed_v = center_around_zeros_mass_weighted(deformed_v, self.vertex_mass, dimension=1)

        pred = self.forward(self.v_source)

        loss = MSE_loss(pred, deformed_v, mass=self.vertex_mass, mass_weighted=self.mass_weighted_mse)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        self.log("lr", self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)

        return loss