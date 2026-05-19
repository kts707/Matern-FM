import os
import random

import torch
from torch.utils.data import Dataset, default_collate

import trimesh

def load_dataset(configs, source_saving_dir):
    if configs.name == 'single_source':
        return Single_Source(source_saving_dir, configs)
    elif configs.name == 'arbitrary_source':
        return Arbitrary_Source(source_saving_dir, configs)
    elif configs.name == 'mesh_to_eigenvecs':
        return Mesh_to_Eigenvecs(source_saving_dir, configs)
    else:
        raise NotImplementedError

class Single_Source(Dataset):
    def __init__(self, source_saving_dir, configs):

        self.fixed_source = True

        data_file = configs.data_file
        data_set = torch.load(data_file)

        '''
        keys: 
        'src_verts', # (V, 3)
        'tar_verts', # (N, V, 3)
        'faces', # (F, 3)
        '''

        self.source_mesh_path = os.path.join(source_saving_dir, 'base.obj')
        self.faces = data_set['faces']

        source_v = data_set['src_verts']

        source_mesh = trimesh.Trimesh(vertices=source_v.numpy(), faces=self.faces.numpy(), process=False)
        source_mesh.export(self.source_mesh_path)

        use_subset = configs.get("use_subset", False)
        if use_subset:
            subset_percentage = configs.subset_percentage

        if use_subset:
            k = max(1, int(subset_percentage * data_set['tar_verts'].shape[0]))
            idx = torch.randperm(data_set['tar_verts'].shape[0])[:k]
            self.deformed_vertices = data_set['tar_verts'][idx]
        else:
            self.deformed_vertices = data_set['tar_verts']

        self.dataset_size = self.deformed_vertices.shape[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        deformed_v = self.deformed_vertices[idx]
        return deformed_v

    def collate_fn(self, batch):
        return default_collate(batch)
    

class Arbitrary_Source(Dataset):
    def __init__(self, source_saving_dir, configs):

        data_file = configs.data_file
        data_set = torch.load(data_file)

        '''
        keys: 
        'src_verts', # (N, V, 3)
        'tar_verts', # (N, V, 3)
        'faces', # (F, 3)
        '''

        use_subset = configs.get("use_subset", False)
        if use_subset:
            subset_percentage = configs.subset_percentage

        self.use_data_augmentation = configs.get("use_data_augmentation", True)

        if use_subset:
            k = max(1, int(subset_percentage * data_set['tar_verts'].shape[0]))
            idx = torch.randperm(data_set['tar_verts'].shape[0])[:k]
            self.deformed_vertices = data_set['tar_verts'][idx]
            self.source_vertices = data_set['src_verts'][idx]
        else:
            self.deformed_vertices = data_set['tar_verts']
            self.source_vertices = data_set['src_verts']

        # save a source base mesh such that we can load mesh faces in model
        self.source_mesh_path = os.path.join(source_saving_dir, 'base.obj')
        self.faces = data_set['faces']

        source_v = self.source_vertices[0]
        source_mesh = trimesh.Trimesh(vertices=source_v.numpy(), faces=self.faces.numpy(), process=False)
        source_mesh.export(self.source_mesh_path)

        self.dataset_size = self.deformed_vertices.shape[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        source_v = self.source_vertices[idx]
        deformed_v = self.deformed_vertices[idx]

        if self.use_data_augmentation:
            source_v, deformed_v = self.data_augmentation(source_v, deformed_v)

        return source_v, deformed_v, idx

    def data_augmentation(self, verts_src, verts_tar):
        scale_xyz = torch.rand(1, 1) * 0.6 + 0.7 # (1, 1)
        verts_src = verts_src * scale_xyz
        verts_tar = verts_tar * scale_xyz

        shift_xyz = torch.randn(1, 3, device=verts_src.device) * 0.15 # (1, 3)
        verts_src = verts_src + shift_xyz
        verts_tar = verts_tar + shift_xyz

        return verts_src, verts_tar

    def collate_fn(self, batch):
        return default_collate(batch)

    def sample_random_sources(self, num_samples, device='cuda', saving_dir=None):
        batch_source = []
        faces = self.faces.numpy()
        for i in range(num_samples):
            random_idx = random.randint(0, self.source_vertices.shape[0] - 1)
            print('------------------random source mesh idx------------------', i, 'data index:', random_idx)
            source_v = self.source_vertices[random_idx]

            batch_source.append(source_v)

        if saving_dir is not None:
            os.makedirs(saving_dir, exist_ok=True)
            for idx, source_v in enumerate(batch_source):
                saving_path = os.path.join(saving_dir, f'source_mesh_{idx:04d}.obj')
                mesh = trimesh.Trimesh(vertices=source_v.numpy(), faces=faces, process=False)
                mesh.export(saving_path)

        batch_source = torch.stack(batch_source, dim=0).to(device)
        return batch_source        


class Mesh_to_Eigenvecs(Dataset):
    def __init__(self, source_saving_dir, configs):

        # load the target eigenvectors
        eigenvecs_file = configs.eigenvecs
        self.target_eigenvecs = torch.load(eigenvecs_file).squeeze(0)
        self.target_eigenvecs = self.target_eigenvecs[:, :configs.use_first_k_eigenvec]


        data_file = configs.data_file
        data_set = torch.load(data_file)

        '''
        keys: 
        'src_verts', # (N, V, 3)
        'tar_verts', # (N, V, 3)
        'faces', # (F, 3)
        '''

        use_subset = configs.get("use_subset", False)
        if use_subset:
            subset_percentage = configs.subset_percentage

        self.use_data_augmentation = configs.get("use_data_augmentation", True)

        # Use arbitrary source meshes as source
        if use_subset:
            k = max(1, int(subset_percentage * data_set['src_verts'].shape[0]))
            idx = torch.randperm(data_set['tar_verts'].shape[0])[:k]
            self.deformed_vertices = data_set['tar_verts'][idx]
            self.source_vertices = data_set['src_verts'][idx]
        else:
            self.deformed_vertices = data_set['tar_verts']
            self.source_vertices = data_set['src_verts']

        self.faces = data_set['faces']

        # save a source base mesh such that we can load mesh faces in model
        self.source_mesh_path = os.path.join(source_saving_dir, 'base.obj')

        source_v = data_set['src_verts'][0]
        source_mesh = trimesh.Trimesh(vertices=source_v.numpy(), faces=self.faces.numpy(), process=False)
        source_mesh.export(self.source_mesh_path)

        # use all meshes as source
        self.source_vertices = torch.concat([self.source_vertices, self.deformed_vertices], dim=0)

        self.dataset_size = self.source_vertices.shape[0]


    def data_augmentation(self, verts_src):
        scale_xyz = torch.rand(1, 1) * 0.6 + 0.7 # (1, 1)
        verts_src = verts_src * scale_xyz

        shift_xyz = torch.randn(1, 3, device=verts_src.device) * 0.15 # (1, 3)
        verts_src = verts_src + shift_xyz

        return verts_src

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        source_v = self.source_vertices[idx]
        if self.use_data_augmentation:
            source_v = self.data_augmentation(source_v)

        return source_v, self.target_eigenvecs, idx

    def collate_fn(self, batch):
        return default_collate(batch)