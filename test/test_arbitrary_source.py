import sys
import os
import argparse

import torch
import pytorch_lightning as pl

import shutil

from omegaconf import OmegaConf

import trimesh

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from model import load_model
from utils.render import render_trajectory, render_trajectory_to_multiple_dirs

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--config", type=str, required=True, help="Path to exp_dir")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--source_dir", type=str, default=None, help="source meshes dir (optional)")
    parser.add_argument("--output_dir", type=str, default="generated_meshes", help="Where to write meshes")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times repeated")
    parser.add_argument("--render", action='store_true', help="Render results using polyscope")
    parser.add_argument("--R", type=float, default=2.5, help="Camera radius for rendering")
    parser.add_argument("--h", type=float, default=1.5, help="Camera height for rendering")

    args = parser.parse_args()
    return args

def main():

    args = get_parser()

    config = OmegaConf.load(args.config)

    config.train.batch_size = 1

    device = "cuda"

    # set seeds and device
    pl.seed_everything(args.seed)

    # load checkpoint
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        output_dir = args.output_dir
    else:
        results_dir = os.path.join('results', config.group, config.exp_name)
        output_dir = os.path.join(results_dir, args.output_dir)

        checkpoint_dir = os.path.join(results_dir, 'checkpoints')
        checkpoint_files = os.listdir(checkpoint_dir)
        assert len(checkpoint_files) == 1

        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])

    # evaluation and saving assets
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    assert args.source_dir is not None, "Please provide a source mesh directory using --source_dir"
    source_mesh_dir = args.source_dir
    eval_source_dir = os.path.join(output_dir, 'eval_source_meshes')
    os.makedirs(eval_source_dir, exist_ok=True)

    mesh_file_list = os.listdir(source_mesh_dir)

    # load source meshes from the provided directory
    eval_source_v = []
    eval_source_f = []
    mesh_filenames = []

    for file in os.listdir(source_mesh_dir):
        if file.endswith('.obj') or file.endswith('.ply'):
            print('loading source mesh:', file)
            source_mesh = trimesh.load(os.path.join(source_mesh_dir, file), process=False)
            source_mesh_verts = torch.tensor(source_mesh.vertices, dtype=torch.float32)
            source_mesh_faces = torch.tensor(source_mesh.faces, dtype=torch.int64)
            eval_source_v.append(source_mesh_verts.to(device))
            eval_source_f.append(source_mesh_faces.to(device))
            mesh_filenames.append(file.split('.')[0])

            # copy source mesh to eval_source_dir
            shutil.copy(os.path.join(source_mesh_dir, file), os.path.join(eval_source_dir, file))
            

    # build model
    model = load_model(config, source_mesh=os.path.join(source_mesh_dir, mesh_file_list[0]))

    print('loading checkpoint from %s' % checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()

    deformed_meshes_dir = os.path.join(output_dir, 'deformed_meshes')
    os.makedirs(deformed_meshes_dir, exist_ok=True)

    if args.repeat > 1:
        for repeat_idx in range(args.repeat):
            model.generate_arbitrary_source_test(eval_source_v, eval_source_f, deformed_meshes_dir, 
                                    save_intermediates=False, evolution_dir=None, repeat_idx=repeat_idx, 
                                    mesh_filenames=mesh_filenames)
    else:
        model.generate_arbitrary_source_test(eval_source_v, eval_source_f, deformed_meshes_dir, 
                                    save_intermediates=False, evolution_dir=None, 
                                    mesh_filenames=mesh_filenames)        

    if args.render:
        source_video_dir = os.path.join(output_dir, 'eval_source_videos')
        render_trajectory(eval_source_dir, source_video_dir, R=args.R, h=args.h)
        
        render_trajectory_to_multiple_dirs(deformed_meshes_dir, os.path.join(output_dir, 'deformed_meshes_videos'),
                                           R=args.R, h=args.h)

if __name__ == "__main__":

    main()