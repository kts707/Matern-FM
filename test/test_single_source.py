import sys
import os
import argparse

import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from model import load_model
from utils.render import render_trajectory, render_all_meshes

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--config", type=str, required=True, help="Path to exp_dir")
    parser.add_argument("--source_mesh", type=str, required=True, help="Path to base mesh")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="generated_meshes", help="Where to write meshes")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of meshes to generate")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    parser.add_argument("--render", action='store_true', help="Render results using polyscope")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times repeated")
    parser.add_argument("--save_evolution", action='store_true', help="Save evolution")
    parser.add_argument("--R", type=float, default=2.5, help="Camera radius for rendering")
    parser.add_argument("--h", type=float, default=1.5, help="Camera height for rendering")

    args = parser.parse_args()
    return args

def main():

    args = get_parser()

    config = OmegaConf.load(args.config)

    config.train.batch_size = 1

    # set seeds and device
    pl.seed_everything(args.seed)

    # results_dir = args.exp_dir
    base_mesh = args.source_mesh

    # build model
    model = load_model(config, source_mesh=base_mesh)

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

    print('loading checkpoint from %s' % checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()

    # evaluation and saving assets
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if args.save_evolution:
        evolution_dir = output_dir + '_evolution'
        if not os.path.exists(evolution_dir):
            os.makedirs(evolution_dir, exist_ok=True)
        save_intermediate = True
    else:
        evolution_dir = None
        save_intermediate = False

    if args.repeat > 1:
        for repeat_idx in range(args.repeat):
            model.generate_fixed_source(output_dir, num_samples=args.num_samples, repeat_idx=repeat_idx, 
                                        save_intermediates=save_intermediate, evolution_dir=evolution_dir)
    else:
        model.generate_fixed_source(output_dir, num_samples=args.num_samples, 
                                    save_intermediates=save_intermediate, evolution_dir=evolution_dir)

    if args.render:
        render_trajectory(output_dir, output_dir+'_videos', R=args.R, h=args.h)

        if save_intermediate:
            evolution_videos_dir = output_dir + '_evolution_videos'
            os.makedirs(evolution_videos_dir, exist_ok=True)
            for meshes_dir in sorted(os.listdir(evolution_dir)):
                render_all_meshes(os.path.join(evolution_dir, meshes_dir), 
                                  os.path.join(evolution_videos_dir, meshes_dir + '.mp4'),
                                  camera_location=config.render.fixed_camera_location,
                                  fps=10)

if __name__ == "__main__":

    main()