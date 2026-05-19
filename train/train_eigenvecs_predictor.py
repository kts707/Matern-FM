import sys
import os
import time

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from omegaconf import OmegaConf

from dataset import load_dataset
from model import load_model

def main(config):
    # set seeds and device
    pl.seed_everything(config.manual_seed)

    results_dir = os.path.join('results', config.group, config.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # save configs
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config, f)

    # init wandb
    wandb_logger = None
    if config.get("wandb", {}).get("enabled"):
        import wandb
        print('use wandb')
        wandb_name = config.exp_name
        flat_cfg = OmegaConf.to_container(config, resolve=True)
        wandb_logger = WandbLogger(name=wandb_name, config=flat_cfg,
                                   project=config["wandb"]["project"],
                                   entity=config["wandb"]["entity"],
                                   group=config["group"])
        disable_wandb = False
    else:
        disable_wandb = True

    # build dataloaders
    train_dataset = load_dataset(config.dataset, results_dir)

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, 
                              pin_memory=True, collate_fn=train_dataset.collate_fn, num_workers=config.train.get("num_workers", 0))

    # build model
    model = load_model(config, source_mesh=train_dataset.source_mesh_path)
    
    # callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join('results', config.group, config.exp_name, 'checkpoints'),
            every_n_train_steps=config.train.save_every_n_steps, 
            # save_last=True, save_top_k=-1,
        )
    ]

    # gradient clipping
    clip_grad_norm = False
    if config.train.get("gradient_clipping", None) is not None:
        clip_grad_norm = config.train.gradient_clipping.enabled

    if clip_grad_norm:
        max_grad_norm = config.train.gradient_clipping.max_norm
    else:
        max_grad_norm = None

    log_freq = config.train.get("log_freq", 1)

    # initialize Trainer
    trainer = pl.Trainer(devices=1, max_steps=config.train.iterations, callbacks=callbacks, logger=wandb_logger, 
                         log_every_n_steps=log_freq, gradient_clip_val=max_grad_norm, gradient_clip_algorithm='norm',
                         accumulate_grad_batches=config.train.get("accumulate_grad_batches", 1),
                         enable_progress_bar=config.get("enable_progress_bar", True))

    # training
    trainer.fit(model, train_dataloaders=train_loader)

    # save final model
    trainer.save_checkpoint(os.path.join(results_dir, 'checkpoints', 'final.ckpt'))

    # save wandb id as txt file
    if not disable_wandb:
        wandb_id_file = os.path.join(results_dir, "wandb_id.txt")
        with open(wandb_id_file, 'w',) as file:
            file.write(wandb.run.id)

if __name__ == "__main__":
    config_path = str(sys.argv[1])
    config = OmegaConf.load(config_path)

    additional_config = OmegaConf.from_cli(args_list=sys.argv[2:])
    
    if additional_config != {}:
        config = OmegaConf.merge(config, additional_config)

    start_time = time.time()
    main(config)

    print('total time:', (time.time() - start_time) / 60, 'mins')