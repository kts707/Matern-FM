import os
import pytorch_lightning as pl

class GenerateEveryNSteps(pl.Callback):
    def __init__(self, every_n_steps: int, base_dir: str, num_samples: int = 1):
        """
        Args:
            every_n_steps: run generate() when trainer.global_step % every_n_steps == 0
            base_dir: root directory under which per-step folders will be created
            num_samples: number of samples to generate each time
        """
        super().__init__()
        self.every_n_steps = every_n_steps
        self.base_dir = base_dir
        self.num_samples = num_samples
        os.makedirs(self.base_dir, exist_ok=True)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                           outputs, batch, batch_idx):
        step = trainer.global_step
        if step > 0 and step % self.every_n_steps == 0:
            # create a folder to save generated meshes
            save_dir = os.path.join(self.base_dir, f"step_{step:08d}")
            os.makedirs(save_dir, exist_ok=True)

            pl_module.generate_fixed_source(save_dir, num_samples=self.num_samples, device=pl_module.device)
            
            print(f"[Callback] Generated meshes at step {step} → {save_dir}")


class GenerateEveryNSteps_Arbitrary_Source(pl.Callback):
    def __init__(self, every_n_steps: int, base_dir: str, num_samples: int = 1, source_v = None):
        """
        Args:
            every_n_steps: run generate() when trainer.global_step % every_n_steps == 0
            base_dir: root directory under which per-step folders will be created
            num_samples: number of samples to generate each time
        """
        super().__init__()
        self.every_n_steps = every_n_steps
        self.base_dir = base_dir
        self.num_samples = num_samples
        self.test_source_v = source_v
        os.makedirs(self.base_dir, exist_ok=True)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                           outputs, batch, batch_idx):
        step = trainer.global_step
        if step > 0 and step % self.every_n_steps == 0:
            # create a folder to save generated meshes
            save_dir = os.path.join(self.base_dir, f"step_{step:08d}")
            os.makedirs(save_dir, exist_ok=True)

            pl_module.generate_arbitrary_source(self.test_source_v, save_dir, device=pl_module.device)
            
            print(f"[Callback] Generated meshes at step {step} → {save_dir}")