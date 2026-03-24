import gc

import matplotlib
import numpy as np
import torch
import torchvision.transforms.functional as functional
from lightning.pytorch.callbacks import Callback
from matplotlib import figure

matplotlib.use("Agg")


class LogIllustris(Callback):
    def __init__(
        self,
        num_samples: int = 6,
        indices: list[int] = [],
    ):
        super().__init__()
        self.num_samples = num_samples
        self.indices = indices

    def on_train_epoch_end(self, trainer, pl_module):
        # Return if no wandb logger is used
        if trainer.logger is None or trainer.logger.__class__.__name__ not in [
            "WandbLogger",
            "MyLogger",
        ]:
            return
        # if trainer.current_epoch % 20 != 0:
        #     return
        # Generate some random samples from the validation set
        data = next(iter(trainer.train_dataloader))[: self.num_samples]
        samples = data.to(pl_module.device)

        # Generate reconstructions of the samples using the model
        with torch.no_grad():
            scaled = samples

            # if trainer.current_epoch % 5 == 0:
            #     embeds = torch.empty(6,8,2,2)
            #     for i in range(8):
            #         embeds[:,i] = pl_module.encode(pl_module.Embedding_Function.embed(functional.rotate(samples,i*30))).squeeze(1)
            #     embeds_reduced = torch.sqrt(torch.sum(torch.square(embeds),dim=-1))
            #     print( embeds)
            #     print(embeds_reduced)

            samples = pl_module.Embedding_Function.embed(samples)

            out = pl_module.forward(samples)
            # out,q_z, p_z = pl_module.forward(samples)
            out = pl_module.Embedding_Function.decode(out)
            rec = pl_module.Embedding_Function.decode(samples)

        # Plot the original samples and their reconstructions side by side


        fig = figure.Figure(figsize=(2 * self.num_samples, 6))
        ax = fig.subplots(3, self.num_samples)
        for i in range(self.num_samples):
            ax[0, i].imshow(((scaled)[i].cpu().detach().numpy().T),vmin=0,vmax=1, cmap='inferno')
            # ax[0, i].set_title("Original"+str(np.sum(np.abs((scaled)[i].cpu().detach().numpy()))))
            ax[0, i].axis("off")
            ax[1, i].imshow(rec[i].cpu().detach().numpy().T,vmin=0,vmax=1, cmap='inferno')
            # ax[1, i].set_title("Reconstruction"+str(np.sum(np.abs((rec)[i].cpu().detach().numpy()))))
            ax[1, i].axis("off")
            ax[2, i].imshow(out[i].cpu().detach().numpy().T,vmin=0,vmax=1, cmap='inferno')
            # ax[2, i].set_title("Model_output"+str(np.sum(np.abs((out)[i].cpu().detach().numpy()))))
            ax[2, i].axis("off")


        # for i in range(self.num_samples):
        #     ax[0, i].imshow(((scaled)[i].cpu().detach().numpy().T),vmin=0,vmax=1, cmap='inferno')
        #     # ax[0, i].set_title("Original"+str(np.sum(np.abs((scaled)[i].cpu().detach().numpy()))))
        #     ax[0, i].axis("off")
        #     ax[1, i].imshow(out[i].cpu().detach().numpy().T,vmin=0,vmax=1, cmap='inferno')
        #     # ax[1, i].set_title("Reconstruction"+str(np.sum(np.abs((rec)[i].cpu().detach().numpy()))))
        #     ax[1, i].axis("off")

        fig.tight_layout()

        # Log the figure at W&B
        trainer.logger.log_image(key="Reconstructions", images=[fig])

        # Clear the figure and free memory
        # Memory leak issue: https://github.com/matplotlib/matplotlib/issues/27138
        for i in range(self.num_samples):
            ax[0, i].clear()
            ax[0, i].clear()
        fig.clear()
        gc.collect()