import gc

import matplotlib
import numpy as np
import torch
import torchvision.transforms.functional as functional
from lightning.pytorch.callbacks import Callback
from matplotlib import figure

matplotlib.use("Agg")


class LogZernike(Callback):
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

        # Generate some random samples from the validation set
        data = next(iter(trainer.train_dataloader))
        y = data[1][: self.num_samples].to(pl_module.device)
        y = torch.nn.functional.one_hot(y,num_classes=10).to(torch.float)
        data = data[0]
        samples = data[: self.num_samples].to(pl_module.device)

        # Generate reconstructions of the samples using the model
        with torch.no_grad():
            scaled = samples

            # crop = functional.center_crop(samples, [pl_module.crop_size, pl_module.crop_size])
            # scaled = functional.resize(
            #     crop, [pl_module.input_size, pl_module.input_size], antialias=True
            # )
            norm = torch.sum(torch.abs(scaled),dim =(-1,-2),keepdim=True)
            scaled = scaled/norm
            if pl_module.__class__.__name__ == "RotationalAutoencoder":
                recon, _ = pl_module(scaled)
            else:
                out,rec = pl_module(scaled)

            #loss_recon = pl_module.criterion(out,rec)
            #out = pl_module.Embedding_Function.decode(out)*norm
            rec  = torch.einsum('ij,...jk->...ik',pl_module.mask2,rec)
            rec = pl_module.Embedding_Function.decode(rec)*norm
            scaled = scaled*norm
            #print(torch.sum(out,dim=(-1,-2))[0:10])

        # with torch.no_grad():
        #     print(out.size())
        #     x = torch.argmax(out, dim=1).to(torch.int)
        #     y = torch.abs(y-1).to(torch.int)
        #     # size = y.size(0)
        #     # z = torch.zeros(size)
        #     # for i in range(size):
        #     #     z[i] = y[i,x[i]]

        #     # num_wrong = torch.sum(x).item()
        # #print(2*self.num_samples)
        # #print(num_wrong)
        # #self.num_samples = min(2*self.num_samples,num_wrong)
        # #scaled = scaled[x==1]
        # print(y)
        # print(x)
        # print(y[x,:])
        # print(y.size())
        # print(x.size())
        # #print(y[x,:].size())
        # asdc
        # print(scaled.size())
        # scaled = scaled[(y[x]).bool()]
        # print(scaled.size())
        # Plot the original samples and their reconstructions side by side

        fig = figure.Figure(figsize=(4 * self.num_samples, 6))
        ax = fig.subplots(3, self.num_samples)
        for i in range(self.num_samples):
            ax[0, i].imshow(((scaled)[i].cpu().detach().numpy().T),vmin=0,vmax=1)
            ax[0, i].set_title("Original"+str(np.sum(np.abs((scaled)[i].cpu().detach().numpy()))))
            ax[0, i].axis("off")
            ax[1, i].imshow(rec[i].cpu().detach().numpy().T,vmin=0,vmax=1)
            ax[1, i].set_title("Reconstruction"+str(np.sum(np.abs((rec)[i].cpu().detach().numpy()))))
            ax[1, i].axis("off")
            # ax[2, i].imshow(out[i].cpu().detach().numpy().T,vmin=0,vmax=1)
            # ax[2, i].set_title("Model_output"+str(np.sum(np.abs((out)[i].cpu().detach().numpy()))))
            # ax[2, i].axis("off")
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