import gc

import matplotlib
import numpy as np
import torch
import torchvision.transforms.functional as functional
import torchvision
from lightning.pytorch.callbacks import Callback
from matplotlib import figure
import random
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
        self.rand_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)

    def on_train_epoch_end(self, trainer, pl_module):
        # Return if no wandb logger is used
        if trainer.logger is None or trainer.logger.__class__.__name__ not in [
            "WandbLogger",
            "MyLogger",
        ]:
            return


        if trainer.current_epoch % 50 == 0:
            # breakpoint()
            # fig = figure.Figure()
            images = trainer.train_dataloader.dataset.images
            num_samples = trainer.train_dataloader.dataset.num_samples
            labels = trainer.train_dataloader.dataset.labels
            markers = ["$0$","$1$","$2$","$3$","$4$","$5$","$6$","$7$","$8$","$9$"]

            for j in range(10):
                images_temp = images[labels==j]
                # for i in range(0,images_temp.size(0),512):
                embeds = pl_module.encode(pl_module.Embedding_Function.embed(self.rand_flip(functional.rotate(images_temp[0:0+112],random.randint(0,360))))).squeeze(1)
                embeds = torch.sqrt(torch.sum(torch.square(embeds),dim=-1))
                x = np.array(embeds[:,0].cpu().detach())
                y = np.array(embeds[:,1].cpu().detach())
                current_labels = np.array(labels[0:0+112].cpu())
                # current_labels_str = current_labels.copy()
                # current_labels_str = []
                # for j in range(len(current_labels)):

                #     current_labels_str.append( markers[current_labels[j]])
                matplotlib.pyplot.scatter(x,y,alpha=0.4,marker=markers[j],linewidths=0)
                # matplotlib.pyplot.scatter(x,y,c=current_labels*10,cmap='viridis')
            matplotlib.pyplot.savefig('scatter_images/scatter_image_no_overwrite{}.png'.format(int(trainer.current_epoch/50)))
            matplotlib.pyplot.clf()
            # trainer.logger.log_image(key="map", images=[fig])



            # for i in range(0,num_samples,512):
            #     embeds = pl_module.encode(pl_module.Embedding_Function.embed(images[i:i+512])).squeeze(1)
            #     embeds = torch.sqrt(torch.sum(torch.square(embeds),dim=-1))
            #     x = np.array(embeds[:,0].cpu().detach())
            #     y = np.array(embeds[:,1].cpu().detach())
            #     current_labels = np.array(labels[i:i+512].cpu())
            #     # current_labels_str = current_labels.copy()
            #     # current_labels_str = []
            #     # for j in range(len(current_labels)):

            #     #     current_labels_str.append( markers[current_labels[j]])
            #     matplotlib.pyplot.scatter(x,y,alpha=0.2,c=current_labels*10,linewidths=0,cmap='inferno')
            #     # matplotlib.pyplot.scatter(x,y,c=current_labels*10,cmap='viridis')
            # matplotlib.pyplot.savefig('scatter_images/scatter_image{}.png'.format(int(trainer.current_epoch/50)))
            # # trainer.logger.log_image(key="map", images=[fig])

            # for i in range(self.num_samples):
            #     ax[0, i].clear()
            #     ax[0, i].clear()


            # fig.clear()
            gc.collect()

        if trainer.current_epoch % 20 != 0:
            return
        # Generate some random samples from the validation set
        data = next(iter(trainer.train_dataloader))[: self.num_samples]
        samples = data[0].to(pl_module.device)

        # Generate reconstructions of the samples using the model
        with torch.no_grad():
            scaled = samples

            samples = pl_module.Embedding_Function.embed(samples)

            out = pl_module(samples)

            #loss_recon = pl_module.criterion(out,rec)
            out = pl_module.Embedding_Function.decode(out)
            out = torch.clip(out,0,1)
            rec = pl_module.Embedding_Function.decode(samples)
            #print(torch.sum(out,dim=(-1,-2))[0:10])


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

