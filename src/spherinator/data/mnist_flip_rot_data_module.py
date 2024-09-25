import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .mnist_flip_rot_dataset import mnist_fliprot_dataset as MNIST
from .spherinator_data_module import SpherinatorDataModule


class MNISTDataModule_flip_rot(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/",
        random_rotation: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        transformations = [
            transforms.ToTensor(),
            # transforms.Pad((0, 0, 1, 1), fill=0),
            # #transforms.Resize((96, 96)),
            # transforms.Resize((87, 87)),
        ]
        # if random_rotation:
        #     transformations += [transforms.RandomAffine(degrees=[0, 360])]
        transformations += [
            #transforms.Resize((32, 32)),
            #transforms.Resize((29, 29)),
            transforms.Lambda(
                lambda x: (x - x.min()) / (x.max() - x.min())
            ),  # Normalize to [0, 1]
            #transforms.RandomHorizontalFlip(p=0.5),
        ]
        self.transform = transforms.Compose(transformations)

    def prepare_data(self):
        MNIST()
        # MNIST(self.data_dir, train=False, download=True)
        print('skip')

    def setup(self, stage: str):
        if stage == "fit":
            self.mnist_train = MNIST('train')
            self.mnist_val = MNIST('valid')

        if stage == "test":
            self.mnist_test = MNIST(
                'test'
            )


        if stage == "predict":
            self.mnist_predict = MNIST(
                'test'
            )


    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
