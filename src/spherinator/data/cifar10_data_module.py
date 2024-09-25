import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


class Cifar10DataModule(pl.LightningDataModule):
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
            transforms.Pad((0, 0, 1, 1), fill=0),
            transforms.Resize((96, 96)),
        ]
        # if random_rotation:
        #     transformations += [transforms.RandomAffine(degrees=[0, 360])]
        transformations += [
            transforms.Resize((32, 32)),
            transforms.Lambda(
                lambda x: (x - x.min()) / (x.max() - x.min())
            ),  # Normalize to [0, 1]
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        self.transform = transforms.Compose(transformations)

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45904, 4096])

        if stage == "test":
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.cifar_predict = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.cifar_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
