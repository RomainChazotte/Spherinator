import lightning.pytorch as pl
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from .spherinator_dataset import SpherinatorDataset


class mnist_fliprot_dataset(SpherinatorDataset):
    """ flip-rotated MNIST dataset """

    def __init__(self, mode='train', transform=None, target_transform=None, reshuffle_seed=None):
        """
        :type  mode: string from ['train', 'valid', 'test']
        :param mode: determines which subset of the dataset is loaded and whether augmentation is used
        :type  transform: callable
        :param transform: transformation applied to PIL images, returning transformed version
        :type  target_transform: callable
        :param target_transform: transformation applied to labels
        :type  reshuffle_seed: int
        :param reshuffle_seed: seed to use to reshuffle train or valid sets. If None (default), they are not reshuffled
        """
        assert mode in ['train', 'valid', 'trainval', 'test']
        assert reshuffle_seed is None or (mode != "test" and mode != 'trainval')

        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

        # load the numpy arrays
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
            transforms.RandomHorizontalFlip(p=0.5),
            #transforms.Pad(padding=3),
            #transforms.RandomResizedCrop(28,scale=(0.9,1.1))
        ]
        self.transform = transforms.Compose(transformations)

        if mode in ["train", "valid", "trainval"]:
            filename = '/home/chazotrn/development/Classifier_mnist/data/stuff/mnist_fliprot_trainval.npz'

            data = np.load(filename)

            num_train = len(data["labels"])
            indices = np.arange(0, num_train)

            if reshuffle_seed is not None:
                rng = np.random.RandomState(reshuffle_seed)
                rng.shuffle(indices)

            split = int(np.floor(num_train * 5 / 6))
            print(num_train)
            print(split)
            print(mode)
            if mode == "train":
                data = {
                    "images": data["images"][indices[:split], :],
                    "labels": data["labels"][indices[:split]]
                }

            elif mode == "valid":
                data = {
                    "images": data["images"][indices[split:], :],
                    "labels": data["labels"][indices[split:]]
                }

        elif mode =="test":
            filename = '/home/chazotrn/development/Classifier_mnist/data/stuff/mnist_fliprot_test.npz'
            data = np.load(filename)

        self.images = data['images'].astype(np.float32)
        self.labels = data['labels'].astype(np.int64)
        self.num_samples = len(self.labels)
        # print(len(self.labels))
        # import matplotlib.pyplot as plt
        # for i in range(10):
        #     plt.figure()
        #     plt.imshow(np.array(self.images[i]))
        #     plt.savefig('im{}.jpg'.format(i))
        #     plt.close()


    def __getitem__(self, index):
        """
        :type  index: int
        :param index: index of data
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image, label = self.images[index], self.labels[index]
        # convert to PIL Image
        image = Image.fromarray(image)
        # transform images and labels
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)



    def get_metadata(self, index: int):
        """Retrieves the metadata of the item/items with the given indices from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            metadata: Metadata of the item/items with the given indices.
        """
        metadata = {
            "filename": self.images[index],
            "labels": self.labels[index],
        }
        return metadata
