from copy import copy
from functools import partial
import pickle
import random
import numpy as np
from omegaconf import DictConfig
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, ConcatDataset
import os
from PIL import Image
from torchvision.datasets import cifar


class WeightedDataset(Dataset):
    def __init__(
        self,
        original_dataset,
        pseudo_labels=None,
        sample_weights=None,
    ):
        self.original_dataset = original_dataset
        self.sample_weights = sample_weights
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        im, tar = self.original_dataset[idx]
        if self.pseudo_labels is not None:
            tar = self.pseudo_labels[idx]
        w = self.sample_weights[idx]
        return im, tar, w


class PseudoLabeledDataset(Dataset):
    def __init__(
        self,
        labeled_dataset,
        unlabeled_dataset,
    ):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.labeled_permutation = np.random.permutation(len(labeled_dataset))

    def __len__(self):
        return len(self.unlabeled_dataset)

    def __getitem__(self, index):
        im_ul, tar_ul, w_ul = self.unlabeled_dataset[index]

        index_l = index % len(self.labeled_dataset)
        index_l = self.labeled_permutation[index_l]
        im_l, tar_l, w_l = self.labeled_dataset[index_l]

        return im_ul, tar_ul, w_ul, im_l, tar_l, w_l


class ImageNet(Dataset):
    def __init__(
        self,
        root,
        transform,
        target_transform,
        train=True,
        images_root="/mnt/data/Public_datasets/imagenet/imagenet_fullres",
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        dset_type = "train" if train else "val"

        with open(f"{root}/imagenet_al_data.pickle", "rb") as f:
            data = pickle.load(f)

        self.paths = [
            os.path.join(images_root, name) for name in data[dset_type]["image"]
        ]
        self.targets = data[dset_type]["label"]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        data, target = self.paths[index], self.targets[index]

        data = Image.open(data).convert("RGB")
        data = self.transform(data)

        target = self.target_transform(target)

        return data, target


class TinyImagenet(Dataset):
    def __init__(self, root, transform, target_transform, train=True):
        self.root = root
        dset_type = "train" if train else "val"

        self.data = []
        self.targets = []
        for num in range(20):
            self.data.append(
                np.load(os.path.join(root, f"processed/x_{dset_type}_{num+1:02d}.npy"))
            )
            self.targets.append(
                np.load(os.path.join(root, f"processed/y_{dset_type}_{num+1:02d}.npy"))
            )
        self.data = np.concatenate(np.array(self.data))
        self.targets = np.concatenate(np.array(self.targets))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        data, target = self.data[index], self.targets[index]

        data = Image.fromarray(np.uint8(255 * data))
        data = self.transform(data)

        target = self.target_transform(target)

        return data, target


def get_augmentations(cfg: DictConfig):
    train_augmentations = []
    test_augmentations = []

    if cfg.dataset.dataset.startswith("imagenet"):
        train_augmentations.extend(
            [transforms.Resize(256), transforms.RandomResizedCrop(224)]
        )
        test_augmentations.extend([transforms.Resize(256), transforms.CenterCrop(224)])

    train_augmentations.extend(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )
    test_augmentations.append(transforms.ToTensor())

    if not cfg.dataset.dataset.startswith("imagenet"):
        train_augmentations.append(
            transforms.Normalize(
                mean=cfg.dataset.normalization.mean,
                std=cfg.dataset.normalization.std,
            )
        )
        test_augmentations.append(
            transforms.Normalize(
                mean=cfg.dataset.normalization.mean,
                std=cfg.dataset.normalization.std,
            )
        )

    train_transform = transforms.Compose(train_augmentations)
    test_transform = transforms.Compose(test_augmentations)

    return train_transform, test_transform


def label_transform_with_mapping(label: int, mapping: dict) -> int:
    return mapping.get(label, len(mapping))


def prepare_data(cfg: DictConfig):
    data_path = cfg.dataset.data_path

    inlier_classes = cfg.dataset.inlier_classes
    num_classes = len(inlier_classes)

    train_transform, test_transform = get_augmentations(cfg)
    label_mapping = {
        original_id: new_id for new_id, original_id in enumerate(inlier_classes)
    }
    label_transform = partial(label_transform_with_mapping, mapping=label_mapping)

    if cfg.dataset.dataset.startswith("imagenet"):
        full_train_dataset = ImageNet(
            data_path,
            train_transform,
            label_transform,
            train=True,
            images_root=cfg.dataset.full_imagenet_path,
        )
        test_dataset = ImageNet(
            data_path,
            test_transform,
            label_transform,
            train=False,
            images_root=cfg.dataset.full_imagenet_path,
        )
    elif cfg.dataset.dataset == "CIFAR100":
        full_train_dataset = cifar.CIFAR100(
            data_path,
            transform=train_transform,
            target_transform=label_transform,
            train=True,
            download=True,
        )
        test_dataset = cifar.CIFAR100(
            data_path,
            transform=test_transform,
            target_transform=label_transform,
            train=False,
            download=True,
        )
    elif cfg.dataset.dataset == "tiny_imagenet":
        full_train_dataset = TinyImagenet(
            f"{data_path}/tiny_imagenet",
            transform=train_transform,
            target_transform=label_transform,
            train=True,
        )
        test_dataset = TinyImagenet(
            f"{data_path}/tiny_imagenet",
            transform=test_transform,
            target_transform=label_transform,
            train=False,
        )
    else:
        raise NotImplementedError("Dataset is not implemented!")

    predefined_split = f"{data_path}/splits/{cfg.dataset.dataset}_al_data_index.pickle"
    with open(predefined_split, "rb") as f:
        predefined = pickle.load(f)

    train_idx = predefined[cfg.seed]["train"]["labeled"][cfg.k_shot]
    valid_idx = copy(train_idx)
    unlabeled_idx = predefined[cfg.seed]["train"]["unlabeled"][cfg.outlier_ratio]
    random.shuffle(unlabeled_idx)

    test_idx = predefined[cfg.seed]["val"]

    private_set = []

    print(f"Train idx: {train_idx}")
    print(f"Valid idx: {valid_idx}")

    assert len(set(train_idx).intersection(unlabeled_idx)) == 0
    assert len(set(unlabeled_idx).intersection(valid_idx)) == 0

    return (
        Subset(full_train_dataset, train_idx),
        Subset(full_train_dataset, valid_idx),
        Subset(full_train_dataset, unlabeled_idx),
        private_set,
        Subset(test_dataset, test_idx),
    ), num_classes


def update_dataset(
    selected_dataset,
    train_dataset,
    valid_dataset,
    unlabeled_dataset,
    private_dataset,
    num_classes,
    cfg,
):
    selected_targets = [label for _, label in selected_dataset]
    id_idxs = [idx for idx, label in enumerate(selected_targets) if label < num_classes]
    ood_idxs = [
        idx for idx, label in enumerate(selected_targets) if label == num_classes
    ]

    if cfg.use_outlier_class:
        new_train_dataset = ConcatDataset([train_dataset, selected_dataset])
    else:
        new_train_dataset = ConcatDataset(
            [train_dataset, Subset(selected_dataset, id_idxs)]
        )

    new_valid_dataset = ConcatDataset(
        [valid_dataset, Subset(selected_dataset, id_idxs)]
    )

    if len(private_dataset) != 0:
        new_private_dataset = ConcatDataset(
            [private_dataset, Subset(selected_dataset, ood_idxs)]
        )
    else:
        new_private_dataset = Subset(selected_dataset, ood_idxs)

    unlabeled_idxs = list(
        set(range(len(unlabeled_dataset))) - set(selected_dataset.indices)
    )
    new_unlabeled_dataset = Subset(unlabeled_dataset, unlabeled_idxs)

    return (
        new_train_dataset,
        new_valid_dataset,
        new_unlabeled_dataset,
        new_private_dataset,
    )
