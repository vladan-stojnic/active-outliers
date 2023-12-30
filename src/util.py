import copy
import os
import random
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import PseudoLabeledDataset, WeightedDataset

from resnet import ResNet18 as resnet18
from resnet import ResNet18ImageNet as resnet18_imagenet
from resnet import EnsembleModel
from resnet import ResNet18ImageNetCSI as resnet18_imagenet_csi
from resnet import ResNet18CSI as resnet18_csi


def seed_torch(seed: int = 0) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(num_classes: int, cfg: DictConfig):
    if cfg.use_outlier_class:
        num_classes += 1

    if cfg.dataset.dataset.startswith("imagenet"):
        model = resnet18_imagenet(num_classes)
        model_path = f"models/ssl/{cfg.dataset.dataset}_{cfg.outlier_ratio}.model"
        pretrained_dict = torch.load(model_path, map_location="cpu")
        missing, unexpected = model.backbone.load_state_dict(
            pretrained_dict, strict=False
        )
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
    else:
        model = resnet18(num_classes=num_classes)
        model_path = f"models/ssl/{cfg.dataset.dataset}_{cfg.outlier_ratio}.model"
        pretrained_dict = torch.load(model_path, map_location="cpu")
        del pretrained_dict["linear.weight"]
        del pretrained_dict["linear.bias"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if cfg.ensemble_len is not None:
        model = EnsembleModel(cfg.ensemble_len, model, cfg)

    return model


def create_csi_model(num_classes, cfg):
    if cfg.use_outlier_class:
        num_classes += 1

    if cfg.dataset.dataset.startswith("imagenet"):
        model = resnet18_imagenet_csi()
        model_path = f"models/ssl/{cfg.dataset.dataset}_{cfg.outlier_ratio}_csi.model"
        pretrained_dict = torch.load(model_path, map_location="cpu")
        missing, unexpected = model.backbone.load_state_dict(
            pretrained_dict, strict=False
        )
        print(f"Missing keys backbone: {missing}")
        print(f"Unexpected keys backbone: {unexpected}")
        missing, unexpected = model.simclr.load_state_dict(
            pretrained_dict, strict=False
        )
        print(f"Missing keys simclr: {missing}")
        print(f"Unexpected keys simclr: {unexpected}")
    else:
        model = resnet18_csi()
        model_path = f"models/ssl/{cfg.dataset.dataset}_{cfg.outlier_ratio}_csi.model"
        pretrained_dict = torch.load(model_path, map_location="cpu")
        del pretrained_dict["linear.weight"]
        del pretrained_dict["linear.bias"]
        missing, unexpected = model.backbone.load_state_dict(
            pretrained_dict, strict=False
        )
        print(f"Missing keys backbone: {missing}")
        print(f"Unexpected keys backbone: {unexpected}")
        missing, unexpected = model.simclr.load_state_dict(
            pretrained_dict, strict=False
        )
        print(f"Missing keys simclr: {missing}")
        print(f"Unexpected keys simclr: {unexpected}")

    return model


def _train_model_epochs(model, optimizer, criterion, loader, cfg, temperature=1.0):
    device = cfg.device

    for _ in range(cfg.al_train.epochs):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(temperature * outputs, targets.long())
            loss.backward()
            optimizer.step()

    return model


def _train_model_epochs_weighted(
    model, optimizer, criterion, loader, cfg, temperature=1.0
):
    device = cfg.device

    for _ in range(cfg.al_train.epochs):
        for inputs_ul, targets_ul, weights_ul, inputs_l, targets_l, weights_l in loader:
            inputs_ul, targets_ul, weights_ul = (
                inputs_ul.to(device),
                targets_ul.to(device),
                weights_ul.to(device),
            )
            inputs_l, targets_l, weights_l = (
                inputs_l.to(device),
                targets_l.to(device),
                weights_l.to(device),
            )

            inputs = torch.cat((inputs_ul, inputs_l))
            targets = torch.cat((targets_ul, targets_l))
            weights = torch.cat((weights_ul, weights_l))

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(temperature * outputs, targets.long())
            loss = loss * weights
            loss.mean().backward()
            optimizer.step()

    return model


def train_single_model(model, train_dataset, cfg, with_pseudo=False, temperature=1.0):
    device = cfg.device
    model.to(device)
    model.train()

    if cfg.al_train.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.al_train.lr)
    else:
        raise NotImplementedError("Optimizer not implemented!")

    reduction = "mean"
    if with_pseudo:
        reduction = "none"

    criterion = nn.CrossEntropyLoss(reduction=reduction)

    loader = DataLoader(
        train_dataset,
        batch_size=cfg.al_train.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    if with_pseudo:
        model = _train_model_epochs_weighted(
            model,
            optimizer,
            criterion,
            loader,
            cfg,
            temperature=temperature,
        )
    else:
        model = _train_model_epochs(
            model,
            optimizer,
            criterion,
            loader,
            cfg,
            temperature=temperature,
        )

    return model


def train_model(
    model,
    train_dataset,
    cfg,
    with_pseudo=False,
    temperature=1.0,
    al_round=None,
):
    if isinstance(model, EnsembleModel):
        if al_round == 0:
            # performance optimization
            # as we don't use ensemble for anything in round one we can train only one
            train_single_model(
                model.get_model(0),
                train_dataset,
                cfg,
                with_pseudo=with_pseudo,
                temperature=temperature,
            )
        else:
            for model_idx in range(cfg.ensemble_len):
                ensemble_model = model.get_model(model_idx)
                ensemble_model = train_single_model(
                    ensemble_model,
                    train_dataset,
                    cfg,
                    with_pseudo=with_pseudo,
                    temperature=temperature,
                )
    else:
        model = train_single_model(
            model,
            train_dataset,
            cfg,
            with_pseudo=with_pseudo,
            temperature=temperature,
        )

    return model


@torch.no_grad()
def eval_single_model(model, dataset, cfg):
    device = cfg.device
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=cfg.al_train.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    acc_all = 0.0
    acc_valid = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        acc_all += torch.sum(1.0 * (torch.max(outputs, 1)[1] == targets)).item()
        acc_valid += torch.sum(
            1.0 * (torch.max(outputs[..., :-1], 1)[1] == targets)
        ).item()

    return acc_all / len(dataset), acc_valid / len(dataset)


@torch.no_grad()
def eval_model(model, dataset, cfg):
    if isinstance(model, EnsembleModel):
        # we always evaluate model at 0 if it is ensemble
        return eval_single_model(model.get_model(0), dataset, cfg)
    else:
        return eval_single_model(model, dataset, cfg)


@torch.no_grad()
def extract_features(model, dataset, cfg):
    device = cfg.device
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset, batch_size=512, num_workers=cfg.num_workers, pin_memory=True
    )

    predictions = None
    features = None
    targets = None
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred, feat = model.forward(x, last=True, freeze=True)

        if predictions is None:
            predictions = pred
        else:
            predictions = torch.cat((predictions, pred), dim=0)

        if features is None:
            features = feat
        else:
            features = torch.cat((features, feat), dim=0)

        if targets is None:
            targets = y
        else:
            targets = torch.cat((targets, y), dim=0)

    return features, predictions, targets


def semi_supervised(
    model, train_dataset, unlabeled_dataset, num_classes, cfg, al_round
):
    if isinstance(model, EnsembleModel):
        pseudo_labels = None
        for i in range(cfg.ensemble_len):
            selected_model = model.get_model(i)
            _, preds, _ = extract_features(selected_model, unlabeled_dataset, cfg)
            if pseudo_labels is None:
                pseudo_labels = F.softmax(preds, dim=1)
            else:
                pseudo_labels += F.softmax(preds, dim=1)
        pseudo_labels /= cfg.ensemble_len
    else:
        _, pseudo_labels, _ = extract_features(model, unlabeled_dataset, cfg)
    sample_weights = 1 - torch.sum(
        -pseudo_labels * torch.log2(pseudo_labels), dim=-1
    ) / torch.log2(torch.tensor(num_classes + 1))
    pseudo_labels = torch.argmax(pseudo_labels, dim=1)
    semi_cfg = copy.deepcopy(cfg)
    semi_cfg.al_train.epochs = cfg.epochs_semi
    semi_cfg.al_train.batch_size = cfg.batch_size_semi
    model = train_model(
        model,
        PseudoLabeledDataset(
            WeightedDataset(
                train_dataset,
                None,
                sample_weights=np.ones((len(train_dataset),)),
            ),
            WeightedDataset(
                unlabeled_dataset,
                pseudo_labels=pseudo_labels.cpu().detach().numpy(),
                sample_weights=sample_weights.cpu().detach().numpy(),
            ),
        ),
        semi_cfg,
        with_pseudo=True,
        al_round=al_round,
    )

    return model
