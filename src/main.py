from collections import Counter
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from dataset import prepare_data, update_dataset
from resnet import EnsembleModel

from util import (
    create_model,
    eval_model,
    extract_features,
    seed_torch,
    semi_supervised,
    train_model,
)
from strategy import get_strategy
from torch.utils.data import Subset


def run(cfg: DictConfig) -> None:
    seed_torch(cfg.seed)

    (
        train_dataset,
        valid_dataset,
        unlabeled_dataset,
        private_dataset,
        test_dataset,
    ), num_classes = prepare_data(cfg)

    print(
        f"train: {len(train_dataset)} val: {len(valid_dataset)} unlabeled: {len(unlabeled_dataset)} test: {len(test_dataset)}",
        flush=True,
    )

    current_valid_size = len(valid_dataset)

    selected_dataset = None

    for al_round in range(cfg.al_rounds):
        model = create_model(num_classes, cfg)

        if al_round != 0:
            _, _, train_targets = extract_features(
                model.get_model(0) if isinstance(model, EnsembleModel) else model,
                train_dataset,
                cfg,
            )

            number_of_samples_per_class = Counter(train_targets.cpu().numpy())
            print(
                f"Number of samples per class: {sorted(number_of_samples_per_class.items())}"
            )

        model = train_model(model, train_dataset, cfg, al_round=al_round)

        test_acc_all, test_acc_valid = eval_model(model, test_dataset, cfg)

        test_acc = test_acc_all
        if cfg.use_outlier_class:
            test_acc = test_acc_valid

        print(f"AL round {al_round} test accuracy before semi: {test_acc}")

        if cfg.use_semi and al_round != 0:
            model = semi_supervised(
                model,
                train_dataset,
                unlabeled_dataset,
                num_classes,
                cfg,
                al_round=al_round,
            )

        if al_round != 0:
            test_acc_all, test_acc_valid = eval_model(model, test_dataset, cfg)

            test_acc = test_acc_all
            if cfg.use_outlier_class:
                test_acc = test_acc_valid

        print(f"AL round {al_round} test accuracy after semi: {test_acc}")

        if al_round + 1 == cfg.al_rounds:
            # there is no need for new selection if we don't train in next round
            break

        selection_strategy = get_strategy(
            model,
            train_dataset,
            private_dataset,
            unlabeled_dataset,
            selected_dataset,
            cfg,
            al_round,
        )

        selected_idx = selection_strategy.select(cfg.budget)

        selected_dataset = Subset(unlabeled_dataset, selected_idx)

        (
            train_dataset,
            valid_dataset,
            unlabeled_dataset,
            private_dataset,
        ) = update_dataset(
            selected_dataset,
            train_dataset,
            valid_dataset,
            unlabeled_dataset,
            private_dataset,
            num_classes,
            cfg,
        )

        inlier_rate = (len(valid_dataset) - current_valid_size) / cfg.budget

        print(f"Inlier rate: {inlier_rate}")

        current_valid_size = len(valid_dataset)

        print(
            f"train: {len(train_dataset)}, valid: {len(valid_dataset)}, unlabeled: {len(unlabeled_dataset)}, private: {len(private_dataset)}"
        )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if torch.cuda.is_available():
        cfg.device = "cuda"

    run(cfg)

    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
