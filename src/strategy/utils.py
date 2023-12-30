from .random import RandomSelection
from torch.utils.data import ConcatDataset
from .variance_ratio import VRSelection


def get_strategy(
    model,
    train_dataset,
    private_dataset,
    unlabeled_dataset,
    query_dataset,
    cfg,
    al_round,
):
    if cfg.strategy == "random":
        strategy = RandomSelection(model, unlabeled_dataset, cfg)
    elif cfg.strategy == "vr":
        strategy = VRSelection(model, unlabeled_dataset, cfg, al_round)
    else:
        raise NotImplementedError("This strategy is not implemented!")

    return strategy
