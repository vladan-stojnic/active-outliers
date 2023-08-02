import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from util import seed_torch


def run(cfg: DictConfig) -> None:
    seed_torch(cfg.seed)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if torch.cuda.is_available():
        cfg.device = "cuda"

    run(cfg)

    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
