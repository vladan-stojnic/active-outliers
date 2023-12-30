import torch
import torch.nn.functional as F
from resnet import EnsembleModel

from util import extract_features


class RandomSelection:
    def __init__(self, model, unlabeled_dataset, cfg, al_round):
        self.model = model
        self.unlabeled_dataset = unlabeled_dataset
        self.cfg = cfg
        self.al_round = al_round

    @torch.no_grad()
    def select(self, budget):
        scores = torch.rand(len(self.unlabeled_dataset), device=self.cfg.device)

        if self.cfg.use_filtering and self.al_round != 0:
            if isinstance(self.model, EnsembleModel):
                preds = None
                for idx in range(self.cfg.ensemble_len):
                    _, single_preds, _ = extract_features(
                        self.model.get_model(idx), self.unlabeled_dataset, self.cfg
                    )
                    if preds is None:
                        preds = F.softmax(single_preds, dim=1)
                    else:
                        preds += F.softmax(single_preds, dim=1)
                preds /= self.cfg.ensemble_len
            else:
                _, preds, _ = extract_features(
                    self.model,
                    self.unlabeled_dataset,
                    self.cfg,
                )
            preds = preds.argmax(dim=1)
            idx = torch.where(preds == len(self.cfg.dataset.inlier_classes))[0]
            scores[idx] = 0

        return (torch.argsort(scores, descending=True)[:budget]).tolist()
