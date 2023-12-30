import torch
import torch.nn.functional as F
from resnet import EnsembleModel
from util import extract_features
from .random import RandomSelection


class VRSelection:
    def __init__(self, model, unlabeled_dataset, cfg, al_round):
        self.model = model
        self.unlabeled_dataset = unlabeled_dataset
        self.cfg = cfg
        self.al_round = al_round

    @torch.no_grad()
    def select(self, budget):
        if not isinstance(self.model, EnsembleModel):
            print("Ensemble size is 1 so selection will be done as random!")
            return RandomSelection(
                self.model, self.unlabeled_dataset, self.cfg, self.al_round
            ).select(budget)

        if self.cfg.ensemble_len == 1:
            print("Ensemble size is 1 so selection will be done as random!")
            return RandomSelection(
                self.model.get_model(0), self.unlabeled_dataset, self.cfg, self.al_round
            ).select(budget)

        if self.al_round == 0:
            return RandomSelection(
                self.model.get_model(0), self.unlabeled_dataset, self.cfg, self.al_round
            ).select(budget)

        argmax_preds_per_model = []
        ensemble_prediction = None
        for i in range(self.cfg.ensemble_len):
            selected_model = self.model.get_model(i)
            _, preds, _ = extract_features(
                selected_model, self.unlabeled_dataset, self.cfg
            )
            argmax_preds_per_model.append(preds.argmax(dim=1))
            if ensemble_prediction is None:
                ensemble_prediction = F.softmax(preds, dim=1)
            else:
                ensemble_prediction += F.softmax(preds, dim=1)

        ensemble_prediction /= self.cfg.ensemble_len
        ensemble_prediction = ensemble_prediction.argmax(dim=1)

        vr = torch.zeros_like(ensemble_prediction, dtype=torch.float32)
        for i in range(self.cfg.ensemble_len):
            vr += ensemble_prediction != argmax_preds_per_model[i]
        vr /= self.cfg.ensemble_len

        if self.cfg.use_filtering:
            idx = torch.where(
                ensemble_prediction == len(self.cfg.dataset.inlier_classes)
            )[0]
            vr[idx] = 0

        return (torch.argsort(vr, descending=True)[:budget]).tolist()
