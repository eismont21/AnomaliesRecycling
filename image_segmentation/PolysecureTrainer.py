import torch
import os

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.solver.build import maybe_add_gradient_clipping, get_default_optimizer_params


class PolysecureTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_dir = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, cfg, False, output_dir)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )
        optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR)
        # optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR)
        return maybe_add_gradient_clipping(cfg, optimizer)
