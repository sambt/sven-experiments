import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import experiments.experiment_code as experiment_code


@hydra.main(config_path="experiments/configs", version_base=None)
def main(cfg: DictConfig):
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; defaulting to CPU")
        cfg.device = "cpu"

    if cfg.print_config:
        print(OmegaConf.to_yaml(cfg))

    exp_fn = getattr(experiment_code, cfg.name)
    exp_fn(cfg=cfg)


if __name__ == "__main__":
    main()
