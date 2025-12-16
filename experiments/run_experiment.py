import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import experiments.experiment_code as experiment_code


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; defaulting to CPU")
        cfg.device = "cpu"
    device = torch.device(cfg.device)

    if cfg.print_config:
        print(OmegaConf.to_yaml(cfg))

    # Instantiate experiment as a partial (see configs/experiment/*.yaml)
    exp_fn = getattr(experiment_code, cfg.experiment.name)

    # Run the experiment with model/optimizer configs (they can be partials)
    exp_fn(cfg=cfg)

if __name__ == "__main__":
    main()
