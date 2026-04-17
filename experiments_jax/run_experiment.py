"""Hydra entry point for JAX Sven experiments.

Run e.g.::

    python -m experiments_jax.run_experiment --config-name toy_1d_scan
"""

from __future__ import annotations

import hydra
import jax
from omegaconf import DictConfig, OmegaConf

import experiments_jax.experiment_code as experiment_code


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Device handling. JAX picks the default device based on the installed
    # backend; we just surface what was selected so results files can record it.
    backends = [d.platform for d in jax.devices()]
    want = cfg.get("device", "auto")
    if want == "cpu":
        jax.config.update("jax_platform_name", "cpu")
    print(f"JAX devices: {jax.devices()}  (requested device={want})")

    if cfg.get("enable_x64", False):
        jax.config.update("jax_enable_x64", True)

    if cfg.get("print_config", False):
        print(OmegaConf.to_yaml(cfg))

    exp_fn = getattr(experiment_code, cfg.name)
    exp_fn(cfg=cfg)


if __name__ == "__main__":
    main()
