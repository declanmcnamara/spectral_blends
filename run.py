import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["DESIMODEL"] = "./desimodel"
import random
from os.path import exists

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

from setup import setup
from utils import (
    favi_step,
    favi_step3,
    make_dataloader,
    make_dataloader3,
    training_loop,
    training_loop3,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize(config_path="conf", version_base=None)
    # cfg = compose(config_name="config")

    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # cfg.data.three_cameras=False
    # # # # cfg.training.live_dream=False
    # cfg.encoder.type='fourier_coadd'

    dir = cfg.dir
    os.chdir(dir)
    (favi_model, encoder, optimizer, scheduler, logger_string, kwargs) = setup(cfg)

    if kwargs["generate_data"]:
        return  # The run is solely for generating data.

    if not kwargs["live_dream"]:
        # Make dataloader
        my_loader = (
            make_dataloader3(**kwargs)
            if kwargs["three_cameras"]
            else make_dataloader(**kwargs)
        )
    else:
        my_loader = None

    if not exists("./logs/{}".format(logger_string)):
        os.mkdir("./logs/{}".format(logger_string))

    test_losses = []
    for j in range(kwargs["epochs"]):

        if kwargs["live_dream"]:
            if kwargs["three_cameras"]:
                test_loss = favi_step3(encoder, scheduler, favi_model, None, **kwargs)
                test_losses.append(test_loss)
            else:
                test_loss = favi_step(encoder, scheduler, favi_model, None, **kwargs)
                test_losses.append(test_loss)

            if (j % cfg.training.save_every) == 0:
                torch.save(
                    encoder.state_dict(),
                    "./logs/{}/encoder_weights_{}.pth".format(logger_string, j),
                )
                np.save(
                    "./logs/{}/test_losses_{}.npy".format(logger_string, j),
                    np.array(test_losses),
                )
        else:
            if kwargs["three_cameras"]:
                test_loss = training_loop3(
                    encoder, scheduler, favi_model, my_loader, None, j, **kwargs
                )
                test_losses.append(test_loss)
            else:
                test_loss = training_loop(
                    encoder, scheduler, favi_model, my_loader, None, j, **kwargs
                )
                test_losses.append(test_loss)

            if (j % cfg.training.save_every) == 0:
                torch.save(
                    encoder.state_dict(),
                    "./logs/{}/encoder_weights_{}.pth".format(logger_string, j),
                )
                np.save(
                    "./logs/{}/test_losses_{}.npy".format(logger_string, j),
                    np.array(test_losses),
                )

    torch.save(
        encoder.state_dict(), "./logs/{}/encoder_weights.pth".format(logger_string)
    )
    np.save("./logs/{}/test_losses.npy".format(logger_string), np.array(test_losses))


if __name__ == "__main__":
    main()
