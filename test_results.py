import os

os.environ["DESIMODEL"] = "./desimodel"
import random
from os.path import exists

import hydra
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from torchmetrics import AUROC

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
    # cfg.data.three_cameras = False
    # # # cfg.training.live_dream = False
    # cfg.encoder.type = "fourier_coadd"
    # cfg.data.generate = False
    # cfg.data.min_blend_scale = 0.1

    dir = cfg.dir
    os.chdir(dir)
    (favi_model, encoder, optimizer, scheduler, logger_string, kwargs) = setup(cfg)

    device = "cpu"
    kwargs["device"] = device

    encoder.load_state_dict(
        torch.load(
            "./logs/{}/encoder_weights.pth".format(logger_string),
            map_location=torch.device("cpu"),
        )
    )
    encoder = encoder.to(device)

    if not cfg.data.three_cameras:
        auroc = AUROC(task="binary")
        corrects = []
        nlls = []
        my_aurocs = []
        for j in range(10):
            print("On trial {}".format(j))
            x, z = favi_model.dream(1000)
            x, z = (
                torch.tensor(x).to(device).float(),
                torch.tensor(z).to(device).float(),
            )
            pis = encoder.get_q(x).probs.view(-1)
            perc_correct = ((pis.round() - z.view(-1)) == 0.0).sum() / len(z)
            qzx = encoder.get_q(x)
            nll = -1 * qzx.log_prob(z.view(-1, 1)).mean()
            corrects.append(perc_correct.item())
            nlls.append(nll.item())
            this_auroc = auroc(pis, z).item()
            my_aurocs.append(this_auroc)

            del x
            del z

        result_df = pd.DataFrame(
            {
                "Accuracy": [
                    "{0:.4f}".format(np.mean(corrects))
                    + " ({0:.4f})".format(np.std(corrects))
                ],
                "NLL": [
                    "{0:.4f}".format(np.mean(nlls)) + " ({0:.4f})".format(np.std(nlls))
                ],
                "AUROC": [
                    "{0:.4f}".format(np.mean(my_aurocs))
                    + " ({0:.4f})".format(np.std(my_aurocs))
                ],
            }
        )

        result_df.to_latex("./logs/{}/test_results.tex".format(logger_string))

    else:

        auroc = AUROC(task="binary")
        corrects = []
        nlls = []
        my_aurocs = []
        for j in range(10):
            print("On trial {}".format(j))
            x1, x2, x3, z = favi_model.dream_fast(256)
            x1, x2, x3 = (
                torch.tensor(x1).float().to(device),
                torch.tensor(x2).float().to(device),
                torch.tensor(x3).float().to(device),
            )
            z = torch.tensor(z).float().to(device)
            pis = encoder.get_q(x1, x2, x3).probs.view(-1)
            perc_correct = ((pis.round() - z.view(-1)) == 0.0).sum() / len(z)
            qzx = encoder.get_q(x1, x2, x3)
            nll = -1 * qzx.log_prob(z.view(-1, 1)).mean()
            corrects.append(perc_correct.item())
            nlls.append(nll.item())
            this_auroc = auroc(pis, z).item()
            my_aurocs.append(this_auroc)

            del x1
            del x2
            del x3
            del z

        result_df = pd.DataFrame(
            {
                "Accuracy": [
                    "{0:.4f}".format(np.mean(corrects))
                    + " ({0:.4f})".format(np.std(corrects))
                ],
                "NLL": [
                    "{0:.4f}".format(np.mean(nlls)) + " ({0:.4f})".format(np.std(nlls))
                ],
                "AUROC": [
                    "{0:.4f}".format(np.mean(my_aurocs))
                    + " ({0:.4f})".format(np.std(my_aurocs))
                ],
            }
        )

        result_df.to_latex("./logs/{}/test_results.tex".format(logger_string))


if __name__ == "__main__":
    main()
