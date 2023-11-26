import os

os.environ["DESIMODEL"] = "./desimodel"
import random
from os.path import exists

import astropy.units as u
import desispec.io  # Input/Output functions related to DESI spectra
import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import specsim
import specsim.simulator
import torch
import torch.distributions as D
from astropy import coordinates as coords
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io import fits
from astropy.table import Table
from astroquery.sdss import SDSS
from desimodel.footprint import radec2pix  # For getting healpix values
from desispec import coaddition  # Functions related to coadding the spectra
from hydra import compose, initialize
from omegaconf import DictConfig
from speclite import filters

from setup import setup
from utils import (
    favi_step,
    favi_step3,
    make_dataloader,
    make_dataloader3,
    training_loop,
    training_loop3,
)

plt.style.use("science")
matplotlib.rcParams.update({"font.size": 30})


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize(config_path="conf", version_base=None)
    # cfg = compose(config_name="config")

    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # cfg.data.three_cameras = True
    # cfg.training.device = "cuda:5"
    # cfg.data.generate = False
    # cfg.data.brightness_multiplier_desi = 1e5

    dir = cfg.dir
    os.chdir(dir)
    (favi_model, encoder, optimizer, scheduler, logger_string, kwargs) = setup(cfg)

    if cfg.data.three_cameras:
        n_obs = 256
        n_blends = int((n_obs * favi_model.prop_blend))
        unbl, bl = favi_model.create_unblended_fast(
            n_obs - n_blends
        ), favi_model.create_blended_fast(n_blends)

        unbl = favi_model.normalize(unbl)  # hasn't been done yet, but blended have been

        seds = np.concatenate([unbl, bl], axis=0)
        labels = np.concatenate([np.zeros(n_obs - n_blends), np.ones(n_blends)], axis=0)
        p = np.random.permutation(len(seds))
        mixed_seds, mixed_labels = seds[p], labels[p]
        scaled_seds = mixed_seds * favi_model.brightness_multiplier_desi

        scaled_specs = scaled_seds * 1e-17 * u.erg / (u.Angstrom * u.s * u.cm**2)
        n_valid = scaled_specs.shape[0]
        dummys = np.repeat(scaled_specs[-1].reshape(1, -1), n_obs - n_valid, axis=0)
        to_use = np.concatenate([scaled_specs, dummys], 0)
        favi_model.desi_specsim2.simulate(source_fluxes=to_use)

        raw_b = np.array(favi_model.desi_specsim2.camera_output[0]["observed_flux"])
        raw_r = np.array(favi_model.desi_specsim2.camera_output[1]["observed_flux"])
        raw_z = np.array(favi_model.desi_specsim2.camera_output[2]["observed_flux"])

        wave_b = np.array(favi_model.desi_specsim2.camera_output[0]["wavelength"])
        wave_r = np.array(favi_model.desi_specsim2.camera_output[1]["wavelength"])
        wave_z = np.array(favi_model.desi_specsim2.camera_output[2]["wavelength"])

        sigma_b = torch.tensor(
            np.array(favi_model.desi_specsim2.camera_output[0]["flux_inverse_variance"])
            ** (-1 / 2)
        )
        sigma_r = torch.tensor(
            np.array(favi_model.desi_specsim2.camera_output[1]["flux_inverse_variance"])
            ** (-1 / 2)
        )
        sigma_z = torch.tensor(
            np.array(favi_model.desi_specsim2.camera_output[2]["flux_inverse_variance"])
            ** (-1 / 2)
        )

        sigma_b = sigma_b.nan_to_num(nan=1e-28)
        sigma_r = sigma_r.nan_to_num(nan=1e-28)
        sigma_z = sigma_z.nan_to_num(nan=1e-28)
        noisy_b = raw_b + D.Normal(0.0, sigma_b).sample().cpu().numpy()
        noisy_r = raw_r + D.Normal(0.0, sigma_r).sample().cpu().numpy()
        noisy_z = raw_z + D.Normal(0.0, sigma_z).sample().cpu().numpy()

        noisy_b = noisy_b.T / 1e-17
        noisy_r = noisy_r.T / 1e-17
        noisy_z = noisy_z.T / 1e-17

        noisy_b = noisy_b[:n_valid]
        noisy_r = noisy_r[:n_valid]
        noisy_z = noisy_z[:n_valid]

        blended_index = np.where(mixed_labels == 1)[0][1]

        coadd_obj = desispec.io.read_spectra(
            "{}/coadd-sv1-bright-10378.fits".format(cfg.data.dir)
        )
        row = 32
        coadd_spec = coadd_obj[row]

        fig, ax = plt.subplots(
            nrows=2, ncols=1, sharex=True, sharey=False, figsize=(20, 10)
        )
        ax[0].plot(
            wave_b,
            noisy_b[blended_index].reshape(-1),
            color="blue",
            alpha=0.7,
        )
        ax[0].plot(
            wave_r,
            noisy_r[blended_index].reshape(-1),
            color="red",
            alpha=0.7,
        )
        ax[0].plot(
            wave_z,
            noisy_z[blended_index].reshape(-1),
            color="green",
            alpha=0.7,
        )

        ax[1].plot(coadd_spec.wave["b"], coadd_spec.flux["b"][0], color="b", alpha=0.7)
        ax[1].plot(coadd_spec.wave["r"], coadd_spec.flux["r"][0], color="r", alpha=0.7)
        ax[1].plot(coadd_spec.wave["z"], coadd_spec.flux["z"][0], color="g", alpha=0.7)
        ax[0].set_ylim(-7, 7)
        ax[1].set_ylim(-7, 7)
        fig.supxlabel("\AA", y=0.1)
        fig.supylabel("Flux [$10^{-17} erg\ s^{-1}\ cm^{-2}\ \AA^{-1}$]", y=0.60)
        plt.tight_layout()
        plt.savefig("./figs/three_camera_example.png", dpi=500)

        # # Over-plotting smoothed spectra in black for all the three arms
        # ax[0].set_xlim(3500, 9900)
        # ax[1].set_xlim(3500, 9900)
        # ax[0].set_ylim(-0.09e0, 0.09e0)
        # ax[1].set_ylim(-0.0035e0, 0.0035e0)
        # fig.supxlabel("$\lambda$ [$\AA$]")
        # # fig.supylabel("$F_{\lambda}$ [$10^{-17} erg\ s^{-1}\ cm^{-2}\ \AA^{-1}$]")
        # plt.tight_layout()
        # plt.savefig("./figs/three_camera_example.png")

        # # Plot example DESI spectrum from EDR
        # coadd_obj = desispec.io.read_spectra(
        #     "{}/coadd-sv1-bright-10378.fits".format(cfg.data.dir)
        # )
        # row = 10
        # coadd_spec = coadd_obj[row]
        # coadd_spec.wave
        # coadd_spec.flux
        # fig, ax = plt.subplots(figsize=(20, 6))
        # ax.plot(coadd_spec.wave["b"], coadd_spec.flux["b"][0], color="b", alpha=0.7)
        # ax.plot(coadd_spec.wave["r"], coadd_spec.flux["r"][0], color="r", alpha=0.7)
        # ax.plot(coadd_spec.wave["z"], coadd_spec.flux["z"][0], color="g", alpha=0.7)
        # fig.supxlabel("$\lambda$ [$\AA$]")
        # fig.supylabel("$F_{\lambda}$ [$10^{-17} erg\ s^{-1}\ cm^{-2}\ \AA^{-1}$]")
        # plt.tight_layout()
        # plt.savefig("./figs/desi_real_example.png")

    else:
        n_blends = int((100 * favi_model.prop_blend))
        unbl, bl = favi_model.create_unblended(
            100 - n_blends
        ), favi_model.create_blended(n_blends)
        unbl = favi_model.normalize(unbl)  # hasn't been done yet, but blended have been

        seds = np.concatenate([unbl, bl], axis=0)
        labels = np.concatenate([np.zeros(100 - n_blends), np.ones(n_blends)], axis=0)
        p = np.random.permutation(len(seds))
        mixed_seds, mixed_labels = seds[p], labels[p]

        noise_multiplier = (
            D.Normal(1.0, 0.1).sample(mixed_seds.shape).clamp(1e-2).numpy()
        )
        # Multiply noise and normalize
        noisy_specs = np.multiply(mixed_seds, noise_multiplier)

        # Find a blended spectrum
        blended_index = np.where((mixed_labels == 1))[0][14]

        wavelength_grid = favi_model.obs_grid
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(20, 10))
        ax[0].plot(
            wavelength_grid,
            mixed_seds[blended_index].reshape(-1),
            color="blue",
            alpha=0.7,
        )
        ax[1].plot(
            wavelength_grid,
            noisy_specs[blended_index].reshape(-1),
            color="blue",
            alpha=0.7,
        )

        # Over-plotting smoothed spectra in black for all the three arms
        # ax[0].set_ylim(-0.2e16,0.2e16)
        # ax[1].set_ylim(-0.2e16,0.2e16)
        fig.supxlabel("\AA", y=0.1)
        # fig.supylabel('$F_{\lambda}$ [$10^{-17} erg\ s^{-1}\ cm^{-2}\ \AA^{-1}$]')
        plt.tight_layout()
        plt.savefig("./figs/coadded_example.png", dpi=500)

        # Real SDSS spectra
        # pos = coords.SkyCoord("0h8m05.63s +14d50m23.3s", frame="icrs")
        # ra = np.array([179.68641])
        # dec = np.array([-0.60292])
        # coordinates = coords.SkyCoord(ra, dec, frame="icrs", unit="deg")
        # xid = SDSS.query_region(coordinates, spectro=True)
        # sp = SDSS.get_spectra(matches=xid)
        # data = sp[0][1].data

        # fig, ax = plt.subplots(figsize=(20, 6))
        # ax.plot(np.power(10, data["loglam"]), data["flux"])
        # ax.set_xlabel("Wavelength [Angstrom]")
        # ax.set_ylabel("Flux ($10^{-17}$ erg/s/cm\u00b2/Angstrom)")
        # plt.savefig("./figs/sdss_spectrum.png")


if __name__ == "__main__":
    main()
