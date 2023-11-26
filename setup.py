import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["DESIMODEL"] = "./desimodel"
from os.path import exists

import hydra
import numpy as np
import pywt
import specsim
import specsim.simulator
import torch
import torch.distributions as D
import torch.nn as nn
from hydra import compose, initialize
from omegaconf import DictConfig

from modules import (
    CNNWavelet,
    Conv,
    Conv1d,
    CustomOptim,
    Dense,
    Dense3,
    FourierCoadd,
    FourierDESI,
    WaveletCoadd,
)
from noisy_model import SpecModel3New, SpecModelNew


def setup(cfg: DictConfig):
    # Some initial hyperparameters
    suggested_device = cfg.training.device
    if "cuda" in suggested_device:
        device = suggested_device if torch.cuda.is_available() else "cpu"
    else:
        device = suggested_device
    epochs = cfg.training.epochs
    lr = cfg.training.lr
    n_dream_batches = cfg.data.n_dream_batches
    dream_batch_size = cfg.data.dream_batch_size
    data_dir = cfg.data.dir
    training_batch_size = cfg.training.batch_size
    three_cameras = cfg.data.three_cameras
    min_blend_scale = cfg.data.min_blend_scale
    brightness_multiplier_desi = cfg.data.brightness_multiplier_desi
    generate_data = cfg.data.generate
    kwargs = {
        "device": device,
        "epochs": epochs,
        "lr": lr,
        "n_dream_batches": n_dream_batches,
        "dream_batch_size": dream_batch_size,
        "data_dir": data_dir,
        "training_batch_size": training_batch_size,
        "three_cameras": three_cameras,
        "min_blend_scale": min_blend_scale,
        "brightness_multiplier_desi": brightness_multiplier_desi,
        "generate_data": generate_data,
    }

    # Set up model/encoder for coadded or three camera version.
    if not cfg.data.three_cameras:
        favi_model = SpecModelNew(device, min_blend_scale=min_blend_scale)
        if cfg.encoder.type == "coadd":
            name = "coadd"
            encoder = Dense(spec_length=len(favi_model.obs_grid))
            encoder = encoder.to(device)
            optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
            scheduler = CustomOptim(optimizer, start_lr=lr)
        elif cfg.encoder.type == "wavelet_coadd":
            name = "wavelet_coadd"
            encoder = WaveletCoadd(
                start=cfg.encoder.wavelet.start,
                stop=cfg.encoder.wavelet.stop,
                wave=cfg.encoder.wavelet.wave,
                device=device,
            )
            encoder = encoder.to(device)
            optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
            scheduler = CustomOptim(optimizer, start_lr=lr)
        else:
            name = "fourier_coadd"
            encoder = FourierCoadd(cfg.encoder.fourier.n_fft)
            encoder = encoder.to(device)
            optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
            scheduler = CustomOptim(optimizer, start_lr=lr)
    else:
        desi = specsim.simulator.Simulator("desi", num_fibers=1)
        wavelength_grid = np.array(desi.simulated["wavelength"])
        sim_batch_size = (
            cfg.data.dream_batch_size if generate_data else cfg.training.batch_size
        )
        favi_model = SpecModel3New(
            device=device,
            prop_blend=cfg.data.prop_blend,
            obs_grid=wavelength_grid,
            sim_batch_size=sim_batch_size,
            min_blend_scale=min_blend_scale,
            brightness_multiplier_desi=brightness_multiplier_desi,
        )
        name = cfg.encoder.type
        # Setup

        if name == "wavelet":
            start = cfg.encoder.wavelet.start
            stop = cfg.encoder.wavelet.stop
            wave = cfg.encoder.wavelet.wave
            encoder = CNNWavelet(start=start, stop=stop, wave=wave, device=device)
        elif name == "fourier":
            n_fft = cfg.encoder.fourier.n_fft
            encoder = FourierDESI(n_fft=n_fft)
        elif name == "dense3":
            x1, x2, x3, z = favi_model.dream_fast(training_batch_size)
            x1shape, x2shape, x3shape = x1.shape[-1], x2.shape[-1], x3.shape[-1]
            encoder = Dense3(
                x1shape,
                x2shape,
                x3shape,
                cfg.encoder.dense3.hidden1,
                cfg.encoder.dense3.latent_dim,
                cfg.encoder.dense3.hidden2,
                device,
            )
        else:
            raise ValueError("Encoder name not appropriate")

        encoder = encoder.to(device)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
        scheduler = CustomOptim(optimizer, start_lr=lr)

    # Details about the training process/logging
    live_dream = cfg.training.live_dream
    replay = cfg.training.replay
    kwargs["live_dream"] = live_dream
    kwargs["replay"] = replay
    data_save_dir = (
        "three_camera={},min_blend_scale={},brightness_multiplier_desi={}".format(
            three_cameras, min_blend_scale, brightness_multiplier_desi
        )
    )
    logger_string = "{},three={},live={},min_blend={},desi_bright={}".format(
        name, three_cameras, live_dream, min_blend_scale, brightness_multiplier_desi
    )
    kwargs["data_save_dir"] = data_save_dir

    # Check if we need to generate dreamt data
    if (not kwargs["live_dream"]) and (generate_data):

        # Check if data save directory exists, if not make it.
        if not exists("{}/{}".format(cfg.data.dir, data_save_dir)):
            os.mkdir("{}/{}".format(cfg.data.dir, data_save_dir))

        if cfg.data.three_cameras:

            for j in range(cfg.data.n_dream_batches):
                print("Generating batch {} of three-camera data.".format(j))
                x1, x2, x3, z = favi_model.dream_fast(cfg.data.dream_batch_size)
                np.save(
                    "{}/{}/three_camera_x1_{}".format(cfg.data.dir, data_save_dir, j),
                    x1,
                )
                np.save(
                    "{}/{}/three_camera_x2_{}".format(cfg.data.dir, data_save_dir, j),
                    x2,
                )
                np.save(
                    "{}/{}/three_camera_x3_{}".format(cfg.data.dir, data_save_dir, j),
                    x3,
                )
                np.save(
                    "{}/{}/three_camera_z_{}".format(cfg.data.dir, data_save_dir, j), z
                )
        else:

            for j in range(cfg.data.n_dream_batches):
                print("Generating batch {} of coadded data.".format(j))
                x, z = favi_model.dream(cfg.data.dream_batch_size)
                np.save("{}/{}/coadded_x_{}".format(cfg.data.dir, data_save_dir, j), x)
                np.save("{}/{}/coadded_z_{}".format(cfg.data.dir, data_save_dir, j), z)

    return (favi_model, encoder, optimizer, scheduler, logger_string, kwargs)
