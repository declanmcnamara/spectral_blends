import astropy.units as u
import numpy as np
import specsim
import specsim.simulator
import torch
import torch.distributions as D
from provabgs import infer as Infer
from provabgs import models as Models
from scipy import ndimage
from scipy.interpolate import CubicSpline


class SpecModelNew:
    """Wrapper class for galaxy model from provabgs.
    Creates model with specified prior, and defines
    helpful sampling functions for FAVI-based training."""

    def __init__(
        self, device, obs_grid=np.arange(3000.0, 10000.0, 5.0), min_blend_scale=0.1
    ):
        self.obs_grid = obs_grid
        self.prop_blend = 0.5
        self._redshift_prior = D.Uniform(0.0, 1.0)
        self.device = device
        self.generator = Models.NMF(burst=True, emulator=True)
        self.prior = Infer.load_priors(
            [
                Infer.UniformPrior(7.0, 12.5, label="sed"),
                Infer.FlatDirichletPrior(4, label="sed"),  # flat dirichilet priors
                Infer.UniformPrior(0.0, 1.0, label="sed"),  # burst fraction
                Infer.UniformPrior(1e-2, 13.27, label="sed"),  # tburst
                Infer.LogUniformPrior(
                    4.5e-5, 1.5e-2, label="sed"
                ),  # log uniform priors on ZH coeff
                Infer.LogUniformPrior(
                    4.5e-5, 1.5e-2, label="sed"
                ),  # log uniform priors on ZH coeff
                Infer.UniformPrior(0.0, 3.0, label="sed"),  # uniform priors on dust1
                Infer.UniformPrior(0.0, 3.0, label="sed"),  # uniform priors on dust2
                Infer.UniformPrior(
                    -2.0, 1.0, label="sed"
                ),  # uniform priors on dust_index
            ]
        )
        self.scale_dist = D.Uniform(min_blend_scale, 1.0)

    def _prior_sample(self, n_obs=100):
        """Sample from the prior. Requires using the .transform
        attribute of the prior object."""
        prior_samples = np.stack(
            [self.prior.transform(self.prior.sample()) for i in range(n_obs)]
        )
        return prior_samples

    def _redshift_sample(self, n_obs=100):
        redshifts = self._redshift_prior.sample((n_obs,))
        return redshifts.numpy()

    def generate_specs(self, n_obs=100):
        prior_samples = self._prior_sample(n_obs)
        redshifts = self._redshift_sample(n_obs)
        outwave, outspec = self.generator.seds(tt=prior_samples, zred=redshifts)
        return outwave, outspec, redshifts

    def clean_specs(self, waves, fluxes, redshifts):
        """Remove observations whose fluxes contain inf entries."""
        to_keep = ~np.isinf(fluxes).any(1)
        return waves[to_keep], fluxes[to_keep], redshifts[to_keep]

    def _resample_one(self, wave, flux):
        """For a single spectrograph, resample onto rest-frame grid.
        wave, flux are 1D ndarrays."""
        cs = CubicSpline(wave, flux)
        res = cs(self.obs_grid)
        # smoothed = ndimage.gaussian_filter1d(res, 1.0)
        return res

    def resample(self, waves, fluxes):
        new_fluxes = np.empty((fluxes.shape[0], len(self.obs_grid)))
        for i in range(len(fluxes)):
            new_fluxes[i] = self._resample_one(waves[i], fluxes[i])
        return new_fluxes

    def normalize(self, fluxes):
        return fluxes / fluxes.sum(1).reshape(-1, 1)

    def blend(self, f1, f2):
        """Takes in two sets of SEDs to be blended, assumed to already
        be on a common grid. Current behavior blends as follows:
            1) Normalize all SEDs
            2) Add SEDs together entrywise
            3) Normalize once again.
        The first normalization step puts fluxes on a similar scale, so blends can be detected.
        The second normalization step ensures that both blended and unblended observations have the same scale.
        """
        assert f1.shape == f2.shape, "Two SED sets are not the same size."
        f1prime, f2prime = self.normalize(f1), self.normalize(f2)

        # Get different magnitudes
        n_spectra = f1.shape[0]
        multipliers = self.scale_dist.sample((n_spectra,)).numpy()
        multf1prime = np.multiply(multipliers.reshape(-1, 1), f1prime)

        blended = multf1prime + f2prime
        return self.normalize(blended)

    def create_unblended(self, n_obs=100):
        waves, specs, redshifts = self.generate_specs(n_obs)
        waves, specs, redshifts = self.clean_specs(waves, specs, redshifts)
        specs = self.resample(waves, specs)
        return specs

    def create_blended(self, n_obs=100):
        set1, set2 = self.create_unblended(n_obs), self.create_unblended(n_obs)
        trunc = min([len(set1), len(set2)])
        return self.blend(set1[:trunc], set2[:trunc])

    def dream(self, n_obs=100):
        n_blends = int((n_obs * self.prop_blend))
        unbl, bl = self.create_unblended(n_obs - n_blends), self.create_blended(
            n_blends
        )
        unbl = self.normalize(unbl)  # hasn't been done yet, but blended have been

        seds = np.concatenate([unbl, bl], axis=0)
        labels = np.concatenate([np.zeros(n_obs - n_blends), np.ones(n_blends)], axis=0)
        p = np.random.permutation(len(seds))
        mixed_seds, mixed_labels = seds[p], labels[p]

        noise_multiplier = (
            D.Normal(1.0, 0.1).sample(mixed_seds.shape).clamp(1e-2).numpy()
        )
        # Multiply noise and normalize
        noisy_specs = np.multiply(mixed_seds, noise_multiplier)
        noisy_specs = self.normalize(noisy_specs)

        # TODO: figure out where these nan's come from, probably from normalization.
        to_keep = ~np.isnan(mixed_seds).any(1)
        return mixed_seds[to_keep], mixed_labels[to_keep]


class SpecModel3New:
    """Wrapper class for galaxy model from provabgs.
    Creates model with specified prior, and defines
    helpful sampling functions for FAVI-based training."""

    def __init__(
        self,
        device,
        prop_blend,
        obs_grid=np.arange(3000.0, 10000.0, 5.0),
        sim_batch_size=100,
        min_blend_scale=0.1,
        brightness_multiplier_desi=1e6,
    ):
        self.obs_grid = obs_grid
        self.prop_blend = prop_blend
        self._redshift_prior = D.Uniform(0.0, 1.0)
        self.device = device
        self.generator = Models.NMF(burst=True, emulator=True)
        self.desi_specsim = specsim.simulator.Simulator("desi", num_fibers=1)
        self.desi_specsim2 = specsim.simulator.Simulator(
            "desi", num_fibers=sim_batch_size
        )
        self.prior = Infer.load_priors(
            [
                Infer.UniformPrior(7.0, 12.5, label="sed"),
                Infer.FlatDirichletPrior(4, label="sed"),  # flat dirichilet priors
                Infer.UniformPrior(0.0, 1.0, label="sed"),  # burst fraction
                Infer.UniformPrior(1e-2, 13.27, label="sed"),  # tburst
                Infer.LogUniformPrior(
                    4.5e-5, 1.5e-2, label="sed"
                ),  # log uniform priors on ZH coeff
                Infer.LogUniformPrior(
                    4.5e-5, 1.5e-2, label="sed"
                ),  # log uniform priors on ZH coeff
                Infer.UniformPrior(0.0, 3.0, label="sed"),  # uniform priors on dust1
                Infer.UniformPrior(0.0, 3.0, label="sed"),  # uniform priors on dust2
                Infer.UniformPrior(
                    -2.0, 1.0, label="sed"
                ),  # uniform priors on dust_index
            ]
        )
        self.scale_dist = D.Uniform(min_blend_scale, 1.0)
        self.brightness_multiplier_desi = brightness_multiplier_desi

    def _prior_sample(self, n_obs=100):
        """Sample from the prior. Requires using the .transform
        attribute of the prior object."""
        prior_samples = np.stack(
            [self.prior.transform(self.prior.sample()) for i in range(n_obs)]
        )
        return prior_samples

    def _redshift_sample(self, n_obs=100):
        redshifts = self._redshift_prior.sample((n_obs,))
        return redshifts.numpy()

    def generate_specs(self, n_obs=100):
        prior_samples = self._prior_sample(n_obs)
        redshifts = self._redshift_sample(n_obs)
        outwave, outspec = self.generator.seds(tt=prior_samples, zred=redshifts)
        return outwave, outspec, redshifts

    def clean_specs(self, waves, fluxes, redshifts):
        """Remove observations whose fluxes contain inf entries."""
        to_keep = ~np.isinf(fluxes).any(1)
        return waves[to_keep], fluxes[to_keep], redshifts[to_keep]

    def _resample_one(self, wave, flux):
        """For a single spectrograph, resample onto rest-frame grid.
        wave, flux are 1D ndarrays."""
        cs = CubicSpline(wave, flux)
        res = cs(self.obs_grid)
        # smoothed = ndimage.gaussian_filter1d(res, 1.0)
        return res

    def resample(self, waves, fluxes):
        new_fluxes = np.empty((fluxes.shape[0], len(self.obs_grid)))
        for i in range(len(fluxes)):
            new_fluxes[i] = self._resample_one(waves[i], fluxes[i])
        return new_fluxes

    def normalize(self, fluxes):
        return fluxes / fluxes.sum(1).reshape(-1, 1)

    def blend(self, f1, f2):
        """Takes in two sets of SEDs to be blended, assumed to already
        be on a common grid. Current behavior blends as follows:
            1) Normalize all SEDs
            2) Add SEDs together entrywise
            3) Normalize once again.
        The first normalization step puts fluxes on a similar scale, so blends can be detected.
        The second normalization step ensures that both blended and unblended observations have the same scale.
        """
        assert f1.shape == f2.shape, "Two SED sets are not the same size."
        f1prime, f2prime = self.normalize(f1), self.normalize(f2)
        # Get different magnitudes
        n_spectra = f1.shape[0]
        multipliers = self.scale_dist.sample((n_spectra,)).numpy()
        multf1prime = np.multiply(multipliers.reshape(-1, 1), f1prime)

        blended = multf1prime + f2prime
        return self.normalize(blended)

    def create_unblended_fast(self, n_obs=100):
        waves, specs, redshifts = self.generate_specs(n_obs)
        waves, specs, redshifts = self.clean_specs(waves, specs, redshifts)
        specs = self.resample(waves, specs)
        return specs

    def create_blended_fast(self, n_obs=100):
        set1, set2 = self.create_unblended_fast(n_obs), self.create_unblended_fast(
            n_obs
        )
        trunc = min([len(set1), len(set2)])
        return self.blend(set1[:trunc], set2[:trunc])

    def dream_fast(self, n_obs=100):
        n_blends = int((n_obs * self.prop_blend))
        unbl, bl = self.create_unblended_fast(
            n_obs - n_blends
        ), self.create_blended_fast(n_blends)

        unbl = self.normalize(unbl)  # hasn't been done yet, but blended have been

        seds = np.concatenate([unbl, bl], axis=0)
        labels = np.concatenate([np.zeros(n_obs - n_blends), np.ones(n_blends)], axis=0)
        p = np.random.permutation(len(seds))
        mixed_seds, mixed_labels = seds[p], labels[p]
        scaled_seds = mixed_seds * self.brightness_multiplier_desi

        scaled_specs = scaled_seds * 1e-17 * u.erg / (u.Angstrom * u.s * u.cm**2)
        n_valid = scaled_specs.shape[0]
        dummys = np.repeat(scaled_specs[-1].reshape(1, -1), n_obs - n_valid, axis=0)
        to_use = np.concatenate([scaled_specs, dummys], 0)
        self.desi_specsim2.simulate(source_fluxes=to_use)

        raw_b = np.array(self.desi_specsim2.camera_output[0]["observed_flux"])
        raw_r = np.array(self.desi_specsim2.camera_output[1]["observed_flux"])
        raw_z = np.array(self.desi_specsim2.camera_output[2]["observed_flux"])
        sigma_b = torch.tensor(
            np.array(self.desi_specsim2.camera_output[0]["flux_inverse_variance"])
            ** (-1 / 2)
        )
        sigma_r = torch.tensor(
            np.array(self.desi_specsim2.camera_output[1]["flux_inverse_variance"])
            ** (-1 / 2)
        )
        sigma_z = torch.tensor(
            np.array(self.desi_specsim2.camera_output[2]["flux_inverse_variance"])
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

        # TODO: figure out where these nan's come from, probably from normalization.
        to_keep_b = ~np.isnan(noisy_b).any(1)
        to_keep_r = ~np.isnan(noisy_r).any(1)
        to_keep_z = ~np.isnan(noisy_z).any(1)
        to_keep = np.logical_or(to_keep_b, to_keep_r, to_keep_z)
        return (
            noisy_b[to_keep],
            noisy_r[to_keep],
            noisy_z[to_keep],
            mixed_labels[to_keep],
        )
