# from torchvision import models
import numpy as np
import pywt
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch.nn import AvgPool1d, Conv1d
from torch.utils.data import DataLoader, Dataset

# From:
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Optim.py#L4

"""A wrapper class for scheduled optimizer """


class CustomOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, start_lr=1e-3):
        self._optimizer = optimizer
        self.start_lr = start_lr
        self.dummy = 1 / self.start_lr
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_steps += 1
        lr = 1 / (self.dummy + self.n_steps / 10)

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


class Dense(nn.Module):
    def __init__(self, spec_length):
        super(Dense, self).__init__()
        self.spec_length = spec_length
        self.network = nn.Sequential(
            nn.Linear(self.spec_length, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.network(x)
        return out

    def get_q(self, x):
        p = self.forward(x)
        return D.Bernoulli(p)


class FourierCoadd(torch.nn.Module):
    def __init__(self, n_fft):
        super(FourierCoadd, self).__init__()
        self.n_fft = n_fft
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(2, 16, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=5, stride=2),
            torch.nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=5, stride=2),
            # torch.nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool1d(kernel_size=5, stride=2),
        )

        self.dense = torch.nn.Sequential(
            nn.Linear(4160, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, reshaped, n):
        realed = torch.view_as_real(reshaped).transpose(-1, -2)
        out = self.conv(realed)  # (n unknown) x 32 x unknown
        out = rearrange(out, "(n d) c w -> n (d c w)", n=n)
        out = self.dense(out)
        return out

    def get_q(self, x):
        n = x.shape[0]
        head = torch.stft(x, n_fft=self.n_fft, return_complex=True)
        reshaped = head.flatten(start_dim=0, end_dim=1)
        probs = self.forward(reshaped, n)
        return D.Bernoulli(probs)


class WaveletCoadd(torch.nn.Module):
    def __init__(self, start, stop, wave, device):
        super(WaveletCoadd, self).__init__()
        self.start = start
        self.stop = stop
        self.wave = wave
        self.device = device
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, 4, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.dense = torch.nn.Sequential(
            nn.Linear(1392, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return

    def get_q(self, x):
        # n = x1.shape[0]
        head = pywt.wavedec(x.cpu().numpy(), self.wave)
        head = head[self.start : self.stop]

        c = [
            self.conv(torch.tensor(signal).unsqueeze(1).to(self.device))
            for signal in head
        ]

        c = torch.cat(c, axis=-1)

        flatten = torch.flatten(c, start_dim=1)
        probs = self.dense(flatten)
        return D.Bernoulli(probs)


class Conv(nn.Module):
    def __init__(self, spec_length):
        super(Conv, self).__init__()
        self.spec_length = spec_length
        self.conv1d = Conv1d(1, 2, kernel_size=25, stride=2)
        self.avgpool = AvgPool1d(kernel_size=2)
        self.dense = nn.Sequential(
            nn.Linear(1750, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 1, len(self.spec_length))
        out = self.avgpool(x)
        out = F.relu(out)
        out = self.avgpool(out)
        out = F.relu(out)
        # out = out.flatten(start_dim=1)
        out = self.dense(out)
        return out

    def get_q(self, x):
        p = self.forward(x)
        return D.Bernoulli(p)


class DESI3CameraDatset(Dataset):
    def __init__(self, specs, inds, cut1, cut2, device):
        self.specs = specs
        self.inds = inds
        self.cut1 = cut1
        self.cut2 = cut2
        self.device = device

    def __getitem__(self, index):
        # full_sed = torch.tensor(self.specs[index], dtype=torch.float64)
        full_sed = self.specs[index].to(self.device)
        response = self.inds[index].to(self.device)
        return (
            full_sed[: self.cut1],
            full_sed[self.cut1 : self.cut2],
            full_sed[self.cut2 :],
            response,
        )
        return (
            torch.tensor(self.specs[index], dtype=torch.float64),
            torch.tensor(self.response[index], dtype=torch.float64),
        )

    def __len__(self):
        return len(self.specs)


class DESICoaddDatset(Dataset):
    def __init__(self, specs, inds, device):
        self.specs = specs
        self.inds = inds
        self.device = device

    def __getitem__(self, index):
        # full_sed = torch.tensor(self.specs[index], dtype=torch.float64)
        full_sed = self.specs[index].to(self.device)
        response = self.inds[index].to(self.device)
        return (
            full_sed,
            response,
        )

    def __len__(self):
        return len(self.specs)


# Load encoder
class FourierDESI(torch.nn.Module):
    def __init__(self, n_fft):
        super(FourierDESI, self).__init__()
        self.n_fft = n_fft

        self.dense1 = torch.nn.Sequential(
            nn.Linear(20644, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )

        self.dense2 = torch.nn.Sequential(
            nn.Linear(18356, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )

        self.dense3 = torch.nn.Sequential(
            nn.Linear(20800, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )

        self.combiner = torch.nn.Sequential(
            nn.Linear(3 * 512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, reshaped1, reshaped2, reshaped3, n):
        realed1 = torch.view_as_real(reshaped1).transpose(-1, -2)
        realed2 = torch.view_as_real(reshaped2).transpose(-1, -2)
        realed3 = torch.view_as_real(reshaped3).transpose(-1, -2)

        out1 = rearrange(realed1, "(n d) c w -> n (d c w)", n=n)
        out2 = rearrange(realed2, "(n d) c w -> n (d c w)", n=n)
        out3 = rearrange(realed3, "(n d) c w -> n (d c w)", n=n)
        out1 = self.dense1(out1)
        out2 = self.dense2(out2)
        out3 = self.dense3(out3)
        combined = torch.cat([out1, out2, out3], -1)
        out = self.combiner(combined)
        return out

    def get_q(self, x1, x2, x3):
        n = x1.shape[0]
        head1 = torch.stft(x1, n_fft=self.n_fft, return_complex=True)
        head2 = torch.stft(x2, n_fft=self.n_fft, return_complex=True)
        head3 = torch.stft(x3, n_fft=self.n_fft, return_complex=True)
        reshaped1 = head1.flatten(start_dim=0, end_dim=1)
        reshaped2 = head2.flatten(start_dim=0, end_dim=1)
        reshaped3 = head3.flatten(start_dim=0, end_dim=1)

        probs = self.forward(reshaped1, reshaped2, reshaped3, n)
        return D.Bernoulli(probs)


class CNNWavelet(torch.nn.Module):
    def __init__(self, start, stop, wave, device):
        super(CNNWavelet, self).__init__()
        self.start = start
        self.stop = stop
        self.wave = wave
        self.device = device
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, 4, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            # torch.nn.Conv1d(4, 8, kernel_size=3, stride=2, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool1d(kernel_size=2, stride=2),
            # torch.nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.dense = torch.nn.Sequential(
            nn.Linear(416, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, n):
        return

    def get_q(self, x1, x2, x3):
        # n = x1.shape[0]
        head1 = pywt.wavedec(x1.cpu().numpy(), self.wave)
        head2 = pywt.wavedec(x2.cpu().numpy(), self.wave)
        head3 = pywt.wavedec(x3.cpu().numpy(), self.wave)
        head1 = head1[self.start : self.stop]
        head2 = head2[self.start : self.stop]
        head3 = head3[self.start : self.stop]

        c1 = [
            self.conv(torch.tensor(signal).unsqueeze(1).to(self.device))
            for signal in head1
        ]
        c2 = [
            self.conv(torch.tensor(signal).unsqueeze(1).to(self.device))
            for signal in head2
        ]
        c3 = [
            self.conv(torch.tensor(signal).unsqueeze(1).to(self.device))
            for signal in head3
        ]

        c1 = torch.cat(c1, axis=-1)
        c2 = torch.cat(c2, axis=-1)
        c3 = torch.cat(c3, axis=-1)

        flatten1 = torch.flatten(c1, start_dim=1)
        flatten2 = torch.flatten(c2, start_dim=1)
        flatten3 = torch.flatten(c3, start_dim=1)

        inputs = torch.cat([flatten1, flatten2, flatten3], axis=-1)
        probs = self.dense(inputs)
        return D.Bernoulli(probs)


class Dense3(torch.nn.Module):
    def __init__(
        self, in1_dim, in2_dim, in3_dim, hidden_dim1, latent_dim, hidden_dim2, device
    ):
        super(Dense3, self).__init__()
        self.in1_dim = in1_dim
        self.in2_dim = in2_dim
        self.in3_dim = in3_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.device = device

        self.dense1 = torch.nn.Sequential(
            nn.Linear(in1_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim1, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )

        self.dense2 = torch.nn.Sequential(
            nn.Linear(in2_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim1, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )

        self.dense3 = torch.nn.Sequential(
            nn.Linear(in3_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim1, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )

        self.combiner = torch.nn.Sequential(
            nn.Linear(3 * latent_dim, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2, x3):
        return

    def get_q(self, x1, x2, x3):
        # n = x1.shape[0]
        latent_rep1 = self.dense1(x1)
        latent_rep2 = self.dense2(x2)
        latent_rep3 = self.dense3(x3)
        combined = torch.cat([latent_rep1, latent_rep2, latent_rep3], -1)
        out = self.combiner(combined)
        return D.Bernoulli(out)
