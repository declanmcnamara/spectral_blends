import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from modules import DESI3CameraDatset, DESICoaddDatset


def make_dataloader3(**kwargs):
    device = kwargs["device"]
    data_dir = kwargs["data_dir"]
    data_save_dir = kwargs["data_save_dir"]
    x1s = []
    x2s = []
    x3s = []
    zs = []
    for i in range(kwargs["n_dream_batches"]):
        x1 = np.load("{}/{}/three_camera_x1_{}.npy".format(data_dir, data_save_dir, i))
        x2 = np.load("{}/{}/three_camera_x2_{}.npy".format(data_dir, data_save_dir, i))
        x3 = np.load("{}/{}/three_camera_x3_{}.npy".format(data_dir, data_save_dir, i))
        z = np.load("{}/{}/three_camera_z_{}.npy".format(data_dir, data_save_dir, i))
        x1, x2, x3 = (
            torch.tensor(x1).float(),
            torch.tensor(x2).float(),
            torch.tensor(x3).float(),
        )
        z = torch.tensor(z).float()

        x1s.append(x1)
        x2s.append(x2)
        x3s.append(x3)
        zs.append(z)
    x1_full = torch.cat(x1s, 0)
    x2_full = torch.cat(x2s, 0)
    x3_full = torch.cat(x3s, 0)
    cut1 = x1_full.shape[1]
    cut2 = cut1 + x2_full.shape[1]
    z_full = torch.cat(zs, 0)
    x_full = torch.cat([x1_full, x2_full, x3_full], -1)
    my_set = DESI3CameraDatset(x_full, z_full, cut1, cut2, device)
    return DataLoader(my_set, batch_size=kwargs["training_batch_size"], shuffle=True)


def make_dataloader(**kwargs):
    data_dir = kwargs["data_dir"]
    data_save_dir = kwargs["data_save_dir"]
    device = kwargs["device"]
    xs = []
    zs = []
    for i in range(kwargs["n_dream_batches"]):
        x = np.load("{}/{}/coadded_x_{}.npy".format(data_dir, data_save_dir, i))
        z = np.load("{}/{}/coadded_z_{}.npy".format(data_dir, data_save_dir, i))
        x = torch.tensor(x).float()
        z = torch.tensor(z).float()

        xs.append(x)
        zs.append(z)
    x_full = torch.cat(xs, 0)
    z_full = torch.cat(zs, 0)
    my_set = DESICoaddDatset(x_full, z_full, device)
    return DataLoader(my_set, batch_size=kwargs["training_batch_size"], shuffle=True)


def training_loop(encoder, scheduler, favi_model, dataloader, writer, j, **kwargs):
    device = kwargs["device"]
    epoch_loss = []
    # offset = dataloader.batch_size * j
    for batch, (x, z) in enumerate(dataloader):
        # Compute loss from encoder
        qzx = encoder.get_q(x)
        scheduler.zero_grad()
        loss = -1 * qzx.log_prob(z.view(-1, 1)).mean()
        loss.backward()
        scheduler.step_and_update_lr()

        print("Epoch {} Minibatch {}: Loss {}".format(j, batch, loss.item()))
        # writer.add_scalar("Loss", loss.item(), offset + batch)
        epoch_loss.append(loss.item())

    avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    print("AVERAGE EPOCH LOSS: {}".format(avg_epoch_loss))

    # TEST LOSS
    x, z = favi_model.dream(1000)
    x, z = torch.tensor(x).to(device).float(), torch.tensor(z).to(device).float()
    qzx = encoder.get_q(x)
    loss = -1 * qzx.log_prob(z.view(-1, 1)).mean()
    print("TEST LOSS: {}".format(loss.item()))
    return loss.item()


def training_loop3(encoder, scheduler, favi_model, dataloader, writer, j, **kwargs):
    device = kwargs["device"]
    epoch_loss = []
    # offset = dataloader.batch_size * j
    for batch, (x1, x2, x3, z) in enumerate(dataloader):
        # Compute loss from encoder
        qzx = encoder.get_q(x1, x2, x3)
        scheduler.zero_grad()
        loss = -1 * qzx.log_prob(z.view(-1, 1)).mean()
        loss.backward()
        scheduler.step_and_update_lr()

        print("Epoch {} Minibatch {}: Loss {}".format(j, batch, loss.item()))
        # writer.add_scalar("Losss", loss.item(), offset + batch)
        epoch_loss.append(loss.item())

    avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    print("AVERAGE EPOCH LOSS: {}".format(avg_epoch_loss))

    # TEST LOSS
    batch_size = kwargs["training_batch_size"]
    x1, x2, x3, z = favi_model.dream_fast(batch_size)
    x1, x2, x3 = (
        torch.tensor(x1).float().to(device),
        torch.tensor(x2).float().to(device),
        torch.tensor(x3).float().to(device),
    )
    z = torch.tensor(z).float().to(device)

    qzx = encoder.get_q(x1, x2, x3)
    loss = -1 * qzx.log_prob(z.view(-1, 1)).mean()
    print("Loss is {}".format(loss.item()))
    return loss.item()


def favi_step(encoder, scheduler, favi_model, writer, **kwargs):
    replay = kwargs["replay"]
    device = kwargs["device"]
    batch_size = kwargs["training_batch_size"]
    x, z = favi_model.dream(batch_size)
    x, z = torch.tensor(x).to(device).float(), torch.tensor(z).to(device).float()

    to_return = None
    for j in range(replay):
        qzx = encoder.get_q(x)
        scheduler.zero_grad()
        loss = -1 * qzx.log_prob(z.view(-1, 1)).mean()
        loss.backward()
        scheduler.step_and_update_lr()
        print("On Replay step {}, loss is {}".format(j, loss.item()))

        if j == 0:
            to_return = loss.item()
    return to_return


def favi_step3(encoder, scheduler, favi_model, writer, **kwargs):
    replay = kwargs["replay"]
    device = kwargs["device"]
    batch_size = kwargs["training_batch_size"]
    x1, x2, x3, z = favi_model.dream_fast(batch_size)
    x1, x2, x3 = (
        torch.tensor(x1).float().to(device),
        torch.tensor(x2).float().to(device),
        torch.tensor(x3).float().to(device),
    )
    z = torch.tensor(z).float().to(device)

    to_return = None
    for j in range(replay):
        qzx = encoder.get_q(x1, x2, x3)
        scheduler.zero_grad()
        loss = -1 * qzx.log_prob(z.view(-1, 1)).mean()
        loss.backward()
        print("On Replay step {}, loss is {}".format(j, loss.item()))
        scheduler.step_and_update_lr()
        if j == 0:
            to_return = loss.item()

    return to_return
