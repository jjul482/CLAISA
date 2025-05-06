
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import os
import sys
import pickle
from tqdm import tqdm
import math
import logging
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

import trainers, datasets, utils
#import models
import cl_models as models
from config import Config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='denmark', help='dataset name')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_samples', type=int, default=16)
parser.add_argument('--adapter_num', type=int, default=10)
parser.add_argument('--max_seqlen', type=int, default=120)
args = parser.parse_args()

cf = Config()
cf.dataset_name = args.dataset
cf.gpu = args.gpu
cf.max_epochs = args.epochs
cf.batch_size = args.batch_size
cf.n_samples = args.n_samples
cf.adapter_num = args.adapter_num
cf.max_seqlen = args.max_seqlen
cf.datadir = f"./data/ct_dma/{cf.dataset_name}"
cf.filename = f"{cf.dataset_name}"\
        + f"-data_size-{cf.lat_size}-{cf.lon_size}-{cf.sog_size}-{cf.cog_size}"\
        + f"-embd_size-{cf.n_lat_embd}-{cf.n_lon_embd}-{cf.n_sog_embd}-{cf.n_cog_embd}"\
        + f"-seqlen-{cf.init_seqlen}-{cf.max_seqlen}"
cf.savedir = "./results/"+cf.filename+"/"
cf.log_path = f"./results/{cf.dataset_name}.txt"
cf.ckpt_path = os.path.join(cf.savedir,"model.pt")  

if cf.dataset_name.lower() == 'gulfofmexico':
    #gulfofmexico
    cf.lat_min = 27.0
    cf.lat_max = 30.0
    cf.lon_min = -90.5
    cf.lon_max = -87.5
elif cf.dataset_name.lower() == 'eastcoast':
    cf.lat_min = 26.23
    cf.lat_max = 45.67
    cf.lon_min = -81.51
    cf.lon_max = -50.99
elif cf.dataset_name.lower() == 'piraeus':
    cf.lat_min = 37.5
    cf.lat_max = 38.1
    cf.lon_min = 23.0
    cf.lon_max = 23.9
elif cf.dataset_name.lower() == 'denmark':
    cf.lat_min = 55.0
    cf.lat_max = 58.0
    cf.lon_min = 10.0
    cf.lon_max = 13.0
TB_LOG = cf.tb_log
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter()


def plot_test(Vs, config, phase):
    #Vs = Vs['train']
    LAT_MIN,LAT_MAX,LON_MIN,LON_MAX = config.lat_min,config.lat_max,config.lon_min,config.lon_max
    LAT_RANGE = LAT_MAX - LAT_MIN
    LON_RANGE = LON_MAX - LON_MIN
    FIG_DPI = 150
    FIG_W = 960
    FIG_H = int(960*LAT_RANGE/LON_RANGE)
    plt.figure(figsize=(FIG_W/FIG_DPI, FIG_H/FIG_DPI), dpi=FIG_DPI)
    cmap = plt.cm.get_cmap('Blues')
    N = len(Vs)
    for d_i in range(N):
        key = Vs[d_i]
        c = cmap(float(d_i)/(N-1))
        #tmp = key['traj']
        tmp = key[0]
        #print(tmp)
        v_lat = tmp[:,0]*LAT_RANGE + LAT_MIN
        v_lon = tmp[:,1]*LON_RANGE + LON_MIN
    #     plt.plot(v_lon,v_lat,linewidth=0.8)
        plt.plot(v_lon,v_lat,color=c,linewidth=0.8)

    plt.xlim([LON_MIN,LON_MAX])
    plt.ylim([LAT_MIN,LAT_MAX])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"{phase}.png")

# make deterministic
utils.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item() * 2

if __name__ == "__main__":

    device = cf.device
    init_seqlen = cf.init_seqlen

    ## Logging
    # ===============================
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: ' + cf.savedir)
    else:
        print('======= Directory to store trained models: ' + cf.savedir)
    utils.new_log(cf.savedir, "log")

    ## Data
    # ===============================
    moving_threshold = 0.05
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}
    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            l_pred_errors = pickle.load(f)
        for V in l_pred_errors:
            try:
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except:
                moving_idx = len(V["traj"]) - 1  # This track will be removed
            V["traj"] = V["traj"][moving_idx:, :]
            if len(V["traj"]) > cf.max_seqlen:
                V["traj"] = V["traj"][:cf.max_seqlen]
        Data[phase] = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
        print(f"Length: {len(Data[phase])}")
        print("Creating pytorch dataset...")
        # Later in this scipt, we will use inputs = x[:-1], targets = x[1:], hence
        # max_seqlen = cf.max_seqlen + 1.
        if cf.mode in ("pos_grad", "grad"):
            aisdatasets[phase] = datasets.AISDataset_grad(Data[phase],
                                                          max_seqlen=cf.max_seqlen+1,
                                                          device=cf.device)
        else:
            aisdatasets[phase] = datasets.AISDataset(Data[phase],
                                                     max_seqlen=cf.max_seqlen+1,
                                                     device=cf.device)
        if phase == "test":
            shuffle = False
        else:
            shuffle = True
        plot_test(aisdatasets[phase], cf, phase)
        aisdls[phase] = DataLoader(aisdatasets[phase],
                                   batch_size=cf.batch_size,
                                   shuffle=shuffle)
    cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen

    ## Model
    # ===============================
    model = models.AdapterModel(cf, partition_model=None)
    print(f"Device: {cf.device}")
    ## Trainer
    # ===============================
    trainer = trainers.Trainer(
        model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.savedir, device=cf.device, aisdls=aisdls, INIT_SEQLEN=init_seqlen)
    trainer.plot_valid()

    ## Training
    # ===============================
    if cf.retrain:
        trainer.train()

    ## Evaluation
    # ===============================
    # Load the best model
    model.load_state_dict(torch.load(cf.ckpt_path))

    v_ranges = torch.tensor([2, 3, 0, 0]).to(cf.device)
    v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(cf.device)
    max_seqlen = init_seqlen + 6 * 4

    model.eval()
    l_min_errors, l_mean_errors, l_masks = [], [], []
    pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))
    with torch.no_grad():
        for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            seqs_init = seqs[:, :init_seqlen, :].to(cf.device)
            masks = masks[:, :max_seqlen].to(cf.device)
            batchsize = seqs.shape[0]
            error_ens = torch.zeros((batchsize, max_seqlen - cf.init_seqlen, cf.n_samples)).to(cf.device)
            for i_sample in range(cf.n_samples):
                preds = trainers.sample(model,
                                        seqs_init,
                                        max_seqlen - init_seqlen,
                                        temperature=1.0,
                                        sample=True,
                                        sample_mode=cf.sample_mode,
                                        r_vicinity=cf.r_vicinity,
                                        top_k=cf.top_k)
                inputs = seqs[:, :max_seqlen, :].to(cf.device)
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
                d = utils.haversine(input_coords, pred_coords) * masks
                error_ens[:, :, i_sample] = d[:, cf.init_seqlen:]
            # Accumulation through batches
            l_min_errors.append(error_ens.min(dim=-1))
            l_mean_errors.append(error_ens.mean(dim=-1))
            l_masks.append(masks[:, cf.init_seqlen:])

    #l_mean = [x.values for x in l_mean_errors]
    l_mean = l_mean_errors
    l_min = [x.values for x in l_min_errors]
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_min, dim=0) * m_masks
    mean_errors = torch.cat(l_mean, dim=0) * m_masks
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
    pred_errors = pred_errors.detach().cpu().numpy()

    # Print final errors?

    ## Plot
    # ===============================
    plt.figure(figsize=(9, 6), dpi=150)
    v_times = np.arange(len(pred_errors)) / 6
    plt.plot(v_times, pred_errors)

    timestep = 6
    plt.plot(1, pred_errors[timestep], "o")
    plt.plot([1, 1], [0, pred_errors[timestep]], "r")
    plt.plot([0, 1], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(1.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 12
    plt.plot(2, pred_errors[timestep], "o")
    plt.plot([2, 2], [0, pred_errors[timestep]], "r")
    plt.plot([0, 2], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(2.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 18
    plt.plot(3, pred_errors[timestep], "o")
    plt.plot([3, 3], [0, pred_errors[timestep]], "r")
    plt.plot([0, 3], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(3.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)
    plt.xlabel("Time (hours)")
    plt.ylabel("Prediction errors (km)")
    plt.xlim([0, 4])
    plt.ylim([0, 5])
    # plt.ylim([0,pred_errors.max()+0.5])
    plt.savefig(cf.savedir + "prediction_error.png")