import argparse
import yaml
import os
import torch
from scape_dataset import ScapeDataset, shape_to_device
from shrec_dataset import ShrecDataset, shape_to_device
from sym_dataset import SymDataset, shape_to_device
from model import DQFMNet
from utils import DQFMLoss, augment_batch, augment_batch_sym
#
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusion_net.utils import toNP


def train_net(cfg):
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    dataset_path = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_train"])

    save_dir_name = f'trained_{cfg["dataset"]["name"]}'
    model_save_path = os.path.join(base_path, f"data/{save_dir_name}/ep" + "_{}.pth")
    if not os.path.exists(os.path.join(base_path, f"data/{save_dir_name}/")):
        os.makedirs(os.path.join(base_path, f"data/{save_dir_name}/"))

    # decide on the use of WKS descriptors
    with_wks = None if cfg["fmap"]["C_in"] <= 3 else cfg["fmap"]["C_in"]

    # is it a dataset to find self-symmetries
    find_sym = False if not "find_sym" in cfg["dataset"] else cfg["dataset"]["find_sym"]

    # create dataset
    # dataset to find self symmetries
    if find_sym:
        train_dataset = SymDataset(dataset_path, name=cfg["dataset"]["name"] + "-" + cfg["dataset"]["subset"],
                                   k_eig=cfg["fmap"]["k_eig"],
                                   n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
                                   with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                   use_cache=True, op_cache_dir=op_cache_dir)

    # standard structured (source <> target) vts dataset
    elif cfg["dataset"]["type"] == "vts":
        train_dataset = ScapeDataset(dataset_path, name=cfg["dataset"]["name"]+"-"+cfg["dataset"]["subset"],
                                     k_eig=cfg["fmap"]["k_eig"],
                                     n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
                                     with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                     use_cache=True, op_cache_dir=op_cache_dir,
                                     find_sym=find_sym)

    # standard structured (source <> target) dataset with gt
    elif cfg["dataset"]["type"] == "gt":
        train_dataset = ShrecDataset(dataset_path, name=cfg["dataset"]["name"]+"-"+cfg["dataset"]["subset"],
                                     k_eig=cfg["fmap"]["k_eig"],
                                     n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
                                     with_wks=with_wks,
                                     use_cache=True, op_cache_dir=op_cache_dir, train=True)

    # else not implemented
    else:
        raise NotImplementedError("dataset not implemented!")

    # train loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True)

    # define model
    dqfm_net = DQFMNet(cfg).to(device)
    lr = float(cfg["optimizer"]["lr"])
    optimizer = torch.optim.Adam(dqfm_net.parameters(), lr=lr, betas=(cfg["optimizer"]["b1"], cfg["optimizer"]["b2"]))
    criterion = DQFMLoss(w_gt=cfg["loss"]["w_gt"],
                         w_ortho=cfg["loss"]["w_ortho"],
                         w_Qortho=cfg["loss"]["w_Qortho"]).to(device)

    # Training loop
    print("start training")
    iterations = 0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        if epoch % cfg["optimizer"]["decay_iter"] == 0:
            lr *= cfg["optimizer"]["decay_factor"]
            print(f"Decaying learning rate, new one: {lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        dqfm_net.train()
        for i, data in tqdm(enumerate(train_loader)):
            data = shape_to_device(data, device)

            # data augmentation (if we have wks descriptors we use sym augmentation)
            if True and with_wks is None:
                data = augment_batch(data, rot_x=180, rot_y=180, rot_z=180,
                                     std=0.01, noise_clip=0.05,
                                     scale_min=0.9, scale_max=1.1)
            elif "with_sym" in cfg["dataset"] and cfg["dataset"]["with_sym"]:
                data = augment_batch_sym(data, rand=not find_sym)
                #data = augment_batch_sym(data, rand=False) #always symmetrize

            # prepare iteration data
            C_gt = data["C_gt"].unsqueeze(0)

            # do iteration
            C_pred, Q_pred = dqfm_net(data)
            loss = criterion(C_gt, C_pred, Q_pred)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log
            iterations += 1

            log_batch = (i + 1) % cfg["misc"]["log_interval"] == 0
            log_iter = (iterations + 1) % cfg["misc"]["log_interval"] == 0
            if log_batch or (find_sym and log_iter):
                print(f"#epoch:{epoch}, #batch:{i + 1}, #iteration:{iterations}, loss:{loss}")
                print("#gt:{:06.4f} | #o:{:06.4f} | #Qo:{:06.4f}".format(criterion.gt_loss,
                                                                         criterion.ortho_loss,
                                                                         criterion.Qortho_loss))
                # print all losses

                # plot image
                plt.subplot(131)
                plt.imshow(toNP(C_pred[0]))
                plt.title('C pred')
                plt.subplot(132)
                plt.imshow(toNP(C_gt[0]))
                plt.title('C gt')
                plt.subplot(133)
                plt.imshow(np.abs(toNP(Q_pred[0])))
                plt.title('Q pred')
                plt.savefig('img/{}_ep-{:02d}_it-{:05d}'.format(cfg["dataset"]["name"], epoch, i+1), dpi=150)
                plt.clf()

        # save model
        if (epoch + 1) % cfg["misc"]["checkpoint_interval"] == 0:
            torch.save(dqfm_net.state_dict(), model_save_path.format(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of DQFM model.")

    parser.add_argument("--config", type=str, default="scape_r", help="Config file name")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    train_net(cfg)
