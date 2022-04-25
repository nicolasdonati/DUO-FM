import argparse
import yaml
import os
import torch
from scape_dataset import ScapeDataset, shape_to_device
from shrec_dataset import ShrecDataset, shape_to_device
from sym_dataset import SymDataset, shape_to_device
from model import DQFMNet
#
import numpy as np
import scipy.io as sio
from utils import read_geodist, augment_batch, augment_batch_sym
from Tools.utils import fMap2pMap, zo_fmap
from diffusion_net.utils import toNP

def eval_geodist(cfg, shape1, shape2, T):
    path_geodist_shape2 = os.path.join(cfg['dataset']['root_geodist'],shape2['name']+'.mat')
    MAT_s = sio.loadmat(path_geodist_shape2)

    G_s, SQ_s = read_geodist(MAT_s)

    n_s = G_s.shape[0]
    # print(SQ_s[0])
    if 'vts' in shape1:
        phi_t = shape1['vts']
        phi_s = shape2['vts']
    elif 'gt' in shape1:
        phi_t = np.arange(shape1['xyz'].shape[0])
        phi_s = shape1['gt']
    else:
        raise NotImplementedError("cannot find ground-truth correspondence for eval")

    # find pairs of points for geodesic error
    pmap = T
    ind21 = np.stack([phi_s, pmap[phi_t]], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[n_s, n_s])

    errC = np.take(G_s, ind21) / SQ_s
    print('{}-->{}: {:.4f}'.format(shape1['name'], shape2['name'], np.mean(errC)))
    return errC

def eval_net(args, model_path, predictions_name):
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    dataset_path = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_test"])

    # decide on the use of WKS descriptors
    with_wks = None if cfg["fmap"]["C_in"] <= 3 else cfg["fmap"]["C_in"]

    # is it a dataset to find self-symmetries
    find_sym = False if not "find_sym" in cfg["dataset"] else cfg["dataset"]["find_sym"]
    # print(find_sym)

    # create dataset
    if find_sym:
        test_dataset = SymDataset(dataset_path, name=cfg["dataset"]["name"]+"-"+cfg["dataset"]["subset"],
                                  k_eig=cfg["fmap"]["k_eig"],
                                  n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
                                  with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                  use_cache=True, op_cache_dir=op_cache_dir,
                                  train=False)

    elif cfg["dataset"]["type"] == "vts":
        test_dataset = ScapeDataset(dataset_path, name=cfg["dataset"]["name"] + "-" + cfg["dataset"]["subset"],
                                    k_eig=cfg["fmap"]["k_eig"],
                                    n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
                                    with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                    use_cache=True, op_cache_dir=op_cache_dir, train=False)

    elif cfg["dataset"]["type"] == "gt":
        test_dataset = ShrecDataset(dataset_path, name=cfg["dataset"]["name"] + "-" + cfg["dataset"]["subset"],
                                    k_eig=cfg["fmap"]["k_eig"],
                                    n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
                                    with_wks=with_wks,
                                    use_cache=True, op_cache_dir=op_cache_dir, train=False)

    else:
        raise NotImplementedError("dataset not implemented!")

    # test loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, shuffle=False)

    # define model
    dqfm_net = DQFMNet(cfg).to(device)
    print(model_path)
    dqfm_net.load_state_dict(torch.load(model_path, map_location=device))
    dqfm_net.eval()

    to_save_list = []
    errs = []
    for i, data in enumerate(test_loader):

        data = shape_to_device(data, device)

        # data augmentation (if using wks descriptors augment with sym)
        if with_wks is None:
            data = augment_batch(data, rot_x=180, rot_y=180, rot_z=180,
                                 std=0.01, noise_clip=0.05,
                                 scale_min=0.9, scale_max=1.1)
        elif "with_sym" in cfg["dataset"] and cfg["dataset"]["with_sym"]:
            data = augment_batch_sym(data, rand=False)

        # prepare iteration data
        C_gt = data["C_gt"].unsqueeze(0)

        # do iteration
        C_pred, Q_pred = dqfm_net(data)

        # save maps
        name1, name2 = data["shape1"]["name"], data["shape2"]["name"]
        # print(name1, name2)
        if Q_pred is None:
            to_save_list.append((name1, name2, C_pred.detach().cpu().squeeze(0),
                                 None, C_gt.detach().cpu().squeeze(0)))
        else:
            to_save_list.append((name1, name2, C_pred.detach().cpu().squeeze(0),
                                 Q_pred.detach().cpu().squeeze(0), C_gt.detach().cpu().squeeze(0)))

        # compute geodesic error (transpose C12 to get C21, and thus T12)
        shape1, shape2 = data["shape1"], data["shape2"]

        # with zo ref
        # C_ref = zo_fmap(toNP(shape1['evecs']), toNP(shape2['evecs']), toNP(C_pred.squeeze(0)).T, k_final=100, k_step=3)
        # T_pred = fMap2pMap(toNP(shape2['evecs']), toNP(shape1['evecs']), C_ref)

        # without zo ref
        T_pred = fMap2pMap(toNP(shape2['evecs']), toNP(shape1['evecs']), toNP(C_pred.squeeze(0)).T)
        err = eval_geodist(args, shape1, shape2, T_pred)
        #errs += [np.mean(err)]
        errs += [err]

    np.save("allmaps.npy", errs)
    print('total geodesic error: ', np.mean(errs))
    torch.save(to_save_list, predictions_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the eval of DQFM model.")

    parser.add_argument("--config", type=str, default="scape_r", help="Config file name")

    parser.add_argument("--model_path", type=str, default="data/trained_scape/ep_5.pth",
                         help="path to saved model")
    #parser.add_argument("--model_path",type=str,default="data/saved_models_remeshed/ep_4.pth",
    #                    help="path to saved model")
    parser.add_argument("--predictions_name", type=str, default="data/pred.pth",
                        help="name of the prediction file")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    eval_net(cfg, args.model_path, args.predictions_name)
