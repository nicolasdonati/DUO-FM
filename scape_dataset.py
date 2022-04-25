import os
from pathlib import Path
import numpy as np
import potpourri3d as pp3d
import torch
from torch.utils.data import Dataset
import diffusion_net as dfn
from utils import auto_WKS, farthest_point_sample, square_distance
#
from tqdm import tqdm
from itertools import permutations
import Tools.mesh as qm
from Tools.utils import op_cpl


class ScapeDataset(Dataset):
    """
    Implementation of shape matching Dataset !WITH VTS! correspondence files (to compute ground-truth).
    This type of dataset is loaded if config['type'] = 'vts'
    It is called Scape Dataset because historically, SCAPE was re-meshed with vts files to track correspondence
    to the original SCAPE dataset. Any dataset using vts (re-meshed as in
    [Continuous and orientation-preserving correspondences via functional maps, Ren et al. 2018 TOG])
    falls into this category and can therefore be utilized via this class.

    ---Parameters:
    @ root_dir: root folder containing shapes_train and shapes_test folder
    @ name: name of the dataset. ex: scape-remeshed, or scape-anisotropic
    @ k_eig: number of Laplace-Beltrami eigenvectors loaded
    @ n_fmap: number of eigenvectors used for fmap computation
    @ n_cfmap: number of complex eigenvectors used for complex fmap computation
    @ with_wks: None if no WKS (C_in <= 3), else the number of WKS descriptors
    @ use_cache: cache for storing dataset (True by default)
    @ op_cache_dir: cache for diffusion net operators (from config['dataset']['cache_dir'])
    @ train: for train or test set

    ---At initialisation, loads:
    1) verts, faces and vts
    2) geometric operators (Laplacian, Gradient)
    3) (optional if C_in = 3) WKS descriptors (for best setting)
    4) (optional if n_cfmap = 0) complex operators (for orientation-aware unsupervised learning)

    ---When delivering an element of the dataset, yields a dictionary with:
    1) shape1 containing all necessary info for source shape
    2) shape2 containing all necessary info for target shape
    3) ground-truth functional map Cgt (obtained with vts files)
    """

    def __init__(self, root_dir, name="scape-remeshed",
                 k_eig=128, n_fmap=30, n_cfmap=20,
                 with_wks=None, with_sym=False,
                 use_cache=True, op_cache_dir=None,
                 train=True):

        self.k_eig = k_eig
        self.n_fmap = n_fmap
        self.n_cfmap = n_cfmap
        self.root_dir = root_dir
        self.cache_dir = root_dir
        self.op_cache_dir = op_cache_dir
        self.with_sym = with_sym
        self.find_sym = find_sym

        # check the cache
        split = "train" if train else "test"
        wks_suf = "" if with_wks is None else "wks_"
        sym_suf = "" if not with_sym else "sym_"
        if use_cache:
            load_cache = os.path.join(self.cache_dir, f"cache_{name}_{sym_suf}{wks_suf}{split}.pt")
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                (
                    # main
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    # diffNet
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    # Q-Maps
                    self.cevecs_list,
                    self.cevals_list,
                    self.spec_grad_list,
                    # misc
                    self.used_shapes,
                    self.vts_list
                ) = torch.load(load_cache)

                if self.find_sym:
                    self.combinations = list(range(len(self.verts_list)))
                else:
                    self.combinations = list(permutations(range(len(self.verts_list)), 2))
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes
        # define files and order
        shapes_split = "shapes_" + split
        self.used_shapes = sorted([x.stem for x in (Path(root_dir) / shapes_split).iterdir() if 'DS_' not in x.stem])

        # set combinations (if we only want to find self-symmetries)
        if self.find_sym:
            self.combinations = list(range(len(self.used_shapes)))
        else:
            self.combinations = list(permutations(range(len(self.used_shapes)), 2))

        #
        mesh_dirpath = Path(root_dir) / shapes_split
        vts_dirpath = Path(root_dir) / "corres"

        # Get all the files
        ext = '.off'
        self.verts_list = []
        self.faces_list = []
        self.vts_list = []
        if with_sym:
            vts_sym_list = []

        # Load the actual files
        for shape_name in tqdm(self.used_shapes):
            #print("loading mesh " + str(shape_name))

            verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}{ext}"))  # off obj
            # verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}.off"))

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            self.verts_list.append(verts)
            self.faces_list.append(faces)

            # vts
            vts = np.loadtxt(str(vts_dirpath / f"{shape_name}.vts"), dtype=int) - 1
            self.vts_list.append(vts)
            if with_sym:
                vts_sym = np.loadtxt(str(vts_dirpath / f"{shape_name}.sym.vts"), dtype=int) - 1
                vts_sym_list.append(vts_sym)

        # we store vts_sym also in vts_list (for saving)
        if with_sym:
            self.vts_list = [self.vts_list, vts_sym_list]


        # Precompute operators
        (
            self.frames_list,
            self.massvec_list,
            self.L_list,
            self.evals_list,
            self.evecs_list,
            self.gradX_list,
            self.gradY_list,
        ) = dfn.geometry.get_all_operators(
            self.verts_list,
            self.faces_list,
            k_eig=self.k_eig,
            op_cache_dir=self.op_cache_dir,
        )

        # compute wks descriptors if required (and replace vertices field with it)
        if with_wks is not None:
            print("compute WKS descriptors")
            for i in tqdm(range(len(self.used_shapes))):
                self.verts_list[i] = auto_WKS(self.evals_list[i], self.evecs_list[i], with_wks).float()

        # now we also need to get the complex Laplacian and the spectral gradients
        print("loading operators for Q...")
        self.cevecs_list = []
        self.cevals_list = []
        self.spec_grad_list = []
        for shape_name in tqdm(self.used_shapes):

            # case where computing complex spectral is not possible (non manifoldness, borders, point cloud, ...)
            if n_cfmap == 0:
                self.cevecs_list += [None]
                self.cevals_list += [None]
                self.spec_grad_list += [None]
                continue

            # else load mesh and compute complex laplacian and gradient operators
            mesh_for_Q = qm.mesh(str(mesh_dirpath / f"{shape_name}{ext}"),
                                 spectral=0, complex_spectral=n_cfmap, spectral_folder=root_dir)
            #
            mesh_for_Q.grad_vert_op()
            mesh_for_Q.grad_vc = op_cpl(mesh_for_Q.gradv.T).T

            self.cevecs_list += [mesh_for_Q.ceig]
            self.cevals_list += [mesh_for_Q.cvals]
            self.spec_grad_list += [np.linalg.pinv(mesh_for_Q.ceig) @ mesh_for_Q.grad_vc]
            # print(self.frames_list[-1].shape)
        print('done')

        # save to cache
        if use_cache:
            dfn.utils.ensure_dir_exists(self.cache_dir)
            torch.save(
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    #
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    #
                    self.cevecs_list,
                    self.cevals_list,
                    self.spec_grad_list,
                    #
                    self.used_shapes,
                    self.vts_list
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):

        # get indexes
        if self.find_sym:
            idx1, idx2 = idx, idx
        else:
            idx1, idx2 = self.combinations[idx]

        # if augmentation with symmetries, get vts list and vts sym list
        if self.with_sym and len(self.vts_list) == 2:
            self.vts_list, self.vts_sym_list = self.vts_list

        shape1 = {
            "xyz": self.verts_list[idx1],
            "faces": self.faces_list[idx1],
            "frames": self.frames_list[idx1],
            #
            "mass": self.massvec_list[idx1],
            "L": self.L_list[idx1],
            "evals": self.evals_list[idx1],
            "evecs": self.evecs_list[idx1],
            "gradX": self.gradX_list[idx1],
            "gradY": self.gradY_list[idx1],
            #
            "cevecs": self.cevecs_list[idx1],
            "cevals": self.cevals_list[idx1],
            "spec_grad": self.spec_grad_list[idx1],
            #
            "vts": self.vts_list[idx1],
            "name": self.used_shapes[idx1],
        }

        shape2 = {
            "xyz": self.verts_list[idx2],
            "faces": self.faces_list[idx2],
            "frames": self.frames_list[idx2],
            #
            "mass": self.massvec_list[idx2],
            "L": self.L_list[idx2],
            "evals": self.evals_list[idx2],
            "evecs": self.evecs_list[idx2],
            "gradX": self.gradX_list[idx2],
            "gradY": self.gradY_list[idx2],
            #
            "cevecs": self.cevecs_list[idx2],
            "cevals": self.cevals_list[idx2],
            "spec_grad": self.spec_grad_list[idx2],
            #
            "vts": self.vts_list[idx2],
            "name": self.used_shapes[idx2],
        }

        # add sym vts if necessary
        if self.with_sym:
            shape1["vts_sym"], shape2["vts_sym"] = self.vts_sym_list[idx1], self.vts_sym_list[idx2]

        # Compute fmap
        evec_1, evec_2 = shape1["evecs"][:, :self.n_fmap], shape2["evecs"][:, :self.n_fmap]
        vts1, vts2 = self.vts_list[idx1], self.vts_list[idx2]

        C_gt = torch.pinverse(evec_2[vts2]) @ evec_1[vts1]

        return {"shape1": shape1, "shape2": shape2, "C_gt": C_gt}


def shape_to_device(dict_shape, device):
    names_to_device = ["xyz", "faces", "mass", "evals", "evecs", "gradX", "gradY",
                       "cevecs", "cevals", "spec_grad"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                if v[name] is not None:
                    v[name] = v[name].to(device)  # .float()
            dict_shape[k] = v
        else:
            dict_shape[k] = v.to(device)

    return dict_shape
