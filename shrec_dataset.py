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


class ShrecDataset(Dataset):
    """
    Implementation of shape matching Dataset !WITH GroundTruth! correspondence files (between given pairs).
    It is called Shrec Dataset because historically, SHREC'19 used that system.
    Any dataset using gt files falls into this category and can therefore be utilized via this class.

    ---Parameters:
    @
    @
    @

    ---At initialisation, loads:
    1) verts, faces and ground-truths
    2) geometric operators (Laplacian, Gradient)
    3) (optional if C_in = 3) WKS descriptors (for best setting)
    4) (optional if n_cfmap = 0) complex operators (for orientation-aware unsupervised learning)

    ---When delivering an element of the dataset, yields a dictionary with:
    1) shape1 containing all necessary info for source shape
    2) shape2 containing all necessary info for target shape
    3) ground-truth functional map Cgt (obtained with gt files)
    """

    def __init__(self, root_dir, name="remeshed",
                 k_eig=128, n_fmap=30, n_cfmap=20,
                 with_wks=None,
                 use_cache=True, op_cache_dir=None,
                 train=False):

        self.k_eig = k_eig
        self.n_fmap = n_fmap
        self.n_cfmap = n_cfmap
        self.root_dir = root_dir
        self.cache_dir = root_dir
        self.op_cache_dir = op_cache_dir

        # check the cache
        split = "train" if train else "test"
        wks_suf = "" if with_wks is None else "wks_"
        if use_cache:
            load_cache = os.path.join(self.cache_dir, f"cache_{name}_{wks_suf}{split}.pt")
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
                    self.gt_list
                ) = torch.load(load_cache)
                self.combinations = np.loadtxt(os.path.join(Path(root_dir), 'test_pairs.txt'), delimiter=',').astype(int)
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes
        # define files and order
        shapes_split = "shapes_" + split
        self.used_shapes = sorted([x.stem for x in (Path(root_dir) / shapes_split).iterdir() if 'DS_' not in x.stem])

        # set combinations
        self.combinations = np.loadtxt(os.path.join(Path(root_dir), 'test_pairs.txt'), delimiter=',').astype(int)

        #
        mesh_dirpath = Path(root_dir) / shapes_split
        gt_dirpath = Path(root_dir) / "groundtruth"

        # Get all the files
        self.verts_list = []
        self.faces_list = []
        self.vts_list = []

        # Load the actual files
        for shape_name in tqdm(self.used_shapes):
            #print("loading mesh " + str(shape_name))

            verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}.off"))

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            self.verts_list.append(verts)
            self.faces_list.append(faces)

        # Load ground-truths
        self.gt_list = {}
        for i, j in tqdm(self.combinations):
            gt_map = np.loadtxt(str(gt_dirpath / f"{i}_{j}.map"), dtype=np.int32).astype(int)
            self.gt_list[(i, j)] = gt_map

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

        # Compute wks descriptors if required (and replace vertices field with it)
        if with_wks is not None:
            print("compute WKS descriptors")
            for i in tqdm(range(len(self.used_shapes))):
                self.verts_list[i] = auto_WKS(self.evals_list[i], self.evecs_list[i], with_wks).float()

        # Now we also need to get the complex Laplacian and the spectral gradients
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
            mesh_for_Q = qm.mesh(str(mesh_dirpath / f"{shape_name}.off"),
                                 spectral=0, complex_spectral=n_cfmap, spectral_folder=root_dir)
            #
            mesh_for_Q.grad_vert_op()
            mesh_for_Q.grad_vc = op_cpl(mesh_for_Q.gradv.T).T

            self.cevecs_list += [mesh_for_Q.ceig]
            self.cevals_list += [mesh_for_Q.cvals]
            self.spec_grad_list += [np.linalg.pinv(mesh_for_Q.ceig) @ mesh_for_Q.grad_vc]

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
                    self.gt_list
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx1, idx2 = self.combinations[idx]

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
            # "vts": self.vts_list[idx1],
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
            # "vts": self.vts_list[idx2],
            "name": self.used_shapes[idx2],
        }

        # Compute fmap
        evec_1, evec_2 = shape1["evecs"][:, :self.n_fmap], shape2["evecs"][:, :self.n_fmap]

        gt = self.gt_list[(idx1, idx2)]
        shape1['gt'] = gt  # add it on shape 1 for eval

        C_gt = torch.pinverse(evec_2[gt]) @ evec_1
        # C_gt = torch.eye(self.n_fmap)  # if we don't want to compute the map at all

        return {"shape1": shape1, "shape2": shape2, "gt": gt, "C_gt": C_gt}


def shape_to_device(dict_shape, device):
    names_to_device = ["xyz", "faces", "mass", "evals", "evecs", "gradX", "gradY",
                       "cevecs", "cevals", "spec_grad"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                if v[name] is not None:
                    v[name] = v[name].to(device)
            dict_shape[k] = v
        else:
            dict_shape[k] = v.to(device)

    return dict_shape
