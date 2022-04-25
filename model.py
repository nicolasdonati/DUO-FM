from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

# feature extractor
from diffusion_net.layers import DiffusionNet
from spectral_net.specConv import SpecConvModule
from point_net2.pointnet2_sem_seg import get_model
from point_net2.pointnet2_utils import farthest_point_sample

# maps block
from utils import get_mask, nn_interpolate


class RegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, lambda_=1e-3, resolvant_gamma=0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.resolvant_gamma = resolvant_gamma

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        # compute linear operator matrix representation C1 and C2
        evecs_trans_x, evecs_trans_y = evecs_trans_x.unsqueeze(0), evecs_trans_y.unsqueeze(0)
        evals_x, evals_y = evals_x.unsqueeze(0), evals_y.unsqueeze(0)

        F_hat = torch.bmm(evecs_trans_x, feat_x)
        G_hat = torch.bmm(evecs_trans_y, feat_y)
        A, B = F_hat, G_hat

        D = get_mask(evals_x.flatten(), evals_y.flatten(), self.resolvant_gamma, feat_x.device).unsqueeze(0)

        A_t = A.transpose(1, 2)
        A_A_t = torch.bmm(A, A_t)
        B_A_t = torch.bmm(B, A_t)

        C_i = []
        for i in range(evals_x.size(1)):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + self.lambda_ * D_i), B_A_t[:, i, :].unsqueeze(1).transpose(1, 2))
            C_i.append(C.transpose(1, 2))
        C = torch.cat(C_i, dim=1)

        return C

class RegularizedCFMNet(nn.Module):
    """Compute the complex functional map matrix representation."""

    def __init__(self, lambda_=1e-3, resolvant_gamma=0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.resolvant_gamma = resolvant_gamma

    def forward(self, feat_x, feat_y, spec_grad_x, spec_grad_y, cevals_x, cevals_y):
        # compute linear operator matrix representation C1 and C2
        cty = torch.complex128
        spec_grad_x, spec_grad_y = spec_grad_x.unsqueeze(0), spec_grad_y.unsqueeze(0)

        F_hat = torch.bmm(spec_grad_x, feat_x.type(cty))
        G_hat = torch.bmm(spec_grad_y, feat_y.type(cty))
        A, B = F_hat, G_hat

        # if normalize input vector fields
        # A, B = A/torch.abs(A), B/torch.abs(B)

        if self.lambda_ == 0:
            Q = (B @ torch.pinverse(A))
            return Q

        # else
        cevals_x, cevals_y = cevals_x.unsqueeze(0), cevals_y.unsqueeze(0)
        D = get_mask(cevals_x.flatten(), cevals_y.flatten(), self.resolvant_gamma, feat_x.device).unsqueeze(0)

        A_t = torch.conj(A.transpose(1, 2))
        A_A_t = torch.bmm(A, A_t)
        B_A_t = torch.bmm(B, A_t)

        Q_i = []
        for i in range(cevals_x.size(1)):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(cevals_x.size(0))], dim=0)
            Q = torch.bmm(torch.inverse(A_A_t + self.lambda_ * D_i),
                          torch.conj(B_A_t[:, i, :].unsqueeze(1).transpose(1, 2)))
            Q_i.append(torch.conj(Q.transpose(1, 2)))
        Q = torch.cat(Q_i, dim=1)

        return Q


class DQFMNet(nn.Module):
    """
    Compilation of the global model :
    - diffusion net as feature extractor
    - fmap + q-fmap
    - unsupervised loss
    """

    def __init__(self, cfg):
        super().__init__()

        # feature extractor #
        with_grad=True

        self.feature_extractor = DiffusionNet(
             C_in=cfg["fmap"]["C_in"],
             C_out=cfg["fmap"]["n_feat"],
             C_width=128,
             N_block=4,
             dropout=True,
             with_gradient_features=with_grad,
             with_gradient_rotations=with_grad,
        )

        # regularized fmap
        self.fmreg_net = RegularizedFMNet(lambda_=cfg["fmap"]["lambda_"],
                                          resolvant_gamma=cfg["fmap"]["resolvant_gamma"])
        self.cfmreg_net = RegularizedCFMNet(lambda_=cfg["fmap"]["lambda_"],
                                            resolvant_gamma=cfg["fmap"]["resolvant_gamma"])

        # parameters
        self.n_fmap = cfg["fmap"]["n_fmap"]
        self.n_cfmap = cfg["fmap"]["n_cfmap"]
        self.robust = cfg["fmap"]["robust"]

    def forward(self, batch):
        verts1, faces1, mass1, L1, evals1, evecs1, gradX1, gradY1 = (batch["shape1"]["xyz"], batch["shape1"]["faces"],
                                                                     batch["shape1"]["mass"], batch["shape1"]["L"],
                                                                     batch["shape1"]["evals"], batch["shape1"]["evecs"],
                                                                     batch["shape1"]["gradX"], batch["shape1"]["gradY"])
        verts2, faces2, mass2, L2, evals2, evecs2, gradX2, gradY2 = (batch["shape2"]["xyz"], batch["shape2"]["faces"],
                                                                     batch["shape2"]["mass"], batch["shape2"]["L"],
                                                                     batch["shape2"]["evals"], batch["shape2"]["evecs"],
                                                                     batch["shape2"]["gradX"], batch["shape2"]["gradY"])

        # set features to vertices
        features1, features2 = verts1, verts2
        # print(features1.shape, features2.shape)

        feat1 = self.feature_extractor(features1, mass1, L=L1, evals=evals1, evecs=evecs1,
                                       gradX=gradX1, gradY=gradY1, faces=faces1).unsqueeze(0)
        feat2 = self.feature_extractor(features2, mass2, L=L2, evals=evals2, evecs=evecs2,
                                       gradX=gradX2, gradY=gradY2, faces=faces2).unsqueeze(0)

        # predict fmap
        evecs_trans1, evecs_trans2 = evecs1.t()[:self.n_fmap] @ torch.diag(mass1), evecs2.t()[:self.n_fmap] @ torch.diag(mass2)
        evals1, evals2 = evals1[:self.n_fmap], evals2[:self.n_fmap]
        #
        C_pred = self.fmreg_net(feat1, feat2, evals1, evals2, evecs_trans1, evecs_trans2)
        #

        # if we don't have complex spectral info we just return C
        if self.n_cfmap == 0:
            return C_pred, None

        # else, also predict cfmap
        spec_grad1, spec_grad2 = batch["shape1"]["spec_grad"][:self.n_cfmap], batch["shape2"]["spec_grad"][:self.n_cfmap]
        cevals1, cevals2 = batch["shape1"]["cevals"][:self.n_fmap], batch["shape2"]["cevals"][:self.n_fmap]
        #

        cfeat1, cfeat2 = feat1, feat2  # network features
        Q_pred = self.cfmreg_net(cfeat1, cfeat2, spec_grad1, spec_grad2, cevals1, cevals2)

        return C_pred, Q_pred
