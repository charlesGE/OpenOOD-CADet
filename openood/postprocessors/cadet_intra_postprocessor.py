from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import os

from .base_postprocessor import BasePostprocessor


class CadetInPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.postprocessor_args = config.postprocessor.postprocessor_args
        self.n_trs = self.config.preprocessor.n_transforms

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        flat_data = data.flatten(0,1)
        logits, features = net(flat_data, return_feature=True)
        m_in_feats = self.get_intra_sim(features)
        m_in_logits = self.get_intra_sim(logits)
        return m_in_feats, m_in_logits

    @torch.no_grad()
    def extract_stats(self, net, save_pth, ood_data_loader):
        net.eval()

        m_in_feats = []
        m_in_logits = []
        for batch in ood_data_loader:
            flat_batch = batch['data'].cuda()
            scores_feats, scores_logits = self.postprocess(net, flat_batch)
            m_in_feats.extend(scores_feats)
            m_in_logits.extend(scores_logits)
        m_in_feats = np.array(m_in_feats)
        m_in_logits = np.array(m_in_logits)

        feats_file_pth = os.path.join(save_pth, 'm_in.npy')
        np.save(feats_file_pth, m_in_feats)
        logits_file_pth = os.path.join(save_pth, 'm_in_logits.npy')
        np.save(logits_file_pth, m_in_logits)

        if self.config.network.name == 'vit':
            m_in_patch = []
            for batch in ood_data_loader:
                flat_batch = batch['data'].cuda().flatten(0,1)
                patch_tokens = net(flat_batch, 'return_patch_token'==True)
                scores_patch = self.get_intra_sim(patch_tokens)
                m_in_patch.extend(scores_patch)
            m_in_patch = np.array(m_in_patch)
                
            patch_file_pth = os.path.join(save_pth, 'm_in_patch.npy')
            np.save(patch_file_pth, m_in_patch)
        
    def get_intra_sim(self, feats):
        res = []
        feature_dim = feats.size()[1]
        feats = feats.view(-1, self.n_trs, feature_dim)
        for embs in feats:
            sim_matrix = embs @ embs.T
            score = (torch.sum(sim_matrix) - torch.sum(sim_matrix.diag())) / (self.n_trs * (self.n_trs-1))
            res.append(score.cpu().item())
        return res