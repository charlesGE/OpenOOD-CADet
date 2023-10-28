from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import os

from .base_postprocessor import BasePostprocessor


class CadetOutPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.postprocessor_args = config.postprocessor.postprocessor_args
        self.n_trs = self.config.preprocessor.n_transforms

    # COMING SOON !