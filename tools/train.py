from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import sys
import logging

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import yaml
import torch 

from pytorchocr.modeling.architectures import build_model
from pytorchocr.data import build_dataloader

import tools.program as program

if __name__ == "__main__":
    config = program.preprocess(is_train=True)
    seed = config['Global']['seed'] if 'seed' in config['Global'] else 1024
    x = torch.rand(1,3,640,640)
    model = build_model(config['Architecture'])
    dataloader = build_dataloader(config, 'Train', logger=logging,seed=seed)
    
    o = model(x)
    print(o['maps'].shape)




