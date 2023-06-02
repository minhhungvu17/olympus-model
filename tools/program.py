from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import sys
import platform
import yaml
import time
import datetime
import torch 
from tqdm import tqdm
import cv2
import numpy as np 
from argparse import ArgumentParser, RawDescriptionHelpFormatter


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class= RawDescriptionHelpFormatter
        )
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")
        self.add_argument(
            '-p',
            '--profiler_options',
            type=str,
            default=None,
            help='The option of profiler, which should be in format ' \
                 '\"key1=value1;key2=value2;key3=value3\".'
        )
    
    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args
    

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config

def load_config(file_path):
    """_summary_
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config

def merge_config(config, opts):
    """
    Merge config into gloabal config

    Args:
        config (dict): Config to be merged
        opts (dict): config

    Returns: Global config
    """
    for key, value in opts.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split('.')
            assert(
                sub_keys[0] in config
            ), "the sub_keys can only be one of global_config: {}, but get: " \
                "{}, please check your running command".format(
                    config.keys(), sub_keys[0]
                )
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) -2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return config

def preprocess(is_train=False):
    FLAGS = ArgsParser().parse_args()
    profiler_options = FLAGS.profiler_options
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    profile_dic = {"profiler_options": profiler_options}
    config = merge_config(config, profile_dic)
    return config