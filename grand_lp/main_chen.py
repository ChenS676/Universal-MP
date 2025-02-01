import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
torch.cuda.empty_cache()
import itertools
from tqdm import tqdm
from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.cmd_args import parse_args
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from torch_geometric.utils import negative_sampling
import argparse
import wandb
from yacs.config import CfgNode as CN

from graphgps.train.heart_train import Trainer_Heart
from graphgps.config import (dump_cfg, dump_run_cfg)
from graphgps.utility.utils import set_cfg, parse_args, get_git_repo_root_path, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger, LinearDecayLR
from graphgps.score.custom_score import mlp_score, InnerProduct
from graphgps.network.heart_gnn import GAT_Variant, GAE_forall, GCN_Variant, \
                                SAGE_Variant, GIN_Variant, DGCNN
from baselines.ncnc.ogbdataset import loaddataset
from baselines.data_utils.check_dataset import check_dimension, check_data_leakage


FILE_PATH = f'{get_git_repo_root_path()}/'



def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='',
                        help='The configuration file path for param tune.')
    parser.add_argument('--data', dest='data', type=str, required=True,
                        default='pubmed',
                        help='data name')
    parser.add_argument('--batch_size', dest='bs', type=int, required=False,
                        default=2**15,
                        help='data name')
    parser.add_argument('--device', dest='device', required=True, 
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=True,
                        default=100,
                        help='data name')
    parser.add_argument('--model', dest='model', type=str, required=True,
                        default='GCN_Variant',
                        help='model name')
    parser.add_argument('--score', dest='score', type=str, required=False, default='mlp_score',
                        help='decoder name')
    parser.add_argument('--wandb', dest='wandb', required=False, 
                        help='data name')
    parser.add_argument('--repeat', type=int, default=5,
                        help='The number of repeated jobs.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    return parser.parse_args()

yaml_file = {   
             'GAT_Variant': 'yamls/cora/gcns/all_gcn_baselines.yaml',
             'GAE_Variant': 'yamls/cora/gcns/all_gcn_baselines.yaml',
             'GIN_Variant': 'yamls/cora/gcns/all_gcn_baselines.yaml',
             'GCN_Variant': 'yamls/cora/gcns/all_gcn_baselines.yaml',
             'SAGE_Variant': 'yamls/cora/gcns/all_gcn_baselines.yaml',
             'DGCNN': 'yamls/cora/gcns/heart_gnn_models.yaml'
            }