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
from models.base_classes import LinkPredictor
from models.GNN_KNN import GNN_KNN
from models.GNN_KNN_early import GNNKNNEarly
from models.GNN import GRAND
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
from best_params import best_params_dict
from data_utils.load_lp import get_grand_dataset, apply_beltrami
FILE_PATH = f'{get_git_repo_root_path()}/'

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def merge_cmd_args(cmd_opt, opt):
  if cmd_opt['beltrami']:
    opt['beltrami'] = True
  if cmd_opt['function'] is not None:
    opt['function'] = cmd_opt['function']
  if cmd_opt['block'] is not None:
    opt['block'] = cmd_opt['block']
  if cmd_opt['attention_type'] != 'scaled_dot':
    opt['attention_type'] = cmd_opt['attention_type']
  if cmd_opt['self_loop_weight'] is not None:
    opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  if cmd_opt['method'] is not None:
    opt['method'] = cmd_opt['method']
  if cmd_opt['step_size'] != 1:
    opt['step_size'] = cmd_opt['step_size']
  if cmd_opt['time'] != 1:
    opt['time'] = cmd_opt['time']
  if cmd_opt['epoch'] != 100:
    opt['epoch'] = cmd_opt['epoch']
  if not cmd_opt['not_lcc']:
    opt['not_lcc'] = False
  if cmd_opt['num_splits'] != 1:
    opt['num_splits'] = cmd_opt['num_splits']
  
  return opt

def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))


def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)


import yaml
def load_yaml_config(file_path):
    """Loads a YAML configuration file."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='',
                        help='The configuration file path for param tune.')
    parser.add_argument('--data', dest='data', type=str, required=True,
                        default='pubmed',
                        help='data name')
    parser.add_argument('--device', dest='device', required=True, 
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=True,
                        default=100,
                        help='data name')
    parser.add_argument('--score', dest='score', type=str, required=False, default='mlp_score',
                        help='decoder name')
    parser.add_argument('--wandb', dest='wandb', required=False, 
                        help='data name')
    parser.add_argument('--repeat', type=int, default=5,
                        help='The number of repeated jobs.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    parser.add_argument('--cfg_file', dest='cfg_file', type=str, required=False,
                        default= 'yamls/cora/grand/early_beltrami.yaml',
                        help='The configuration file path.')
    parser.add_argument('--batch_size', type=int, default=2**12)
    parser.add_argument('--dataset', type=str, default='ogbl-ppa')
    parser.add_argument('--dataset_dir', type=str, default='./data')
    
    return parser.parse_args()



if __name__=='__main__':
    
    # process params
    args = parse_args()
    yaml_opt = load_yaml_config(FILE_PATH + args.cfg_file)
    
    cmd_opt = vars(args)
    opt = yaml_opt | cmd_opt  
    
    try:
        best_opt = best_params_dict[cmd_opt['dataset']]
        opt = {**opt, **best_opt}
    except KeyError:
        pass
    
    cfg = set_cfg(FILE_PATH, args.cfg_file)

    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    data, splits = get_grand_dataset(opt['dataset_dir'], opt, opt['dataset'], opt['use_valedges_as_input'])

    if args.dataset == "ogbl-citation2":
        opt['metric'] = "MRR"
    if data.x is None:
        opt['use_feature'] = False

    if opt['beltrami']:
      pos_encoding = apply_beltrami(data.to('cpu'), opt).to(device)
      opt['pos_enc_dim'] = pos_encoding.shape[1]
    else:
      pos_encoding = None

    data = data.to(device)
    predictor = LinkPredictor(opt['hidden_dim'], opt['hidden_dim'], 1, opt['mlp_num_layers'], opt['dropout']).to(device)
    batch_size = opt['batch_size']  

    if opt['rewire_KNN'] or opt['fa_layer']:
      model = GNN_KNN(opt, data, splits, predictor, batch_size, device).to(device) if opt["no_early"] else GNNKNNEarly(opt, data, splits, predictor, batch_size, device).to(device)
    else:
      print(opt["no_early"])
      model = GRAND(opt, data, splits, predictor, batch_size, device).to(device) if opt["no_early"] else GNNEarly(opt, data, splits, predictor, batch_size, device).to(device)

    
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])

    trainer = Trainer_GRAND(
        opt=opt,
        model=model,
        predictor=predictor,
        optimizer=optimizer,
        data=data,
        pos_encoding=pos_encoding,
        splits=splits,
        batch_size=batch_size,
        device=device,
        log_dir='./logs'
    )

    best_results = trainer.train()

    trainer.finalize()