import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import time
from tqdm import tqdm

from data_utils.load_lp import *
from data_utils.graph_rewiring import apply_KNN
from metrics.metrics import *
from models.base_classes import LinkPredictor
from models.GNN_KNN import GNN_KNN
from models.GNN_KNN_early import GNNKNNEarly
from models.GNN import GRAND
from models.GNN_early import GNNEarly
from grand_lp.models.trainer import Trainer_GRAND
from torch_geometric.nn import Node2Vec
from best_params import best_params_dict
from graphgps.utility.utils import set_cfg, parse_args, get_git_repo_root_path, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger, LinearDecayLR

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


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='yamls/cora/gcn.yaml',
                        help='The configuration file path.')
    args = parser.parse_args()
    
    yaml_config = load_yaml_config(args.cfg_file)
    opt = yaml_config[next(iter(yaml_config))]
    
    cmd_opt = vars(args)
    try:
        best_opt = best_params_dict[cmd_opt['dataset']]
        opt = {**cmd_opt, **best_opt}
    except KeyError:
        opt = cmd_opt
    
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

