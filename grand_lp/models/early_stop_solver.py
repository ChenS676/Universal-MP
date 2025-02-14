import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torchdiffeq
from torchdiffeq._impl.dopri5 import _DORMAND_PRINCE_SHAMPINE_TABLEAU, DPS_C_MID
from torchdiffeq._impl.solvers import FixedGridODESolver
import torch
from torchdiffeq._impl.misc import _check_inputs, _flat_to_shape
import torch.nn.functional as F
import copy

from torchdiffeq._impl.interp import _interp_evaluate
from torchdiffeq._impl.rk_common import RKAdaptiveStepsizeODESolver, rk4_alt_step_func
from ogb.linkproppred import Evaluator
from metrics.metrics import *
from torch.utils.data import DataLoader

class EarlyStopDopri5(RKAdaptiveStepsizeODESolver):
  order = 5
  tableau = _DORMAND_PRINCE_SHAMPINE_TABLEAU
  mid = DPS_C_MID

  def __init__(self, func, y0, rtol, atol, opt, splits, predictor, batch_size, **kwargs):
    super(EarlyStopDopri5, self).__init__(func, y0, rtol, atol, **kwargs)

    self.data = None
    self.splits = splits 
    self.predictor = predictor 
    self.batch_size = batch_size
    self.best_val = 0
    self.best_test = 0
    self.max_test_steps = opt['max_test_steps']
    self.best_time = 0
    self.ode_test = self.test_OGB #if opt['dataset'] in ['ogbn-arxiv', 'ogbl-collab'] else self.test
    self.opt = opt
    self.dataset = opt['dataset']
    if opt['dataset'].startswith('ogbl-'):
      self.evaluator = Evaluator(name=opt['dataset'])
    else:
      self.evaluator = Evaluator(name='ogbl-collab')

  def set_accs(self, train, val, test, time):
    self.best_train = train
    self.best_val = val
    self.best_test = test
    self.best_time = time.item()

  def integrate(self, t):
    solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
    solution[0] = self.y0
    t = t.to(self.dtype)
    self._before_integrate(t)
    new_t = t
    for i in range(1, len(t)):
      new_t, y = self.advance(t[i])
      solution[i] = y
    return new_t, solution

  def advance(self, next_t):
    """
    Takes steps dt to get to the next user specified time point next_t. In practice this goes past next_t and then interpolates
    :param next_t:
    :return: The state, x(next_t)
    """
    n_steps = 0
    
    while next_t > self.rk_state.t1 and n_steps < self.max_test_steps:
      self.rk_state = self._adaptive_step(self.rk_state)
      n_steps += 1
      train_hits100, val_hits100, test_hits100 = self.evaluate(self.rk_state)
      
      if val_hits100 > self.best_test:
        self.set_accs(train_hits100, val_hits100, test_hits100, self.rk_state.t1)

    new_t = next_t
    if n_steps < self.max_test_steps:
      return (new_t, _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t))
    else:
      return (new_t, _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, self.rk_state.t1))

  @torch.no_grad()
  def test_OGB(self, h):
    # data = self.data
    batch_size = self.batch_size
    # splits = self.splits
    predictor = self.predictor
    predictor.eval()
    
    # pos_train_edge = splits['train']['pos_edge_label_index'].to(data.x.device)
    # neg_train_edge = splits['train']['neg_edge_label_index'].to(data.x.device)
    # pos_valid_edge = splits['valid']['pos_edge_label_index'].to(data.x.device)
    # neg_valid_edge = splits['valid']['neg_edge_label_index'].to(data.x.device)
    # pos_test_edge = splits['test']['pos_edge_label_index'].to(data.x.device)
    # neg_test_edge = splits['test']['neg_edge_label_index'].to(data.x.device)
    
    pos_train_edge = self.splits['train']['edge'].to(self.data.x.device)
    pos_valid_edge = self.splits['valid']['edge'].to(self.data.x.device)
    neg_valid_edge = self.splits['valid']['edge_neg'].to(self.data.x.device)
    pos_test_edge = self.splits['test']['edge'].to(self.data.x.device)
    neg_test_edge = self.splits['test']['edge_neg'].to(self.data.x.device)
    
    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(1)), batch_size):
        # edge = pos_train_edge[:, perm]
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    # neg_train_preds = []
    # for perm in DataLoader(range(neg_train_edge.size(1)), batch_size):
    #     edge = neg_train_edge[:, perm]
    #     neg_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    # neg_train_pred = torch.cat(neg_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(1)), batch_size):
        # edge = pos_valid_edge[:, perm]
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(1)), batch_size):
        # edge = neg_valid_edge[:, perm]
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(1)), batch_size):
        # edge = pos_test_edge[:, perm]
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    print(f"Positive predictions (min, max): {pos_test_pred.min().item()}, {pos_test_pred.max().item()}")
    print(f"Positive predictions (mean, std): {pos_test_pred.mean().item()}, {pos_test_pred.std().item()}")
    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(1)), batch_size):
        # edge = neg_test_edge[:, perm]
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    print(f"Negative predictions (min, max): {neg_test_pred.min().item()}, {neg_test_pred.max().item()}")
    print(f"Negative predictions (mean, std): {neg_test_pred.mean().item()}, {neg_test_pred.std().item()}")
    
    results = {}
    for K in [1, 3, 10, 20, 50, 100]:
        self.evaluator.K = K
        train_hits = self.evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred, # neg_train_pred,
        })[f'hits@{K}']
        valid_hits = self.evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = self.evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
    
    # print(f"Shape of pos_val_pred: {pos_test_pred.shape}")
    # print(f"Shape of neg_val_pred: {neg_test_pred.shape}")

    result_mrr_test = evaluate_mrr(pos_test_pred, neg_test_pred, self.opt)  
    
    for name in ['MRR', 'mrr_hit1', 'mrr_hit3', 'mrr_hit10', 'mrr_hit20', 'mrr_hit50', 'mrr_hit100']:
        results[name] = (result_mrr_test[name])
        # writer.add_scalar(f'Accuracy/Test_{name}', result_mrr_test[name], epoch)
    
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])
    
    result_auc_test = evaluate_auc(test_pred, test_true)
    for name in ['AUC', 'AP']:
        results[name] = (result_auc_test[name])
        # writer.add_scalar(f'Accuracy/Test_{name}',result_auc_test[name], epoch)

    result_acc_test = acc(pos_test_pred, neg_test_pred)
    results['ACC'] = (result_acc_test)
    print(results)
    return results[f'Hits@{100}']

  @torch.no_grad()
  def evaluate(self, rkstate):
    z = rkstate.y1
    train_acc, val_acc, test_acc = self.ode_test(z)
    # log = 'ODE eval t0 {:.3f}, t1 {:.3f} Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(t0, t1, loss, train_acc, val_acc, tmp_test_acc))
    return train_acc, val_acc, test_acc

  def set_data(self, data):
    if self.data is None:
      self.data = data

class EarlyStopRK4(FixedGridODESolver):
  order = 4

  def __init__(self, func, y0, opt, splits, predictor, batch_size, eps=0, **kwargs):
    super(EarlyStopRK4, self).__init__(func, y0, **kwargs)
    self.eps = torch.as_tensor(eps, dtype=self.dtype, device=self.device)
    self.data = None
    self.splits = splits 
    self.predictor = predictor 
    self.batch_size = batch_size
    self.best_val = 0
    self.best_test = 0
    self.best_time = 0
    self.ode_test = self.test_OGB #if opt['dataset'] in ['ogbn-arxiv', 'ogbl-collab'] else self.test
    self.dataset = opt['dataset']
    if opt['dataset'].startswith('ogbl-'):
      self.evaluator = Evaluator(name=opt['dataset'])
    else:
      self.evaluator = Evaluator(name='ogbl-collab')

  def _step_func(self, func, t, dt, t1, y):
    ver = torchdiffeq.__version__[0] + torchdiffeq.__version__[2] + torchdiffeq.__version__[4]
    if int(ver) >= 22:  # '0.2.2'
      return rk4_alt_step_func(func, t + self.eps, dt - 2 * self.eps, t1, y)
    else:
      return rk4_alt_step_func(func, t + self.eps, dt - 2 * self.eps, y)

  def set_accs(self, train, val, test, time):
    self.best_train = train
    self.best_val = val
    self.best_test = test
    self.best_time = time.item()

  def integrate(self, t):
    time_grid = self.grid_constructor(self.func, self.y0, t)
    assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

    solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
    solution[0] = self.y0

    j = 1
    y0 = self.y0
    for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
      dy = self._step_func(self.func, t0, t1 - t0, t1, y0)
      y1 = y0 + dy
      train_hits100, val_hits100, test_hits100 = self.evaluate(y1, t0, t1)
      if test_hits100 > self.best_val:
        self.set_accs(train_hits100, val_hits100, test_hits100, t1)

      while j < len(t) and t1 >= t[j]:
        solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
        j += 1
      y0 = y1

    return t1, solution

  @torch.no_grad()
  def test_OGB(self, h):
    # data = self.data
    batch_size = self.batch_size
    # splits = self.splits
    predictor = self.predictor
    predictor.eval()
    
    # pos_train_edge = splits['train']['pos_edge_label_index'].to(data.x.device)
    # neg_train_edge = splits['train']['neg_edge_label_index'].to(data.x.device)
    # pos_valid_edge = splits['valid']['pos_edge_label_index'].to(data.x.device)
    # neg_valid_edge = splits['valid']['neg_edge_label_index'].to(data.x.device)
    # pos_test_edge = splits['test']['pos_edge_label_index'].to(data.x.device)
    # neg_test_edge = splits['test']['neg_edge_label_index'].to(data.x.device)
    
    pos_train_edge = self.splits['train']['edge'].to(self.data.x.device)
    pos_valid_edge = self.splits['valid']['edge'].to(self.data.x.device)
    neg_valid_edge = self.splits['valid']['edge_neg'].to(self.data.x.device)
    pos_test_edge = self.splits['test']['edge'].to(self.data.x.device)
    neg_test_edge = self.splits['test']['edge_neg'].to(self.data.x.device)
    
    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(1)), batch_size):
        # edge = pos_train_edge[:, perm]
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    # neg_train_preds = []
    # for perm in DataLoader(range(neg_train_edge.size(1)), batch_size):
    #     edge = neg_train_edge[:, perm]
    #     neg_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    # neg_train_pred = torch.cat(neg_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(1)), batch_size):
        # edge = pos_valid_edge[:, perm]
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(1)), batch_size):
        # edge = neg_valid_edge[:, perm]
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(1)), batch_size):
        # edge = pos_test_edge[:, perm]
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    print(f"Positive predictions (min, max): {pos_test_pred.min().item()}, {pos_test_pred.max().item()}")
    print(f"Positive predictions (mean, std): {pos_test_pred.mean().item()}, {pos_test_pred.std().item()}")
    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(1)), batch_size):
        # edge = neg_test_edge[:, perm]
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    print(f"Negative predictions (min, max): {neg_test_pred.min().item()}, {neg_test_pred.max().item()}")
    print(f"Negative predictions (mean, std): {neg_test_pred.mean().item()}, {neg_test_pred.std().item()}")
    
    results = {}
    for K in [1, 3, 10, 20, 50, 100]:
        self.evaluator.K = K
        train_hits = self.evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred, # neg_train_pred,
        })[f'hits@{K}']
        valid_hits = self.evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = self.evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
    
    # print(f"Shape of pos_val_pred: {pos_test_pred.shape}")
    # print(f"Shape of neg_val_pred: {neg_test_pred.shape}")

    result_mrr_test = evaluate_mrr(pos_test_pred, neg_test_pred)  
    
    for name in ['MRR', 'mrr_hit1', 'mrr_hit3', 'mrr_hit10', 'mrr_hit20', 'mrr_hit50', 'mrr_hit100']:
        results[name] = (result_mrr_test[name])
        # writer.add_scalar(f'Accuracy/Test_{name}', result_mrr_test[name], epoch)
    
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])
    
    result_auc_test = evaluate_auc(test_pred, test_true)
    for name in ['AUC', 'AP']:
        results[name] = (result_auc_test[name])
        # writer.add_scalar(f'Accuracy/Test_{name}',result_auc_test[name], epoch)

    result_acc_test = acc(pos_test_pred, neg_test_pred)
    results['ACC'] = (result_acc_test)
    print(results)
    return results[f'Hits@{100}']

  @torch.no_grad()
  def evaluate(self, z, t0, t1):
    train_acc, val_acc, test_acc = self.ode_test(z)
    # log = 'ODE eval t0 {:.3f}, t1 {:.3f} Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(t0, t1, loss, train_acc, val_acc, tmp_test_acc))
    return train_acc, val_acc, test_acc

  def set_data(self, data):
    if self.data is None:
      self.data = data


SOLVERS = {
  'dopri5': EarlyStopDopri5,
  'rk4': EarlyStopRK4
}


class EarlyStopInt(torch.nn.Module):
  def __init__(self, t, opt, device=None, splits=None, predictor=None, batch_size=None):
    super(EarlyStopInt, self).__init__()
    self.device = device
    self.splits = splits
    self.predictor = predictor
    self.batch_size = batch_size
    self.solver = None
    self.data = None
    self.max_test_steps = opt['max_test_steps']
    self.opt = opt
    self.t = torch.tensor([0, opt['earlystopxT'] * t], dtype=torch.float).to(self.device)

  def __call__(self, func, y0, t, method=None, rtol=1e-7, atol=1e-9,
               adjoint_method="dopri5", adjoint_atol=1e-9, adjoint_rtol=1e-7, options=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a Tensor holding the state `y` and a scalar Tensor
            `t` into a Tensor of state derivatives with respect to time.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. May
            have any floating point or complex dtype.
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`. The initial time point should be the first element of this sequence,
            and each time must be larger than the previous time. May have any floating
            point dtype. Converted to a Tensor with float64 dtype.
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.
        name: Optional name for this operation.

    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.

    Raises:
        ValueError: if an invalid `method` is provided.
        TypeError: if `options` is supplied without `method`, or if `t` or `y0` has
            an invalid dtype.
    """
    method = self.opt['method']
    assert method in ['rk4', 'dopri5'], "Only dopri5 and rk4 implemented with early stopping"

    ver = torchdiffeq.__version__
    if int(ver[0] + ver[2] + ver[4]) >= 20:  # 0.2.0 change of signature on this release for event_fn
      event_fn = None
      shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed = _check_inputs(func, y0, self.t, rtol,
                                                                                                atol, method, options,
                                                                                                event_fn, SOLVERS)
    else:
      shapes, func, y0, t, rtol, atol, method, options = _check_inputs(func, y0, self.t, rtol, atol, method, options,
                                                                     SOLVERS)

    self.solver = SOLVERS[method](func, y0, rtol=rtol, atol=atol, opt=self.opt, splits=self.splits, predictor=self.predictor, batch_size=self.batch_size, **options)
    if self.solver.data is None:
      self.solver.data = self.data
    t, solution = self.solver.integrate(t)
    if shapes is not None:
      solution = _flat_to_shape(solution, (len(t),), shapes)
    return solution
