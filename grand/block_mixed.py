import torch
from torch import nn
from base_classes import ODEblock
from utils import get_rw_adj
from torchdiffeq import odeint
import torch
import torch_sparse
from base_classes import ODEFunc
from utils import MaxNFEException
from torch_geometric.utils import softmax
import numpy as np
from utils import squareplus


class MixedODEblock(ODEblock):
  def __init__(self, 
               odefunc, 
               opt, 
               data, 
               device, 
               t=torch.tensor([0, 1]), 
               gamma=0.):
    super(MixedODEblock, self).__init__(odefunc, opt, data, device, t)
    
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], 
                           self.aug_dim * opt['hidden_dim'], 
                           opt, 
                           data, 
                           device)
    # Do you need this? 
    edge_index, edge_weight = get_rw_adj(data.edge_index, 
                                         edge_weight=data.edge_attr, 
                                         norm_dim=1,
                                         fill_value=opt['self_loop_weight'],
                                         num_nodes=data.num_nodes,
                                         dtype=data.x.dtype)
    self.odefunc.edge_index = edge_index.to(device)
    self.odefunc.edge_weight = edge_weight.to(device)

    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()
    # parameter trading off between attention and the Laplacian
    self.gamma = nn.Parameter(gamma * torch.ones(1))
    self.multihead_att_layer = SpGraphTransAttentionLayer(opt['hidden_dim'], 
                                                          opt['hidden_dim'], 
                                                          opt,
                                                          device).to(device)

  def get_attention_weights(self, x):
    attention, values = self.multihead_att_layer(x, self.odefunc.edge_index)
    return attention

  def get_mixed_attention(self, x):
    gamma = torch.sigmoid(self.gamma)
    attention = self.get_attention_weights(x)
    # gamma balance between original edge weight and attention
    # does it change the edge_index too? 
    mixed_attention = attention.mean(dim=1) * (1 - gamma) + self.odefunc.edge_weight * gamma
    return mixed_attention

  def forward(self, x):
    t = self.t.type_as(x)
    self.odefunc.attention_weights = self.get_mixed_attention(x) #\bar A(X)

    z = odeint(
      self.odefunc, x, t,
      method=self.opt['method'],
      options={'step_size': self.opt['step_size']},
      atol=self.atol,
      rtol=self.rtol)[1]
    return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"



# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):
  # currently requires in_features = out_features
  def __init__(self, 
               in_features, 
               out_features, 
               opt, 
               data, 
               device):
    super(LaplacianODEFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))

  def sparse_multiply(self, x):
    # import pdb; pdb.set_trace()
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      # this does not work
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']: 
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    else:  # constant 
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    ax = self.sparse_multiply(x) 
    alpha = torch.sigmoid(self.alpha_train)

    f = alpha * (ax - x)
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0
    return f



class SpGraphTransAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, opt, device, concat=True, edge_weights=None):
    super(SpGraphTransAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = opt['leaky_relu_slope']
    self.concat = concat
    self.device = device
    self.opt = opt
    self.h = int(opt['heads'])
    self.edge_weights = edge_weights

    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % self.h == 0, "Number of heads ({}) must be a factor of the dimension size ({})".format(
      self.h, self.attention_dim)
    self.d_k = self.attention_dim // self.h

    if self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
      self.output_var_x = nn.Parameter(torch.ones(1))
      self.lengthscale_x = nn.Parameter(torch.ones(1))
      self.output_var_p = nn.Parameter(torch.ones(1))
      self.lengthscale_p = nn.Parameter(torch.ones(1))
      self.Qx = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Qx)
      self.Vx = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Vx)
      self.Kx = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Kx)

      self.Qp = nn.Linear(opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Qp)
      self.Vp = nn.Linear(opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Vp)
      self.Kp = nn.Linear(opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Kp)

    else:
      if self.opt['attention_type'] == "exp_kernel":
        self.output_var = nn.Parameter(torch.ones(1))
        self.lengthscale = nn.Parameter(torch.ones(1))

      self.Q = nn.Linear(in_features, self.attention_dim)
      self.init_weights(self.Q)

      self.V = nn.Linear(in_features, self.attention_dim)
      self.init_weights(self.V)

      self.K = nn.Linear(in_features, self.attention_dim)
      self.init_weights(self.K)

    self.activation = nn.Sigmoid()  # nn.LeakyReLU(self.alpha)

    self.Wout = nn.Linear(self.d_k, in_features)
    self.init_weights(self.Wout)

  def init_weights(self, m):
    if type(m) == nn.Linear:
      # nn.init.xavier_uniform_(m.weight, gain=1.414)
      # m.bias.data.fill_(0.01)
      nn.init.constant_(m.weight, 1e-5)

  def forward(self, x, edge):
    """
    x might be [features, augmentation, positional encoding, labels]
    """
    # if self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
    if self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
      label_index = self.opt['feat_hidden_dim'] + self.opt['pos_enc_hidden_dim']
      p = x[:, self.opt['feat_hidden_dim']: label_index]
      x = torch.cat((x[:, :self.opt['feat_hidden_dim']], x[:, label_index:]), dim=1)

      qx = self.Qx(x)
      kx = self.Kx(x)
      vx = self.Vx(x)
      # perform linear operation and split into h heads
      kx = kx.view(-1, self.h, self.d_k)
      qx = qx.view(-1, self.h, self.d_k)
      vx = vx.view(-1, self.h, self.d_k)
      # transpose to get dimensions [n_nodes, attention_dim, n_heads]
      kx = kx.transpose(1, 2)
      qx = qx.transpose(1, 2)
      vx = vx.transpose(1, 2)
      src_x = qx[edge[0, :], :, :]
      dst_x = kx[edge[1, :], :, :]

      qp = self.Qp(p)
      kp = self.Kp(p)
      vp = self.Vp(p)
      # perform linear operation and split into h heads
      kp = kp.view(-1, self.h, self.d_k)
      qp = qp.view(-1, self.h, self.d_k)
      vp = vp.view(-1, self.h, self.d_k)
      # transpose to get dimensions [n_nodes, attention_dim, n_heads]
      kp = kp.transpose(1, 2)
      qp = qp.transpose(1, 2)
      vp = vp.transpose(1, 2)
      src_p = qp[edge[0, :], :, :]
      dst_p = kp[edge[1, :], :, :]

      prods = self.output_var_x ** 2 * torch.exp(
        -torch.sum((src_x - dst_x) ** 2, dim=1) / (2 * self.lengthscale_x ** 2)) \
              * self.output_var_p ** 2 * torch.exp(
        -torch.sum((src_p - dst_p) ** 2, dim=1) / (2 * self.lengthscale_p ** 2))

      v = None

    else:
      q = self.Q(x)
      k = self.K(x)
      v = self.V(x)

      # perform linear operation and split into h heads
      k = k.view(-1, self.h, self.d_k)
      q = q.view(-1, self.h, self.d_k)
      v = v.view(-1, self.h, self.d_k)

      # transpose to get dimensions [n_nodes, attention_dim, n_heads]
      k = k.transpose(1, 2)
      q = q.transpose(1, 2)
      v = v.transpose(1, 2)
      src = q[edge[0, :], :, :]
      dst_k = k[edge[1, :], :, :]

    # prods = Q(X) \cdot K(X) -  A = (a(Xi, Xj))
    if not self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
      prods = self.output_var ** 2 * torch.exp(-(torch.sum((src - dst_k) ** 2, dim=1) / (2 * self.lengthscale ** 2)))
    elif self.opt['attention_type'] == "scaled_dot":
      prods = torch.sum(src * dst_k, dim=1) / np.sqrt(self.d_k)
    elif self.opt['attention_type'] == "cosine_sim":
      cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
      prods = cos(src, dst_k)
    elif self.opt['attention_type'] == "pearson":
      src_mu = torch.mean(src, dim=1, keepdim=True)
      dst_mu = torch.mean(dst_k, dim=1, keepdim=True)
      src = src - src_mu
      dst_k = dst_k - dst_mu
      cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
      prods = cos(src, dst_k)

    if self.opt['reweight_attention'] and self.edge_weights is not None:
      prods = prods * self.edge_weights.unsqueeze(dim=1)
    if self.opt['square_plus']:
      attention = squareplus(prods, edge[self.opt['attention_norm_idx']])
    else:
      attention = softmax(prods, edge[self.opt['attention_norm_idx']])
    return attention, (v, prods)

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


