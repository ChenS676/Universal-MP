import torch
import torch.nn.functional as F
from base_classes import BaseGNN
from block_mixed import MixedODEblock
from block_mixed import LaplacianODEFunc

# Define the GNN model.
class GNN(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNN, self).__init__(opt, dataset, device)
    self.f = LaplacianODEFunc
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = MixedODEblock(self.f, opt, dataset.data, device, t=time_tensor).to(device)

  def forward(self, x, pos_encoding=None):
    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    x = self.m1(x) # linear layer
    if self.opt['batch_norm']:
      x = self.bn_in(x)
    
    self.odeblock.set_x0(x)
    z = self.odeblock(x) # Is there any parameters and how does they change during training and evaluation time.
    z = F.relu(z)
    z = F.dropout(z, self.opt['dropout'], training=self.training)
    z = self.m2(z)
    return z


