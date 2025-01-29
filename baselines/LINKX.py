importmath

importtorch
fromtorchimportTensor
fromtorch.nnimportBatchNorm1d,Parameter

fromtorch_geometric.nnimportinits
fromtorch_geometric.nn.convimportMessagePassing
fromtorch_geometric.nn.modelsimportMLP
fromtorch_geometric.typingimportAdj,OptTensor
fromtorch_geometric.utilsimportspmm

importmath

importtorch
fromtorchimportTensor
fromtorch.nnimportBatchNorm1d,Parameter

fromtorch_geometric.nnimportinits
fromtorch_geometric.nn.convimportMessagePassing
fromtorch_geometric.nn.modelsimportMLP
fromtorch_geometric.typingimportAdj,OptTensor
fromtorch_geometric.utilsimportspmm


classSparseLinear(MessagePassing):
def__init__(self,in_channels:int,out_channels:int,bias:bool=True):
super().__init__(aggr='add')
self.in_channels=in_channels
self.out_channels=out_channels

self.weight=Parameter(torch.empty(in_channels,out_channels))
ifbias:
self.bias=Parameter(torch.empty(out_channels))
else:
self.register_parameter('bias',None)

self.reset_parameters()

defreset_parameters(self):
inits.kaiming_uniform(self.weight,fan=self.in_channels,
a=math.sqrt(5))
inits.uniform(self.in_channels,self.bias)

defforward(
self,
edge_index:Adj,
edge_weight:OptTensor=None,
)->Tensor:
#propagate_type:(weight:Tensor,edge_weight:OptTensor)
out=self.propagate(edge_index,weight=self.weight,
edge_weight=edge_weight)

ifself.biasisnotNone:
out=out+self.bias

returnout

defmessage(self,weight_j:Tensor,edge_weight:OptTensor)->Tensor:
ifedge_weightisNone:
returnweight_j
else:
returnedge_weight.view(-1,1)*weight_j

defmessage_and_aggregate(self,adj_t:Adj,weight:Tensor)->Tensor:
returnspmm(adj_t,weight,reduce=self.aggr)


classLINKX(torch.nn.Module):
r"""TheLINKXmodelfromthe`"LargeScaleLearningonNon-Homophilous
Graphs:NewBenchmarksandStrongSimpleMethods"
<https://arxiv.org/abs/2110.14446>`_paper.

..math::
\mathbf{H}_{\mathbf{A}}&=\textrm{MLP}_{\mathbf{A}}(\mathbf{A})

\mathbf{H}_{\mathbf{X}}&=\textrm{MLP}_{\mathbf{X}}(\mathbf{X})

\mathbf{Y}&=\textrm{MLP}_{f}\left(\sigma\left(\mathbf{W}
[\mathbf{H}_{\mathbf{A}},\mathbf{H}_{\mathbf{X}}]+
\mathbf{H}_{\mathbf{A}}+\mathbf{H}_{\mathbf{X}}\right)\right)

..note::

ForanexampleofusingLINKX,see`examples/linkx.py<https://
github.com/pyg-team/pytorch_geometric/blob/master/examples/linkx.py>`_.

Args:
num_nodes(int):Thenumberofnodesinthegraph.
in_channels(int):Sizeofeachinputsample,or:obj:`-1`toderive
thesizefromthefirstinput(s)totheforwardmethod.
hidden_channels(int):Sizeofeachhiddensample.
out_channels(int):Sizeofeachoutputsample.
num_layers(int):Numberoflayersof:math:`\textrm{MLP}_{f}`.
num_edge_layers(int,optional):Numberoflayersof
:math:`\textrm{MLP}_{\mathbf{A}}`.(default::obj:`1`)
num_node_layers(int,optional):Numberoflayersof
:math:`\textrm{MLP}_{\mathbf{X}}`.(default::obj:`1`)
dropout(float,optional):Dropoutprobabilityofeachhidden
embedding.(default::obj:`0.0`)
"""
def__init__(
self,
num_nodes:int,
in_channels:int,
hidden_channels:int,
out_channels:int,
num_layers:int,
num_edge_layers:int=1,
num_node_layers:int=1,
dropout:float=0.0,
):
super().__init__()

Num_nodes=num_nodes
self.in_channels=in_channels
self.out_channels=out_channels
Num_edge_layers=num_edge_layers

self.edge_lin=SparseLinear(num_nodes,hidden_channels)

ifNum_edge_layers>1:
self.edge_norm=BatchNorm1d(hidden_channels)
channels=[hidden_channels]*num_edge_layers
self.edge_mlp=MLP(channels,dropout=0.,act_first=True)
else:
self.edge_norm=None
self.edge_mlp=None

channels=[in_channels]+[hidden_channels]*num_node_layers
Node_mlp=MLP(channels,dropout=0.,act_first=True)

self.cat_lin1=torch.nn.Linear(hidden_channels,hidden_channels)
self.cat_lin2=torch.nn.Linear(hidden_channels,hidden_channels)

channels=[hidden_channels]*num_layers+[out_channels]
self.final_mlp=MLP(channels,dropout=dropout,act_first=True)

self.reset_parameters()

defreset_parameters(self):
r"""Resetsalllearnableparametersofthemodule."""
self.edge_lin.reset_parameters()
ifself.edge_normisnotNone:
self.edge_norm.reset_parameters()
ifself.edge_mlpisnotNone:
self.edge_mlp.reset_parameters()
Node_mlp.reset_parameters()
self.cat_lin1.reset_parameters()
self.cat_lin2.reset_parameters()
self.final_mlp.reset_parameters()

defforward(
self,
x:OptTensor,
edge_index:Adj,
edge_weight:OptTensor=None,
)->Tensor:
""""""#noqa:D419
out=self.edge_lin(edge_index,edge_weight)

ifself.edge_normisnotNoneandself.edge_mlpisnotNone:
out=out.relu_()
out=self.edge_norm(out)
out=self.edge_mlp(out)

out=out+self.cat_lin1(out)

ifxisnotNone:
x=Node_mlp(x)
out=out+x
out=out+self.cat_lin2(x)

returnself.final_mlp(out.relu_())

def__repr__(self)->str:
return(f'{self.__class__.__name__}(num_nodes={Num_nodes},'
f'in_channels={self.in_channels},'
f'out_channels={self.out_channels})')