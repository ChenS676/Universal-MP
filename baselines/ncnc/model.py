from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from utils import adjoverlap
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add
from torch_sparse import SparseTensor
import torch_sparse
from torch_scatter import scatter_add
from typing import Iterable, Final

# a vanilla message passing layer 
class PureConv(nn.Module):
    aggr: Final[str]
    def __init__(self, indim, outdim, aggr="gcn") -> None:
        super().__init__()
        self.aggr = aggr
        if indim == outdim:
            self.lin = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x, adj_t):
        x = self.lin(x)
        if self.aggr == "mean":
            return spmm_mean(adj_t, x)
        elif self.aggr == "max":
            return spmm_max(adj_t, x)[0]
        elif self.aggr == "sum":
            return spmm_add(adj_t, x)
        elif self.aggr == "gcn":
            norm = torch.rsqrt_((1+adj_t.sum(dim=-1))).reshape(-1, 1)
            x = norm * x
            x = spmm_add(adj_t, x) + x
            x = norm * x
            return x
    

convdict = {
    "gcn":
    GCNConv,
    "gcn_cached":
    lambda indim, outdim: GCNConv(indim, outdim, cached=True),
    "sage":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="mean", normalize=False, add_self_loops=False),
    "gin":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="sum", normalize=False, add_self_loops=False),
    "max":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="max", normalize=False, add_self_loops=False),
    "puremax": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="max"),
    "puresum": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="sum"),
    "puremean": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="mean"),
    "puregcn": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="gcn"),
    "none":
    None
}

predictor_dict = {}

# Edge dropout
class DropEdge(nn.Module):

    def __init__(self, dp: float = 0.0) -> None:
        super().__init__()
        self.dp = dp

    def forward(self, edge_index: Tensor):
        if self.dp == 0:
            return edge_index
        mask = torch.rand_like(edge_index[0], dtype=torch.float) > self.dp
        return edge_index[:, mask]

# Edge dropout with adjacency matrix as input
class DropAdj(nn.Module):
    doscale: Final[bool] # whether to rescale edge weight
    def __init__(self, dp: float = 0.0, doscale=True) -> None:
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1/(1-dp)))
        self.doscale = doscale

    def forward(self, adj: SparseTensor)->SparseTensor:
        if self.dp < 1e-6 or not self.training:
            return adj
        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = torch_sparse.masked_select_nnz(adj, mask, layout="coo")
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value()*self.ratio, layout="coo")
            else:
                adj.fill_value_(1/(1-self.dp), dtype=torch.float)
        return adj


# Vanilla MPNN composed of several layers.
class GCN(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 ln=False,
                 res=False,
                 max_x=-1,
                 conv_fn="gcn",
                 jk=False,
                 edrop=0.0,
                 xdropout=0.0,
                 taildropout=0.0,
                 noinputlin=False):
        super().__init__()
        
        self.adjdrop = DropAdj(edrop)
        
        if max_x >= 0:
            tmp = nn.Embedding(max_x + 1, hidden_channels)
            nn.init.orthogonal_(tmp.weight)
            self.xemb = nn.Sequential(tmp, nn.Dropout(dropout))
            in_channels = hidden_channels
        else:
            self.xemb = nn.Sequential(nn.Dropout(xdropout)) #nn.Identity()
            if not noinputlin and ("pure" in conv_fn or num_layers==0):
                self.xemb.append(nn.Linear(in_channels, hidden_channels))
                self.xemb.append(nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity())
        
        self.res = res
        self.jk = jk
        if jk:
            self.register_parameter("jkparams", nn.Parameter(torch.randn((num_layers,))))
            
        if num_layers == 0 or conv_fn =="none":
            self.jk = False
            return
        
        convfn = convdict[conv_fn]
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        if num_layers == 1:
            hidden_channels = out_channels

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        if "pure" in conv_fn:
            self.convs.append(convfn(hidden_channels, hidden_channels))
            for i in range(num_layers-1):
                self.lins.append(nn.Identity())
                self.convs.append(convfn(hidden_channels, hidden_channels))
            self.lins.append(nn.Dropout(taildropout, True))
        else:
            self.convs.append(convfn(in_channels, hidden_channels))
            self.lins.append(
                nn.Sequential(lnfn(hidden_channels, ln), nn.Dropout(dropout, True),
                            nn.ReLU(True)))
            for i in range(num_layers - 1):
                self.convs.append(
                    convfn(
                        hidden_channels,
                        hidden_channels if i == num_layers - 2 else out_channels))
                if i < num_layers - 2:
                    self.lins.append(
                        nn.Sequential(
                            lnfn(
                                hidden_channels if i == num_layers -
                                2 else out_channels, ln),
                            nn.Dropout(dropout, True), nn.ReLU(True)))
                else:
                    self.lins.append(nn.Identity())
        

    def forward(self, x, adj_t):
        x = self.xemb(x)
        jkx = []
        for i, conv in enumerate(self.convs):
            x1 = self.lins[i](conv(x, self.adjdrop(adj_t)))
            if self.res and x1.shape[-1] == x.shape[-1]: # residual connection
                x = x1 + x
            else:
                x = x1
            if self.jk:
                jkx.append(x)
        if self.jk: # JumpingKnowledge Connection
            jkx = torch.stack(jkx, dim=0)
            sftmax = self.jkparams.reshape(-1, 1, 1)
            x = torch.sum(jkx*sftmax, dim=0)
        return x


# GAE predictor
class LinkPredictor(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 **kwargs):
        super(LinkPredictor, self).__init__()

        self.lins = nn.Sequential()

        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        if num_layers == 1:
            self.lins = nn.Linear(in_channels, out_channels)
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.lins.append(lnfn(hidden_channels, ln))
            self.lins.append(nn.Dropout(dropout, inplace=True))
            self.lins.append(nn.ReLU(inplace=True))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.lins.append(lnfn(hidden_channels, ln))
                self.lins.append(nn.Dropout(dropout, inplace=True))
                self.lins.append(nn.ReLU(inplace=True))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = [0.25]):
        x = x[tar_ei].prod(dim=0)
        x = self.lins(x)
        return x.expand(-1, len(cndropprobs) + 1)

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])


# GAE + CN link predictor
class SCNLinkPredictor(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        xcn = cn.sum(dim=-1).float().reshape(-1, 1)
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij)],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])

# another GAE + CN predictor
class CatSCNLinkPredictor(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels+1, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        xcn = cn.sum(dim=-1).float().reshape(-1, 1)
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(torch.cat((xcn, xij), dim=-1) )],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])

# GAE + CN predictor boosted by CNC trick
class IncompleteSCN1Predictor(SCNLinkPredictor):
    learnablept: Final[bool]
    depth: Final[int]
    splitsize: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0,
                 alpha=1.0,
                 scale=5,
                 offset=3,
                 trainresdeg=8,
                 testresdeg=128,
                 pt=0.5,
                 learnablept=False,
                 depth=1,
                 splitsize=-1,
                 ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, edrop, ln, cndeg, use_xlin, tailact, twolayerlin, beta)
        self.learnablept= learnablept
        self.depth = depth
        self.splitsize = splitsize
        self.lins = nn.Sequential()
        self.register_buffer("alpha", torch.tensor([alpha]))
        self.register_buffer("pt", torch.tensor([pt]))
        self.register_buffer("scale", torch.tensor([scale]))
        self.register_buffer("offset", torch.tensor([offset]))

        self.trainresdeg = trainresdeg
        self.testresdeg = testresdeg
        self.ptlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True), nn.Linear(hidden_channels, 1), nn.Sigmoid())
        # print(self.xcnlin)

    def clampprob(self, prob, pt):
        p0 = torch.sigmoid_(self.scale*(prob-self.offset))
        return self.alpha*pt*p0/(pt*p0+1-p0)

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = [],
                           depth: int=None):
        assert len(cndropprobs) == 0
        if depth is None:
            depth = self.depth
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        xij = xi*xj
        if depth > 0.5:
            cn, cnres1, cnres2 = adjoverlap(
                    adj,
                    adj,
                    tar_ei,
                    filled1,
                    calresadj=True,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)
        else:
            cn = adjoverlap(
                    adj,
                    adj,
                    tar_ei,
                    filled1,
                    calresadj=False,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)
        xcns = [cn.sum(dim=-1).float().reshape(-1, 1)]
        
        if depth > 0.5:
            potcn1 = cnres1.coo()
            potcn2 = cnres2.coo()
            with torch.no_grad():
                if self.splitsize < 0:
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = self.forward(
                        x, adj, ei1,
                        filled1, depth-1).flatten()
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = self.forward(
                        x, adj, ei2,
                        filled1, depth-1).flatten()
                else:
                    num1 = potcn1[1].shape[0]
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = torch.empty_like(potcn1[1], dtype=torch.float)
                    for i in range(0, num1, self.splitsize):
                        probcn1[i:i+self.splitsize] = self.forward(x, adj, ei1[:, i: i+self.splitsize], filled1, depth-1).flatten()
                    num2 = potcn2[1].shape[0]
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = torch.empty_like(potcn2[1], dtype=torch.float)
                    for i in range(0, num2, self.splitsize):
                        probcn2[i:i+self.splitsize] = self.forward(x, adj, ei2[:, i: i+self.splitsize],filled1, depth-1).flatten()
            if self.learnablept:
                pt = self.ptlin(xij)
                probcn1 = self.clampprob(probcn1, pt[potcn1[0]]) 
                probcn2 = self.clampprob(probcn2, pt[potcn2[0]])
            else:
                probcn1 = self.clampprob(probcn1, self.pt)
                probcn2 = self.clampprob(probcn2, self.pt)
            probcn1 = probcn1 * potcn1[-1]
            probcn2 = probcn2 * potcn2[-1]
            cnres1.set_value_(probcn1, layout="coo")
            cnres2.set_value_(probcn2, layout="coo")
            xcn1 = cnres1.sum(dim=-1).float().reshape(-1, 1)
            xcn2 = cnres2.sum(dim=-1).float().reshape(-1, 1)
            xcns[0] = xcns[0] + xcn2 + xcn1
        xij = self.xijlin(xij)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs

    def setalpha(self, alpha: float):
        self.alpha.fill_(alpha)
        print(f"set alpha: {alpha}")

    def forward(self,
                x,
                adj,
                tar_ei,
                filled1: bool = False,
                depth: int = None):
        if depth is None:
            depth = self.depth
        return self.multidomainforward(x, adj, tar_ei, filled1, [],
                                       depth)


# NCN predictor
class CNLinkPredictor(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        # optimized node features 
        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        xcns = [spmm_add(cn, x)]
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])


class CN1LinkPredictor(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels+1, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg
        self.final_lin = nn.Linear(3, 1) 

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        # optimized node features 
        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        cn_score = cn.sum(1).unsqueeze(1)
        xij_score = (xi * xj).sum(1).unsqueeze(1)
        
        xcns = torch.cat([spmm_add(cn, x), cn_score], dim=-1)
        xij = self.xijlin(xi * xj)
        
        xs = self.lin(self.xcnlin(xcns) * self.beta + xij)
        final_score = torch.cat([xs, xij_score, cn_score], dim=-1)
        xs = self.final_lin(final_score)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])
    
# fuse embedding from gcn and linkx
class FuseLinkPredictor(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels+1, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.hijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())        
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg
        self.final_lin = nn.Linear(3, 1) 

    def multidomainforward(self,
                           x,
                           h,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        
        hij = self.hijlin(h[tar_ei[0]]*h[tar_ei[1]])
        # optimized node features 
        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        cn_score = cn.sum(1).unsqueeze(1)
        xij_score = (xi * xj).sum(1).unsqueeze(1)
        
        xcns = torch.cat([spmm_add(cn, x), cn_score], dim=-1)
        xij = self.xijlin(xi * xj)
        
        xs = self.lin(self.xcnlin(xcns) * self.beta + xij)
        final_score = torch.cat([xs, xij_score, cn_score], dim=-1)
        xs = self.final_lin(final_score)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])
    
    
# GAE predictor for ablation study
class CN0LinkPredictor(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(xij)],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])

# NCNC predictor
class IncompleteCN1Predictor(CNLinkPredictor):
    learnablept: Final[bool]
    depth: Final[int]
    splitsize: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0,
                 alpha=1.0,
                 scale=5,
                 offset=3,
                 trainresdeg=8,
                 testresdeg=128,
                 pt=0.5,
                 learnablept=False,
                 depth=1,
                 splitsize=-1,
                 ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, edrop, ln, cndeg, use_xlin, tailact, twolayerlin, beta)
        self.learnablept= learnablept
        self.depth = depth
        self.splitsize = splitsize
        self.lins = nn.Sequential()
        self.register_buffer("alpha", torch.tensor([alpha]))
        self.register_buffer("pt", torch.tensor([pt]))
        self.register_buffer("scale", torch.tensor([scale]))
        self.register_buffer("offset", torch.tensor([offset]))

        self.trainresdeg = trainresdeg
        self.testresdeg = testresdeg
        self.ptlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True), nn.Linear(hidden_channels, 1), nn.Sigmoid())

    def clampprob(self, prob, pt):
        p0 = torch.sigmoid_(self.scale*(prob-self.offset))
        return self.alpha*pt*p0/(pt*p0+1-p0)

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = [],
                           depth: int=None):
        assert len(cndropprobs) == 0
        if depth is None:
            depth = self.depth
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        xij = xi*xj
        x = x + self.xlin(x)
        if depth > 0.5:
            cn, cnres1, cnres2 = adjoverlap(
                    adj,
                    adj,
                    tar_ei,
                    filled1,
                    calresadj=True,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)
        else:
            cn = adjoverlap(
                    adj,
                    adj,
                    tar_ei,
                    filled1,
                    calresadj=False,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)
        xcns = [spmm_add(cn, x)]
        
        if depth > 0.5:
            potcn1 = cnres1.coo()
            potcn2 = cnres2.coo()
            with torch.no_grad():
                if self.splitsize < 0:
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = self.forward(
                        x, adj, ei1,
                        filled1, depth-1).flatten()
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = self.forward(
                        x, adj, ei2,
                        filled1, depth-1).flatten()
                else:
                    num1 = potcn1[1].shape[0]
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = torch.empty_like(potcn1[1], dtype=torch.float)
                    for i in range(0, num1, self.splitsize):
                        probcn1[i:i+self.splitsize] = self.forward(x, adj, ei1[:, i: i+self.splitsize], filled1, depth-1).flatten()
                    num2 = potcn2[1].shape[0]
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = torch.empty_like(potcn2[1], dtype=torch.float)
                    for i in range(0, num2, self.splitsize):
                        probcn2[i:i+self.splitsize] = self.forward(x, adj, ei2[:, i: i+self.splitsize],filled1, depth-1).flatten()
            if self.learnablept:
                pt = self.ptlin(xij)
                probcn1 = self.clampprob(probcn1, pt[potcn1[0]]) 
                probcn2 = self.clampprob(probcn2, pt[potcn2[0]])
            else:
                probcn1 = self.clampprob(probcn1, self.pt)
                probcn2 = self.clampprob(probcn2, self.pt)
            probcn1 = probcn1 * potcn1[-1]
            probcn2 = probcn2 * potcn2[-1]
            cnres1.set_value_(probcn1, layout="coo")
            cnres2.set_value_(probcn2, layout="coo")
            xcn1 = spmm_add(cnres1, x)
            xcn2 = spmm_add(cnres2, x)
            xcns[0] = xcns[0] + xcn2 + xcn1
        
        xij = self.xijlin(xij)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs

    def setalpha(self, alpha: float):
        self.alpha.fill_(alpha)
        print(f"set alpha: {alpha}")

    def forward(self,
                x,
                adj,
                tar_ei,
                filled1: bool = False,
                depth: int = None):
        if depth is None:
            depth = self.depth
        return self.multidomainforward(x, adj, tar_ei, filled1, [],
                                       depth)


# NCN2 predictor
class CNhalf2LinkPredictor(CNLinkPredictor):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 ln=False,
                 tailact=False,
                 **kwargs):
        super().__init__(in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout, ln=ln, tailact=tailact, **kwargs)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()
        self.xcn12lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
            
    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        adj2 = adj@adj
        cn12 = adjoverlap(adj, adj2, tar_ei, filled1, cnsampledeg=self.cndeg)
        cn21 = adjoverlap(adj2, adj, tar_ei, filled1, cnsampledeg=self.cndeg)

        xcns = [(spmm_add(cn, x), spmm_add(cn12, x)+spmm_add(cn21, x))]
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn[0]) * self.beta + self.xcn12lin(xcn[1]) + xij) for xcn in xcns],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])



# NCN-diff
class CNResLinkPredictor(CNLinkPredictor):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 ln=False,
                 tailact=False,
                 **kwargs):
        super().__init__(in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout, ln=ln, tailact=tailact, **kwargs)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()
        self.xcnreslin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
            
    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        x = x + self.xlin(x)
        cn, cnres1, cnres2 = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg, calresadj=True)

        xcns = [(spmm_add(cn, x), spmm_add(cnres1, x)+spmm_add(cnres2, x))]
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn[0]) * self.beta + self.xcnreslin(xcn[1]) + xij) for xcn in xcns],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])

# NCN with higher order neighborhood overlaps than NCN-2
class CN2LinkPredictor(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1):
        super().__init__()

        self.lins = nn.Sequential()

        self.register_parameter("alpha", nn.Parameter(torch.ones((3))))
        self.register_parameter("beta", nn.Parameter(torch.ones((1))))
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xcn1lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn2lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn4lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, out_channels))

    def forward(self, x, adj: SparseTensor, tar_ei, filled1: bool = False):
        spadj = adj.to_torch_sparse_coo_tensor()
        adj2 = SparseTensor.from_torch_sparse_coo_tensor(spadj @ spadj, False)
        cn1 = adjoverlap(adj, adj, tar_ei, filled1)
        cn2 = adjoverlap(adj, adj2, tar_ei, filled1)
        cn3 = adjoverlap(adj2, adj, tar_ei, filled1)
        cn4 = adjoverlap(adj2, adj2, tar_ei, filled1)
        xij = self.xijlin(x[tar_ei[0]] * x[tar_ei[1]])
        xcn1 = self.xcn1lin(spmm_add(cn1, x))
        xcn2 = self.xcn2lin(spmm_add(cn2, x))
        xcn3 = self.xcn2lin(spmm_add(cn3, x))
        xcn4 = self.xcn4lin(spmm_add(cn4, x))
        alpha = torch.sigmoid(self.alpha).cumprod(-1)
        x = self.lin(alpha[0] * xcn1 + alpha[1] * xcn2 * xcn3 +
                     alpha[2] * xcn4 + self.beta * xij)
        return x


predictor_dict = {
    "cn0": CN0LinkPredictor,
    "catscn1": CatSCNLinkPredictor,
    "scn1": SCNLinkPredictor,
    "sincn1cn1": IncompleteSCN1Predictor,
    "cn1": CNLinkPredictor,
    "cn1.1": CN1LinkPredictor,
    "cn1.5": CNhalf2LinkPredictor,
    "cn1res": CNResLinkPredictor,
    "cn2": CN2LinkPredictor,
    "incn1cn1": IncompleteCN1Predictor,
    "fuse1": FuseLinkPredictor,
}


from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import MLP
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import (
    to_networkx,
    train_test_split_edges,
    to_undirected,
    spmm
)
from torch.nn import BatchNorm1d, Parameter
import math
from torch_geometric.nn import inits

class SparseLinear(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        inits.kaiming_uniform(self.weight, fan=self.in_channels,
                              a=math.sqrt(5))
        inits.uniform(self.in_channels, self.bias)

    def forward(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        # propagate_type: (weight: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, weight=self.weight,
                             edge_weight=edge_weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, weight_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is None:
            return weight_j
        else:
            return edge_weight.view(-1, 1) * weight_j

    def message_and_aggregate(self, adj_t: Adj, weight: Tensor) -> Tensor:
        return spmm(adj_t, weight, reduce=self.aggr)



class LINKX(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        num_edge_layers: int = 1,
        num_node_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_layers = num_edge_layers
        self.edge_lin = SparseLinear(num_nodes, hidden_channels)

        if self.num_edge_layers > 1:
            self.edge_norm = BatchNorm1d(hidden_channels)
            channels = [hidden_channels] * num_edge_layers
            self.edge_mlp = MLP(channels, dropout=0., act_first=True)
        else:
            self.edge_norm = None
            self.edge_mlp = None

        channels = [in_channels] + [hidden_channels] * num_node_layers
        self.node_mlp = MLP(channels, dropout=0., act_first=True)

        self.cat_lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.cat_lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

        channels = [hidden_channels] * num_layers + [out_channels]
        self.final_mlp = MLP(channels, dropout=dropout, act_first=True)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.edge_lin.reset_parameters()
        if self.edge_norm is not None:
            self.edge_norm.reset_parameters()
        if self.edge_mlp is not None:
            self.edge_mlp.reset_parameters()
        self.node_mlp.reset_parameters()
        self.cat_lin1.reset_parameters()
        self.cat_lin2.reset_parameters()
        self.final_mlp.reset_parameters()

    def forward(
        self,
        x: OptTensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        """"""  # noqa: D419
        out = self.edge_lin(edge_index, edge_weight)

        if self.edge_norm is not None and self.edge_mlp is not None:
            out = out.relu_()
            out = self.edge_norm(out)
            out = self.edge_mlp(out)
        out = out + self.cat_lin1(out)
        if x is not None:
            x = self.node_mlp(x)
            out = out + x
            out = out + self.cat_lin2(x)
        return self.final_mlp(out.relu_())



class LINKX_WL(torch.nn.Module):
    r"""The LINKX model from the `"Large Scale Learning on Non-Homophilous
    """
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        num_edge_layers: int = 1,
        num_node_layers: int = 1,
        dropout: float = 0.0,
        wl_emb_dim: int = 0,
        num_wl: int = 0,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_layers = num_edge_layers
        self.edge_lin = SparseLinear(num_nodes, hidden_channels)

        if self.num_edge_layers > 1:
            self.edge_norm = BatchNorm1d(hidden_channels)
            channels = [hidden_channels] * num_edge_layers
            self.edge_mlp = MLP(channels, dropout=0., act_first=True)
        else:
            self.edge_norm = None
            self.edge_mlp = None

        channels = [in_channels+wl_emb_dim] + [hidden_channels] * num_node_layers 
        self.node_mlp = MLP(channels, dropout=0., act_first=True)
        self.wl_emb = nn.Embedding(num_wl, wl_emb_dim)
        self.cat_lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.cat_lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        channels = [hidden_channels] * num_layers + [out_channels]
        self.final_mlp = MLP(channels, dropout=dropout, act_first=True)
        

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.edge_lin.reset_parameters()
        if self.edge_norm is not None:
            self.edge_norm.reset_parameters()
        if self.edge_mlp is not None:
            self.edge_mlp.reset_parameters()
        self.node_mlp.reset_parameters()
        self.cat_lin1.reset_parameters()
        self.cat_lin2.reset_parameters()
        self.final_mlp.reset_parameters()

    def forward(
        self,
        wl_indices: Tensor,
        x: OptTensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        """"""  # noqa: D419
        out = self.edge_lin(edge_index, edge_weight)
        if self.edge_norm is not None and self.edge_mlp is not None:
            out = out.relu_()
            out = self.edge_norm(out)
            out = self.edge_mlp(out)

        out = out + self.cat_lin1(out)
        if x is not None and wl_indices is not None:
            wl_embedding = self.wl_emb(wl_indices)
            x = torch.cat((x, wl_embedding), dim=1)
            x = self.node_mlp(x)
            out = out + x
            out = out + self.cat_lin2(x) 
        
        return self.final_mlp(out.relu_())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_nodes={self.num_nodes}, '
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels})')

