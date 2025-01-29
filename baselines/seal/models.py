#Copyright(c)Facebook,Inc.anditsaffiliates.
#ThissourcecodeislicensedundertheMITlicensefoundinthe
#LICENSEfileintherootdirectoryofthissourcetree.

importmath
importnumpyasnp
importtorch
fromtorch.nnimport(ModuleList,Linear,Conv1d,MaxPool1d,Embedding,ReLU,
Sequential,BatchNorm1dasBN)
importtorch.nn.functionalasF
fromtorch_geometric.nnimport(GCNConv,SAGEConv,GINConv,
global_sort_pool,global_add_pool,global_mean_pool)
importpdb


classGCN(torch.nn.Module):
def__init__(self,hidden_channels,num_layers,max_z,train_dataset,
use_feature=False,node_embedding=None,dropout=0.5):
super(GCN,self).__init__()
self.use_feature=use_feature
self.node_embedding=node_embedding
self.max_z=max_z
self.z_embedding=Embedding(self.max_z,hidden_channels)

self.convs=ModuleList()
initial_channels=hidden_channels
ifself.use_feature:
initial_channels+=train_dataset.num_features
ifself.node_embeddingisnotNone:
initial_channels+=node_embedding.embedding_dim
self.convs.append(GCNConv(initial_channels,hidden_channels))
for_inrange(num_layers-1):
self.convs.append(GCNConv(hidden_channels,hidden_channels))

self.dropout=dropout
self.lin1=Linear(hidden_channels,hidden_channels)
self.lin2=Linear(hidden_channels,1)

defreset_parameters(self):
forconvinself.convs:
conv.reset_parameters()

defforward(self,z,edge_index,batch,x=None,edge_weight=None,node_id=None):
z_emb=self.z_embedding(z)
ifz_emb.ndim==3:#incasezhasmultipleintegerlabels
z_emb=z_emb.sum(dim=1)
ifself.use_featureandxisnotNone:
x=torch.cat([z_emb,x.to(torch.float)],1)
else:
x=z_emb
ifself.node_embeddingisnotNoneandnode_idisnotNone:
n_emb=self.node_embedding(node_id)
x=torch.cat([x,n_emb],1)
forconvinself.convs[:-1]:
x=conv(x,edge_index,edge_weight)
x=F.relu(x)
x=F.dropout(x,p=self.dropout,training=self.training)
x=self.convs[-1](x,edge_index,edge_weight)
ifTrue:#centerpooling
_,center_indices=np.unique(batch.cpu().numpy(),return_index=True)
x_src=x[center_indices]
x_dst=x[center_indices+1]
x=(x_src*x_dst)
x=F.relu(self.lin1(x))
x=F.dropout(x,p=self.dropout,training=self.training)
x=self.lin2(x)
else:#sumpooling
x=global_add_pool(x,batch)
x=F.relu(self.lin1(x))
x=F.dropout(x,p=self.dropout,training=self.training)
x=self.lin2(x)

returnx


classSAGE(torch.nn.Module):
def__init__(self,hidden_channels,num_layers,max_z,train_dataset=None,
use_feature=False,node_embedding=None,dropout=0.5):
super(SAGE,self).__init__()
self.use_feature=use_feature
self.node_embedding=node_embedding
self.max_z=max_z
self.z_embedding=Embedding(self.max_z,hidden_channels)

self.convs=ModuleList()
initial_channels=hidden_channels
ifself.use_feature:
initial_channels+=train_dataset.num_features
ifself.node_embeddingisnotNone:
initial_channels+=node_embedding.embedding_dim
self.convs.append(SAGEConv(initial_channels,hidden_channels))
for_inrange(num_layers-1):
self.convs.append(SAGEConv(hidden_channels,hidden_channels))

self.dropout=dropout
self.lin1=Linear(hidden_channels,hidden_channels)
self.lin2=Linear(hidden_channels,1)

defreset_parameters(self):
forconvinself.convs:
conv.reset_parameters()

defforward(self,z,edge_index,batch,x=None,edge_weight=None,node_id=None):
z_emb=self.z_embedding(z)
ifz_emb.ndim==3:#incasezhasmultipleintegerlabels
z_emb=z_emb.sum(dim=1)
ifself.use_featureandxisnotNone:
x=torch.cat([z_emb,x.to(torch.float)],1)
else:
x=z_emb
ifself.node_embeddingisnotNoneandnode_idisnotNone:
n_emb=self.node_embedding(node_id)
x=torch.cat([x,n_emb],1)
forconvinself.convs[:-1]:
x=conv(x,edge_index)
x=F.relu(x)
x=F.dropout(x,p=self.dropout,training=self.training)
x=self.convs[-1](x,edge_index)
ifTrue:#centerpooling
_,center_indices=np.unique(batch.cpu().numpy(),return_index=True)
x_src=x[center_indices]
x_dst=x[center_indices+1]
x=(x_src*x_dst)
x=F.relu(self.lin1(x))
x=F.dropout(x,p=self.dropout,training=self.training)
x=self.lin2(x)
else:#sumpooling
x=global_add_pool(x,batch)
x=F.relu(self.lin1(x))
x=F.dropout(x,p=self.dropout,training=self.training)
x=self.lin2(x)

returnx


#Anend-to-enddeeplearningarchitectureforgraphclassification,AAAI-18.
classDGCNN(torch.nn.Module):
def__init__(self,hidden_channels,num_layers,max_z,k=0.6,train_dataset=None,
dynamic_train=False,GNN=GCNConv,use_feature=False,
node_embedding=None):
super(DGCNN,self).__init__()

self.use_feature=use_feature
self.node_embedding=node_embedding

ifk<=1:#Transformpercentiletonumber.
iftrain_datasetisNone:
k=30
else:
ifdynamic_train:
sampled_train=train_dataset[:1000]
else:
sampled_train=train_dataset
num_nodes=sorted([g.num_nodesforginsampled_train])
k=num_nodes[int(math.ceil(k*len(num_nodes)))-1]
k=max(10,k)
self.k=int(k)

self.max_z=max_z
self.z_embedding=Embedding(self.max_z,hidden_channels)

self.convs=ModuleList()
initial_channels=hidden_channels
ifself.use_feature:
initial_channels+=train_dataset.num_features
ifself.node_embeddingisnotNone:
initial_channels+=node_embedding.embedding_dim

self.convs.append(GNN(initial_channels,hidden_channels))
foriinrange(0,num_layers-1):
self.convs.append(GNN(hidden_channels,hidden_channels))
self.convs.append(GNN(hidden_channels,1))

conv1d_channels=[16,32]
total_latent_dim=hidden_channels*num_layers+1
conv1d_kws=[total_latent_dim,5]
self.conv1=Conv1d(1,conv1d_channels[0],conv1d_kws[0],
conv1d_kws[0])
self.maxpool1d=MaxPool1d(2,2)
self.conv2=Conv1d(conv1d_channels[0],conv1d_channels[1],
conv1d_kws[1],1)
dense_dim=int((self.k-2)/2+1)
dense_dim=(dense_dim-conv1d_kws[1]+1)*conv1d_channels[1]
self.lin1=Linear(dense_dim,128)
self.lin2=Linear(128,1)

defforward(self,z,edge_index,batch,x=None,edge_weight=None,node_id=None):
z_emb=self.z_embedding(z)
ifz_emb.ndim==3:#incasezhasmultipleintegerlabels
z_emb=z_emb.sum(dim=1)
ifself.use_featureandxisnotNone:
x=torch.cat([z_emb,x.to(torch.float)],1)
else:
x=z_emb
ifself.node_embeddingisnotNoneandnode_idisnotNone:
n_emb=self.node_embedding(node_id)
x=torch.cat([x,n_emb],1)
xs=[x]

forconvinself.convs:
xs+=[torch.tanh(conv(xs[-1],edge_index,edge_weight))]
x=torch.cat(xs[1:],dim=-1)

#Globalpooling.
x=global_sort_pool(x,batch,self.k)
x=x.unsqueeze(1)#[num_graphs,1,k*hidden]
x=F.relu(self.conv1(x))
x=self.maxpool1d(x)
x=F.relu(self.conv2(x))
x=x.view(x.size(0),-1)#[num_graphs,dense_dim]

#MLP.
x=F.relu(self.lin1(x))
x=F.dropout(x,p=0.5,training=self.training)
x=self.lin2(x)
returnx


classGIN(torch.nn.Module):
def__init__(self,hidden_channels,num_layers,max_z,train_dataset,
use_feature=False,node_embedding=None,dropout=0.5,
jk=True,train_eps=False):
super(GIN,self).__init__()
self.use_feature=use_feature
self.node_embedding=node_embedding
self.max_z=max_z
self.z_embedding=Embedding(self.max_z,hidden_channels)
self.jk=jk

initial_channels=hidden_channels
ifself.use_feature:
initial_channels+=train_dataset.num_features
ifself.node_embeddingisnotNone:
initial_channels+=node_embedding.embedding_dim
self.conv1=GINConv(
Sequential(
Linear(initial_channels,hidden_channels),
ReLU(),
Linear(hidden_channels,hidden_channels),
ReLU(),
BN(hidden_channels),
),
train_eps=train_eps)
self.convs=torch.nn.ModuleList()
foriinrange(num_layers-1):
self.convs.append(
GINConv(
Sequential(
Linear(hidden_channels,hidden_channels),
ReLU(),
Linear(hidden_channels,hidden_channels),
ReLU(),
BN(hidden_channels),
),
train_eps=train_eps))

self.dropout=dropout
ifself.jk:
self.lin1=Linear(num_layers*hidden_channels,hidden_channels)
else:
self.lin1=Linear(hidden_channels,hidden_channels)
self.lin2=Linear(hidden_channels,1)

defforward(self,z,edge_index,batch,x=None,edge_weight=None,node_id=None):
z_emb=self.z_embedding(z)
ifz_emb.ndim==3:#incasezhasmultipleintegerlabels
z_emb=z_emb.sum(dim=1)
ifself.use_featureandxisnotNone:
x=torch.cat([z_emb,x.to(torch.float)],1)
else:
x=z_emb
ifself.node_embeddingisnotNoneandnode_idisnotNone:
n_emb=self.node_embedding(node_id)
x=torch.cat([x,n_emb],1)
x=self.conv1(x,edge_index)
xs=[x]
forconvinself.convs:
x=conv(x,edge_index)
xs+=[x]
ifself.jk:
x=global_mean_pool(torch.cat(xs,dim=1),batch)
else:
x=global_mean_pool(xs[-1],batch)
x=F.relu(self.lin1(x))
x=F.dropout(x,p=self.dropout,training=self.training)
x=self.lin2(x)

returnx

