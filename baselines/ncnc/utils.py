importtorch
fromtorch_sparseimportSparseTensor
fromtorchimportTensor
importtorch_sparse
fromtypingimportList,Tuple


classPermIterator:
'''
Iteratorofapermutation
'''
def__init__(self,device,size,bs,training=True)->None:
self.bs=bs
self.training=training
self.idx=torch.randperm(
size,device=device)iftrainingelsetorch.arange(size,
device=device)

def__len__(self):
return(self.idx.shape[0]+(self.bs-1)*
(notself.training))//self.bs

def__iter__(self):
self.ptr=0
returnself

def__next__(self):
ifself.ptr+self.bs*self.training>self.idx.shape[0]:
raiseStopIteration
ret=self.idx[self.ptr:self.ptr+self.bs]
self.ptr+=self.bs
returnret


defsparsesample(adj:SparseTensor,deg:int)->SparseTensor:
'''
samplingelementsfromaadjacencymatrix
'''
rowptr,col,_=adj.csr()
rowcount=adj.storage.rowcount()
mask=rowcount>0
rowcount=rowcount[mask]
rowptr=rowptr[:-1][mask]

rand=torch.rand((rowcount.size(0),deg),device=col.device)
rand.mul_(rowcount.to(rand.dtype).reshape(-1,1))
rand=rand.to(torch.long)
rand.add_(rowptr.reshape(-1,1))

samplecol=col[rand]

samplerow=torch.arange(adj.size(0),device=adj.device())[mask]

ret=SparseTensor(row=samplerow.reshape(-1,1).expand(-1,deg).flatten(),
col=samplecol.flatten(),
sparse_sizes=adj.sparse_sizes()).to_device(
adj.device()).coalesce().fill_value_(1.0)
#print(ret.storage.value())
returnret


defsparsesample2(adj:SparseTensor,deg:int)->SparseTensor:
'''
anotherimplementationforsamplingelementsfromaadjacencymatrix
'''
rowptr,col,_=adj.csr()
rowcount=adj.storage.rowcount()
mask=rowcount>deg

rowcount=rowcount[mask]
rowptr=rowptr[:-1][mask]

rand=torch.rand((rowcount.size(0),deg),device=col.device)
rand.mul_(rowcount.to(rand.dtype).reshape(-1,1))
rand=rand.to(torch.long)
rand.add_(rowptr.reshape(-1,1))

samplecol=col[rand].flatten()

samplerow=torch.arange(adj.size(0),device=adj.device())[mask].reshape(
-1,1).expand(-1,deg).flatten()

mask=torch.logical_not(mask)
nosamplerow,nosamplecol=adj[mask].coo()[:2]
nosamplerow=torch.arange(adj.size(0),
device=adj.device())[mask][nosamplerow]

ret=SparseTensor(
row=torch.cat((samplerow,nosamplerow)),
col=torch.cat((samplecol,nosamplecol)),
sparse_sizes=adj.sparse_sizes()).to_device(
adj.device()).fill_value_(1.0).coalesce()#.fill_value_(1)
#assert(ret.sum(dim=-1)==torch.clip(adj.sum(dim=-1),0,deg)).all()
returnret


defsparsesample_reweight(adj:SparseTensor,deg:int)->SparseTensor:
'''
anotherimplementationforsamplingelementsfromaadjacencymatrix.Itwillalsoscalethesampledelements.

'''
rowptr,col,_=adj.csr()
rowcount=adj.storage.rowcount()
mask=rowcount>deg

rowcount=rowcount[mask]
rowptr=rowptr[:-1][mask]

rand=torch.rand((rowcount.size(0),deg),device=col.device)
rand.mul_(rowcount.to(rand.dtype).reshape(-1,1))
rand=rand.to(torch.long)
rand.add_(rowptr.reshape(-1,1))

samplecol=col[rand].flatten()

samplerow=torch.arange(adj.size(0),device=adj.device())[mask].reshape(
-1,1).expand(-1,deg).flatten()
samplevalue=(rowcount*(1/deg)).reshape(-1,1).expand(-1,deg).flatten()

mask=torch.logical_not(mask)
nosamplerow,nosamplecol=adj[mask].coo()[:2]
nosamplerow=torch.arange(adj.size(0),
device=adj.device())[mask][nosamplerow]

ret=SparseTensor(row=torch.cat((samplerow,nosamplerow)),
col=torch.cat((samplecol,nosamplecol)),
value=torch.cat((samplevalue,
torch.ones_like(nosamplerow))),
sparse_sizes=adj.sparse_sizes()).to_device(
adj.device()).coalesce()#.fill_value_(1)
#assert(ret.sum(dim=-1)==torch.clip(adj.sum(dim=-1),0,deg)).all()
returnret


defelem2spm(element:Tensor,sizes:List[int])->SparseTensor:
#Convertadjacencymatrixtoa1-dvector
col=torch.bitwise_and(element,0xffffffff)
row=torch.bitwise_right_shift(element,32)
returnSparseTensor(row=row,col=col,sparse_sizes=sizes).to_device(
element.device).fill_value_(1.0)


defspm2elem(spm:SparseTensor)->Tensor:
#Convert1-dvectortoanadjacencymatrix
sizes=spm.sizes()
elem=torch.bitwise_left_shift(spm.storage.row(),
32).add_(spm.storage.col())
#elem=spm.storage.row()*sizes[-1]+spm.storage.col()
#asserttorch.all(torch.diff(elem)>0)
returnelem


defspmoverlap_(adj1:SparseTensor,adj2:SparseTensor)->SparseTensor:
'''
Computetheoverlapofneighbors(rowsinadj).Thereturnedmatrixissimilartothehadamardproductofadj1andadj2
'''
assertadj1.sizes()==adj2.sizes()
element1=spm2elem(adj1)
element2=spm2elem(adj2)

ifelement2.shape[0]>element1.shape[0]:
element1,element2=element2,element1

idx=torch.searchsorted(element1[:-1],element2)
mask=(element1[idx]==element2)
retelem=element2[mask]
'''
nnz1=adj1.nnz()
element=torch.cat((adj1.storage.row(),adj2.storage.row()),dim=-1)
element.bitwise_left_shift_(32)
element[:nnz1]+=adj1.storage.col()
element[nnz1:]+=adj2.storage.col()

element=torch.sort(element,dim=-1)[0]
mask=(element[1:]==element[:-1])
retelem=element[:-1][mask]
'''

returnelem2spm(retelem,adj1.sizes())


defspmnotoverlap_(adj1:SparseTensor,
adj2:SparseTensor)->Tuple[SparseTensor,SparseTensor]:
'''
returnelementsinadj1butnotinadj2andinadj2butnotadj1
'''
#assertadj1.sizes()==adj2.sizes()
element1=spm2elem(adj1)
element2=spm2elem(adj2)

idx=torch.searchsorted(element1[:-1],element2)
matchedmask=(element1[idx]==element2)

maskelem1=torch.ones_like(element1,dtype=torch.bool)
maskelem1[idx[matchedmask]]=0
retelem1=element1[maskelem1]

retelem2=element2[torch.logical_not(matchedmask)]
returnelem2spm(retelem1,adj1.sizes()),elem2spm(retelem2,adj2.sizes())


defspmoverlap_notoverlap_(
adj1:SparseTensor,
adj2:SparseTensor)->Tuple[SparseTensor,SparseTensor,SparseTensor]:
'''
returnelementsinadj1butnotinadj2andinadj2butnotadj1
'''
#assertadj1.sizes()==adj2.sizes()
element1=spm2elem(adj1)
element2=spm2elem(adj2)

ifelement1.shape[0]==0:
retoverlap=element1
retelem1=element1
retelem2=element2
else:
idx=torch.searchsorted(element1[:-1],element2)
matchedmask=(element1[idx]==element2)

maskelem1=torch.ones_like(element1,dtype=torch.bool)
maskelem1[idx[matchedmask]]=0
retelem1=element1[maskelem1]

retoverlap=element2[matchedmask]
retelem2=element2[torch.logical_not(matchedmask)]
sizes=adj1.sizes()
returnelem2spm(retoverlap,
sizes),elem2spm(retelem1,
sizes),elem2spm(retelem2,sizes)


defadjoverlap(adj1:SparseTensor,
adj2:SparseTensor,
tarei:Tensor,
filled1:bool=False,
calresadj:bool=False,
cnsampledeg:int=-1,
ressampledeg:int=-1):
#awrapperforfunctionsabove.
adj1=adj1[tarei[0]]
adj2=adj2[tarei[1]]
ifcalresadj:
adjoverlap,adjres1,adjres2=spmoverlap_notoverlap_(adj1,adj2)
ifcnsampledeg>0:
adjoverlap=sparsesample_reweight(adjoverlap,cnsampledeg)
ifressampledeg>0:
adjres1=sparsesample_reweight(adjres1,ressampledeg)
adjres2=sparsesample_reweight(adjres2,ressampledeg)
returnadjoverlap,adjres1,adjres2
else:
adjoverlap=spmoverlap_(adj1,adj2)
ifcnsampledeg>0:
adjoverlap=sparsesample_reweight(adjoverlap,cnsampledeg)
returnadjoverlap


if__name__=="__main__":
adj1=SparseTensor.from_edge_index(
torch.LongTensor([[0,0,1,2,3],[0,1,1,2,3]]))
adj2=SparseTensor.from_edge_index(
torch.LongTensor([[0,3,1,2,3],[0,1,1,2,3]]))
adj3=SparseTensor.from_edge_index(
torch.LongTensor([[0,1,2,2,2,2,3,3,3],[1,0,2,3,4,5,4,5,6]]))
print(spmnotoverlap_(adj1,adj2))
print(spmoverlap_(adj1,adj2))
print(spmoverlap_notoverlap_(adj1,adj2))
print(sparsesample2(adj3,2))
print(sparsesample_reweight(adj3,2))