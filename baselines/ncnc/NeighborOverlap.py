importargparse
importnumpyasnp
importtorch
importtorch.nn.functionalasF
importtorch.nnasnn
fromtorch_sparseimportSparseTensor
importtorch_geometric.transformsasT
frommodelimportpredictor_dict,convdict,GCN,DropEdge
fromfunctoolsimportpartial
fromsklearn.metricsimportroc_auc_score,average_precision_score
fromogb.linkproppredimportPygLinkPropPredDataset,Evaluator
fromtorch_geometric.utilsimportnegative_sampling
fromtorch.utils.tensorboardimportSummaryWriter
fromutilsimportPermIterator
importtime
fromogbdatasetimportloaddataset
fromtypingimportIterable


defset_seed(seed):
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


deftrain(model,
predictor,
data,
split_edge,
optimizer,
batch_size,
maskinput:bool=True,
cnprobs:Iterable[float]=[],
alpha:float=None):

ifalphaisnotNone:
predictor.setalpha(alpha)

model.train()
predictor.train()

pos_train_edge=split_edge['train']['edge'].to(data.x.device)
pos_train_edge=pos_train_edge.t()

total_loss=[]
adjmask=torch.ones_like(pos_train_edge[0],dtype=torch.bool)

negedge=negative_sampling(data.edge_index.to(pos_train_edge.device),data.adj_t.sizes()[0])
forperminPermIterator(
adjmask.device,adjmask.shape[0],batch_size
):
optimizer.zero_grad()
ifmaskinput:
adjmask[perm]=0
tei=pos_train_edge[:,adjmask]
adj=SparseTensor.from_edge_index(tei,
sparse_sizes=(data.num_nodes,data.num_nodes)).to_device(
pos_train_edge.device,non_blocking=True)
adjmask[perm]=1
adj=adj.to_symmetric()
else:
adj=data.adj_t
h=model(data.x,adj)
edge=pos_train_edge[:,perm]
pos_outs=predictor.multidomainforward(h,
adj,
edge,
cndropprobs=cnprobs)

pos_losss=-F.logsigmoid(pos_outs).mean()
edge=negedge[:,perm]
neg_outs=predictor.multidomainforward(h,adj,edge,cndropprobs=cnprobs)
neg_losss=-F.logsigmoid(-neg_outs).mean()
loss=neg_losss+pos_losss
loss.backward()
optimizer.step()

total_loss.append(loss)
total_loss=np.average([_.item()for_intotal_loss])
returntotal_loss


@torch.no_grad()
deftest(model,predictor,data,split_edge,evaluator,batch_size,
use_valedges_as_input):
model.eval()
predictor.eval()

pos_train_edge=split_edge['train']['edge'].to(data.adj_t.device())
pos_valid_edge=split_edge['valid']['edge'].to(data.adj_t.device())
neg_valid_edge=split_edge['valid']['edge_neg'].to(data.adj_t.device())
pos_test_edge=split_edge['test']['edge'].to(data.adj_t.device())
neg_test_edge=split_edge['test']['edge_neg'].to(data.adj_t.device())

adj=data.adj_t
h=model(data.x,adj)


pos_train_pred=torch.cat([
predictor(h,adj,pos_train_edge[perm].t()).squeeze().cpu()
forperminPermIterator(pos_train_edge.device,
pos_train_edge.shape[0],batch_size,False)
],
dim=0)


pos_valid_pred=torch.cat([
predictor(h,adj,pos_valid_edge[perm].t()).squeeze().cpu()
forperminPermIterator(pos_valid_edge.device,
pos_valid_edge.shape[0],batch_size,False)
],
dim=0)
neg_valid_pred=torch.cat([
predictor(h,adj,neg_valid_edge[perm].t()).squeeze().cpu()
forperminPermIterator(neg_valid_edge.device,
neg_valid_edge.shape[0],batch_size,False)
],
dim=0)
ifuse_valedges_as_input:
adj=data.full_adj_t
h=model(data.x,adj)

pos_test_pred=torch.cat([
predictor(h,adj,pos_test_edge[perm].t()).squeeze().cpu()
forperminPermIterator(pos_test_edge.device,pos_test_edge.shape[0],
batch_size,False)
],
dim=0)

neg_test_pred=torch.cat([
predictor(h,adj,neg_test_edge[perm].t()).squeeze().cpu()
forperminPermIterator(neg_test_edge.device,neg_test_edge.shape[0],
batch_size,False)
],
dim=0)

results={}
forKin[20,50,100]:
evaluator.K=K

train_hits=evaluator.eval({
'y_pred_pos':pos_train_pred,
'y_pred_neg':neg_valid_pred,
})[f'hits@{K}']

valid_hits=evaluator.eval({
'y_pred_pos':pos_valid_pred,
'y_pred_neg':neg_valid_pred,
})[f'hits@{K}']
test_hits=evaluator.eval({
'y_pred_pos':pos_test_pred,
'y_pred_neg':neg_test_pred,
})[f'hits@{K}']

results[f'Hits@{K}']=(train_hits,valid_hits,test_hits)
returnresults,h.cpu()


defparseargs():
parser=argparse.ArgumentParser()
parser.add_argument('--use_valedges_as_input',action='store_true',help="whethertoaddvalidationedgestotheinputadjacencymatrixofgnn")
parser.add_argument('--epochs',type=int,default=40,help="numberofepochs")
parser.add_argument('--runs',type=int,default=3,help="numberofrepeatedruns")
parser.add_argument('--dataset',type=str,default="collab")

parser.add_argument('--batch_size',type=int,default=8192,help="batchsize")
parser.add_argument('--testbs',type=int,default=8192,help="batchsizefortest")
parser.add_argument('--maskinput',action="store_true",help="whethertousetargetlinkremoval")

parser.add_argument('--mplayers',type=int,default=1,help="numberofmessagepassinglayers")
parser.add_argument('--nnlayers',type=int,default=3,help="numberofmlplayers")
parser.add_argument('--hiddim',type=int,default=32,help="hiddendimension")
parser.add_argument('--ln',action="store_true",help="whethertouselayernorminMPNN")
parser.add_argument('--lnnn',action="store_true",help="whethertouselayernorminmlp")
parser.add_argument('--res',action="store_true",help="whethertouseresidualconnection")
parser.add_argument('--jk',action="store_true",help="whethertouseJumpingKnowledgeconnection")
parser.add_argument('--gnndp',type=float,default=0.3,help="dropoutratioofgnn")
parser.add_argument('--xdp',type=float,default=0.3,help="dropoutratioofgnn")
parser.add_argument('--tdp',type=float,default=0.3,help="dropoutratioofgnn")
parser.add_argument('--gnnedp',type=float,default=0.3,help="edgedropoutratioofgnn")
parser.add_argument('--predp',type=float,default=0.3,help="dropoutratioofpredictor")
parser.add_argument('--preedp',type=float,default=0.3,help="edgedropoutratioofpredictor")
parser.add_argument('--gnnlr',type=float,default=0.0003,help="learningrateofgnn")
parser.add_argument('--prelr',type=float,default=0.0003,help="learningrateofpredictor")
#detailedhyperparameters
parser.add_argument('--beta',type=float,default=1)
parser.add_argument('--alpha',type=float,default=1)
parser.add_argument("--use_xlin",action="store_true")
parser.add_argument("--tailact",action="store_true")
parser.add_argument("--twolayerlin",action="store_true")
parser.add_argument("--increasealpha",action="store_true")

parser.add_argument('--splitsize',type=int,default=-1,help="splitsomeoperationsinnerthemodel.OnlyspeedandGPUmemoryconsumptionareaffected.")

#parametersusedtocalibratetheedgeexistenceprobabilityinNCNC
parser.add_argument('--probscale',type=float,default=5)
parser.add_argument('--proboffset',type=float,default=3)
parser.add_argument('--pt',type=float,default=0.5)
parser.add_argument("--learnpt",action="store_true")

#Forscalability,NCNCsamplesneighborstocompletecommonneighbor.
parser.add_argument('--trndeg',type=int,default=-1,help="maximumnumberofsampledneighborsduringthetrainingprocess.-1meansnosample")
parser.add_argument('--tstdeg',type=int,default=-1,help="maximumnumberofsampledneighborsduringthetestprocess")
#NCNcansamplecommonneighborsforscalability.Generallynotused.
parser.add_argument('--cndeg',type=int,default=-1)

#predictorused,suchasNCN,NCNC
parser.add_argument('--predictor',choices=predictor_dict.keys())
parser.add_argument("--depth",type=int,default=1,help="numberofcompletionstepsinNCNC")
#gnnused,suchasgin,gcn.
parser.add_argument('--model',choices=convdict.keys())

parser.add_argument('--save_gemb',action="store_true",help="whethertosavenoderepresentationsproducedbyGNN")
parser.add_argument('--load',type=str,help="wheretoloadnoderepresentationsproducedbyGNN")
parser.add_argument("--loadmod",action="store_true",help="whethertoloadtrainedmodels")
parser.add_argument("--savemod",action="store_true",help="whethertosavetrainedmodels")

parser.add_argument("--savex",action="store_true",help="whethertosavetrainednodeembeddings")
parser.add_argument("--loadx",action="store_true",help="whethertoloadtrainednodeembeddings")


#notusedinexperiments
parser.add_argument('--cnprob',type=float,default=0)
args=parser.parse_args()
returnargs


defmain():
args=parseargs()
print(args,flush=True)

hpstr=str(args).replace("","").replace("Namespace(","").replace(
")","").replace("True","1").replace("False","0").replace("=","").replace("epochs","").replace("runs","").replace("save_gemb","")
writer=SummaryWriter(f"./rec/{args.model}_{args.predictor}")
writer.add_text("hyperparams",hpstr)

ifargs.datasetin["Cora","Citeseer","Pubmed"]:
evaluator=Evaluator(name=f'ogbl-ppa')
else:
evaluator=Evaluator(name=f'ogbl-{args.dataset}')

device=torch.device(f'cuda'iftorch.cuda.is_available()else'cpu')
data,split_edge=loaddataset(args.dataset,args.use_valedges_as_input,args.load)
data=data.to(device)

predfn=predictor_dict[args.predictor]
ifargs.predictor!="cn0":
predfn=partial(predfn,cndeg=args.cndeg)
ifargs.predictorin["cn1","incn1cn1","scn1","catscn1","sincn1cn1"]:
predfn=partial(predfn,use_xlin=args.use_xlin,tailact=args.tailact,twolayerlin=args.twolayerlin,beta=args.beta)
ifargs.predictor=="incn1cn1":
predfn=partial(predfn,depth=args.depth,splitsize=args.splitsize,scale=args.probscale,offset=args.proboffset,trainresdeg=args.trndeg,testresdeg=args.tstdeg,pt=args.pt,learnablept=args.learnpt,alpha=args.alpha)

ret=[]

forruninrange(0,args.runs):
set_seed(run)
ifargs.datasetin["Cora","Citeseer","Pubmed"]:
data,split_edge=loaddataset(args.dataset,args.use_valedges_as_input,args.load)#getanewsplitofdataset
data=data.to(device)
bestscore=None

#buildmodel
model=GCN(data.num_features,args.hiddim,args.hiddim,args.mplayers,
args.gnndp,args.ln,args.res,data.max_x,
args.model,args.jk,args.gnnedp,xdropout=args.xdp,taildropout=args.tdp,noinputlin=args.loadx).to(device)
ifargs.loadx:
withtorch.no_grad():
model.xemb[0].weight.copy_(torch.load(f"gemb/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt",map_location="cpu"))
model.xemb[0].weight.requires_grad_(False)
predictor=predfn(args.hiddim,args.hiddim,1,args.nnlayers,
args.predp,args.preedp,args.lnnn).to(device)
ifargs.loadmod:
keys=model.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt",map_location="cpu"),strict=False)
print("unmatchedparams",keys,flush=True)
keys=predictor.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pre.pt",map_location="cpu"),strict=False)
print("unmatchedparams",keys,flush=True)


optimizer=torch.optim.Adam([{'params':model.parameters(),"lr":args.gnnlr},
{'params':predictor.parameters(),'lr':args.prelr}])

forepochinrange(1,1+args.epochs):
alpha=max(0,min((epoch-5)*0.1,1))ifargs.increasealphaelseNone
t1=time.time()
loss=train(model,predictor,data,split_edge,optimizer,
args.batch_size,args.maskinput,[],alpha)
print(f"trntime{time.time()-t1:.2f}s",flush=True)
ifTrue:
t1=time.time()
results,h=test(model,predictor,data,split_edge,evaluator,
args.testbs,args.use_valedges_as_input)
print(f"testtime{time.time()-t1:.2f}s")
ifbestscoreisNone:
bestscore={key:list(results[key])forkeyinresults}
forkey,resultinresults.items():
writer.add_scalars(f"{key}_{run}",{
"trn":result[0],
"val":result[1],
"tst":result[2]
},epoch)

ifTrue:
forkey,resultinresults.items():
train_hits,valid_hits,test_hits=result
ifvalid_hits>bestscore[key][1]:
bestscore[key]=list(result)
ifargs.save_gemb:
torch.save(h,f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}.pt")
ifargs.savex:
torch.save(model.xemb[0].weight.detach(),f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
ifargs.savemod:
torch.save(model.state_dict(),f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
torch.save(predictor.state_dict(),f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pre.pt")
print(key)
print(f'Run:{run+1:02d},'
f'Epoch:{epoch:02d},'
f'Loss:{loss:.4f},'
f'Train:{100*train_hits:.2f}%,'
f'Valid:{100*valid_hits:.2f}%,'
f'Test:{100*test_hits:.2f}%')
print('---',flush=True)
print(f"best{bestscore}")
ifargs.dataset=="collab":
ret.append(bestscore["Hits@50"][-2:])
elifargs.dataset=="ppa":
ret.append(bestscore["Hits@100"][-2:])
elifargs.dataset=="ddi":
ret.append(bestscore["Hits@20"][-2:])
elifargs.dataset=="citation2":
ret.append(bestscore[-2:])
elifargs.datasetin["Pubmed","Cora","Citeseer"]:
ret.append(bestscore["Hits@100"][-2:])
else:
raiseNotImplementedError
ret=np.array(ret)
print(ret)
print(f"Finalresult:val{np.average(ret[:,0]):.4f}{np.std(ret[:,0]):.4f}tst{np.average(ret[:,1]):.4f}{np.std(ret[:,1]):.4f}")


if__name__=="__main__":
main()