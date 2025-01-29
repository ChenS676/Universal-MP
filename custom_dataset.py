fromtorch_geometric.dataimportInMemoryDataset
fromsyn_randomimportRandomType
importos.pathasosp
fromtypingimportUnion
fromtorch_geometric.utilsimportfrom_networkx


classSyntheticRandom(InMemoryDataset):
def__init__(
self,
root:str,
name:str,
graphtype:RandomType,
transform=None,
N:int=10000,
):
self.dataset_name=name
self.N=N
self.graphtype=graphtype
super().__init__(root,transform)
self.load(self.processed_paths[0])

@property
defprocessed_dir(self)->str:
returnosp.join(self.root,self.__class__.__name__,'processed')

@property
defprocessed_file_names(self)->str:
returnf'{self.dataset_name}_{self.N}.pt'

defprocess(self):
fromsyn_randomimportinit_pyg_randomasgenerate_graph
graph_type_str=f"RandomType.{self.dataset_name}"
nx_data=generate_graph(self.N,eval(graph_type_str),seed=0)
data=from_networkx(nx_data)
self.save([data],self.processed_paths[0])



classSyntheticRegularTilling(InMemoryDataset):
def__init__(
self,
root:str,
name:str,
graphtype:RandomType,
transform=None,
N:int=10000,
):
self.dataset_name=name
self.N=N
self.graphtype=graphtype
super().__init__(root,transform)
self.load(self.processed_paths[0])

defprocess(self):
fromsyn_regulartillingimportinit_regular_tillingasgenerate_graph
graph_type_str=f"RegularTilling.{self.dataset_name}"
nx_data=generate_graph(self.N,eval(graph_type_str),seed=0)
data=from_networkx(nx_data)
self.save([data],self.processed_paths[0])




classSyntheticDataset(InMemoryDataset):
def__init__(
self,
root:str,
name:str,
graphtype:GraphType,
transform=None,
N:int=1000,
):
self.dataset_name=name
self.N=N
super().__init__(root,transform)
self.load(self.processed_paths[0])

@property
defprocessed_dir(self)->str:
returnosp.join(self.root,self.__class__.__name__,'processed')

@property
defprocessed_file_names(self)->str:
returnf'{self.dataset_name}_{self.N}.pt'

defprocess(self):
graph_type_str=f"GraphType.{self.dataset_name}"
nx_data=generate_graph(self.N,eval(graph_type_str),seed=0)
data=from_networkx(nx_data)
self.save([data],self.processed_paths[0])