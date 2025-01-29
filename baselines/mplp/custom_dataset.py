from torch_geometric.data import Data, Dataset, InMemoryDataset

import os.path as osp

from torch_geometric.utils import coalesce, to_undirected, from_networkx

from torch_geometric.data import InMemoryDataset
from syn_random import RandomType
import os.path as osp
from typing import Union
from torch_geometric.utils import from_networkx

from graph_generation import generate_graph, GraphType
from syn_random import RandomType
from syn_regulartilling import RegularTilling


class SyntheticDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        N: int=1000,
    ):
        self.dataset_name = name
        self.N = N
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, 'processed')

    @property
    def processed_file_names(self) -> str:
        return f'{self.dataset_name}_{self.N}.pt'

    def process(self):
        graph_type_str = f"GraphType.{self.dataset_name}"
        from graph_generation import generate_graph
        nx_data = generate_graph(self.N, eval(graph_type_str), seed=0)
        data = from_networkx(nx_data)
        self.save([data], self.processed_paths[0])
        
        
        
class SyntheticRandom(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        N: int=1000,
    ):
        self.dataset_name = name
        self.N = N
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, 'processed')

    @property
    def processed_file_names(self) -> str:
        return f'{self.dataset_name}_{self.N}.pt'

    def process(self):
        from syn_random import init_random_graph as generate_graph
        graph_type_str = f"RandomType.{self.dataset_name}"
        nx_data = generate_graph(self.N, eval(graph_type_str), seed=0)[0]
        data = from_networkx(nx_data)
        self.save([data], self.processed_paths[0])



class SyntheticRegularTilling(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        N: int=100,
    ):
        self.dataset_name = name
        self.N = N
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, 'processed')

    @property
    def processed_file_names(self) -> str:
        return f'{self.dataset_name}_{self.N}.pt'


    def process(self):
        from syn_regulartilling import init_regular_tilling as generate_graph
        graph_type_str = f"RegularTilling.{self.dataset_name}"
        nx_data = generate_graph(self.N, eval(graph_type_str), seed=0)[0]
        data = from_networkx(nx_data)
        self.save([data], self.processed_paths[0])




