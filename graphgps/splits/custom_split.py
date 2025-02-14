import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data

class CustomRandomLinkSplit(RandomLinkSplit):
    def __init__(self, num_val: float = 0.1, num_test: float = 0.2,
                 is_undirected: bool = False, add_negative_train_samples: bool = True):
        super().__init__(num_val=num_val, num_test=num_test, 
                         is_undirected=is_undirected, 
                         add_negative_train_samples=add_negative_train_samples)
    
    def __call__(self, data: Data):
        # Custom pre-processing or checks can be added here
        print("Custom preprocessing before split")
        
        # Call the parent class's __call__ method to perform the actual split
        train_data, val_data, test_data = super().__call__(data)
        
        # Custom post-processing can be added here
        print("Custom postprocessing after split")
        
        return train_data, val_data, test_data

# Example usage:
data = Data()  # Assume this is your graph data object
transform = CustomRandomLinkSplit(num_val=0.15, num_test=0.25)
train_data, val_data, test_data = transform(data)

# Verify the splits
print(train_data)
print(val_data)
print(test_data)



# adopted from https://github.com/LARS-research/HL-GNN/blob/main/OGB/utils.py
import torch
from torch_geometric.utils import negative_sampling, add_self_loops


def global_neg_sample(edge_index, num_nodes, num_samples,
                      num_neg, method='sparse'):
    new_edge_index, _ = add_self_loops(edge_index)
    neg_edge = negative_sampling(new_edge_index, num_nodes=num_nodes,
                                 num_neg_samples=num_samples * num_neg, method=method)

    neg_src = neg_edge[0]
    neg_dst = neg_edge[1]
    if neg_edge.size(1) < num_samples * num_neg:
        k = num_samples * num_neg - neg_edge.size(1)
        rand_index = torch.randperm(neg_edge.size(1))[:k]
        neg_src = torch.cat((neg_src, neg_src[rand_index]))
        neg_dst = torch.cat((neg_dst, neg_dst[rand_index]))
    return torch.reshape(torch.stack((neg_src, neg_dst), dim=-1), (-1, num_neg, 2))


def global_perm_neg_sample(edge_index, num_nodes, num_samples,
                           num_neg, method='sparse'):
    new_edge_index, _ = add_self_loops(edge_index)
    neg_edge = negative_sampling(new_edge_index, num_nodes=num_nodes,
                                 num_neg_samples=num_samples, method=method)
    return sample_perm_copy(neg_edge, num_samples, num_neg)


def local_neg_sample(pos_edges, num_nodes, num_neg, random_src=False):
    if random_src:
        neg_src = pos_edges[torch.arange(pos_edges.size(0)), torch.randint(0, 2, (pos_edges.size(0),), dtype=torch.long)]
    else:
        neg_src = pos_edges[:, 0]
    neg_src = torch.reshape(neg_src, (-1, 1)).repeat(1, num_neg)
    neg_src = torch.reshape(neg_src, (-1,))
    neg_dst = torch.randint(0, num_nodes, (num_neg * pos_edges.size(0),), dtype=torch.long)

    return torch.reshape(torch.stack((neg_src, neg_dst), dim=-1), (-1, num_neg, 2))



def sample_perm_copy(edge_index, target_num_sample, num_perm_copy):
    src = edge_index[0]
    dst = edge_index[1]
    if edge_index.size(1) < target_num_sample:
        k = target_num_sample - edge_index.size(1)
        rand_index = torch.randperm(edge_index.size(1))[:k]
        src = torch.cat((src, src[rand_index]))
        dst = torch.cat((dst, dst[rand_index]))
    tmp_src = src
    tmp_dst = dst
    for i in range(num_perm_copy - 1):
        rand_index = torch.randperm(target_num_sample)
        src = torch.cat((src, tmp_src[rand_index]))
        dst = torch.cat((dst, tmp_dst[rand_index]))
    return torch.reshape(torch.stack(
        (src, dst), dim=-1), (-1, num_perm_copy, 2))
    
    
def get_pos_neg_edges(split, split_edge, edge_index=None, num_nodes=None, neg_sampler_name=None, num_neg=None):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge']
    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        pos_edge = torch.stack([source, target]).t()

    if split == 'train':
        if neg_sampler_name == 'local':
            neg_edge = local_neg_sample(
                pos_edge,
                num_nodes=num_nodes,
                num_neg=num_neg)
        elif neg_sampler_name == 'global':
            neg_edge = global_neg_sample(
                edge_index,
                num_nodes=num_nodes,
                num_samples=pos_edge.size(0),
                num_neg=num_neg)
        else:
            neg_edge = global_perm_neg_sample(
                edge_index,
                num_nodes=num_nodes,
                num_samples=pos_edge.size(0),
                num_neg=num_neg)
    else:
        if 'edge' in split_edge['train']:
            neg_edge = split_edge[split]['edge_neg']
        elif 'source_node' in split_edge['train']:
            target_neg = split_edge[split]['target_node_neg']
            neg_per_target = target_neg.size(1)
            neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                    target_neg.view(-1)]).t()
    return pos_edge, neg_edge


def generate_neg_dist_table(num_nodes, adj_t, power=0.75, table_size=1e8):
    table_size = int(table_size)
    adj_t = adj_t.set_diag()
    node_degree = adj_t.sum(dim=1).to(torch.float)
    node_degree = node_degree.pow(power)

    norm = float((node_degree).sum())  # float is faster than tensor when visited
    node_degree = node_degree.tolist()  # list has fastest visit speed
    sample_table = np.zeros(table_size, dtype=np.int32)
    p = 0
    i = 0
    for j in range(num_nodes):
        p += node_degree[j] / norm
        while i < table_size and float(i) / float(table_size) < p:
            sample_table[i] = j
            i += 1
    sample_table = torch.from_numpy(sample_table)
    return sample_table