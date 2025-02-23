import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import time
from tqdm import tqdm

from metrics.metrics import *
from data_utils.graph_rewiring import apply_KNN
from ogb.linkproppred import Evaluator
from torch_geometric.utils import negative_sampling
from utils.utils import PermIterator


class Trainer_GRAND_Citation2:
    def __init__(self,
                 opt,
                 model,
                 predictor,
                 optimizer,
                 data,
                 pos_encoding,
                 splits,
                 batch_size,
                 device,
                 log_dir='./logs'):
        self.opt = opt
        self.model = model.to(device)
        self.predictor = predictor.to(device)
        self.optimizer = optimizer
        self.data = data
        self.pos_encoding = pos_encoding
        self.splits = splits
        self.batch_size = batch_size
        self.device = device
        self.epochs = opt['epoch']
        self.batch_size = opt['batch_size']
        self.log_dir = log_dir

        self.results_file = os.path.join(log_dir, f"{opt['dataset']}_results.txt")
        os.makedirs(log_dir, exist_ok=True)

        self.best_epoch = 0
        self.best_metric = 0
        self.best_results = None
        
        # Preprocessing 
        

    def train_epoch(self):
        self.predictor.train()
        self.model.train()
        
        pos_encoding = self.pos_encoding.to(self.model.device) if self.pos_encoding is not None else None
        source_edge = self.splits['train']['source_node'].to(self.data.x.device)
        target_edge = self.splits['train']['target_node'].to(self.data.x.device)
        pos_train_edge = torch.stack((source_edge, target_edge), dim=0)

        neg_train_edge = negative_sampling(
            self.data.edge_index.to(pos_train_edge.device),
            num_nodes=self.data.num_nodes,
            num_neg_samples=pos_train_edge.size(0)
        ).t().to(self.data.x.device)

        total_loss = total_examples = 0
        indices = torch.randperm(pos_train_edge.size(0), device=pos_train_edge.device)

    
        for start in tqdm(range(0, pos_train_edge.size(0), self.batch_size)):
            import IPython; IPython.embed()
            self.optimizer.zero_grad()
            h = self.model(self.data.x, pos_encoding)
            
            end = start + self.batch_size
            perm = indices[start:end]
            pos_out = self.predictor(h[pos_train_edge[perm, 0]], h[pos_train_edge[perm, 1]])
            pos_loss = -torch.log(pos_out + 1e-15).mean()
            neg_out = self.predictor(h[neg_train_edge[perm, 0]], h[neg_train_edge[perm, 1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            loss = pos_loss + neg_loss
            
            if self.model.odeblock.nreg > 0:
                reg_states = tuple(torch.mean(rs) for rs in self.model.reg_states)
                regularization_coeffs = self.model.regularization_coeffs
                reg_loss = sum(
                    reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                )
                loss = loss + reg_loss
            
            num_examples = (end - start)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
            
            self.model.fm.update(self.model.getNFE())
            self.model.resetNFE()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
            self.optimizer.step()
            self.model.bm.update(self.model.getNFE())
            self.model.resetNFE()
            #######################
            
        return total_loss / total_examples
    
    @torch.no_grad()
    def test_epoch(self):
        # copy from NCNC NeighborOverlapCitation2
        self.model.eval()
        self.predictor.eval()
        
        evaluator = Evaluator(name='ogbl-citation2')

        h = self.model(self.data.x, self.pos_encoding)
        
        def test_split(split):
            source = self.splits[split]['source_node'].to(h.device)
            target = self.splits[split]['target_node'].to(h.device)
            target_neg = self.splits[split]['target_node_neg'].to(h.device)


            pos_preds = []
            for perm in PermIterator(source.device, source.shape[0], self.batch_size, False):
                src, dst = source[perm], target[perm]
                pos_preds += [self.predictor(h[src], h[dst]).squeeze().cpu()]
            pos_pred = torch.cat(pos_preds, dim=0)

            neg_preds = []
            source = source.view(-1, 1).repeat(1, 1000).view(-1)
            target_neg = target_neg.view(-1)
            for perm in PermIterator(source.device, source.shape[0], self.batch_size, False):
                src, dst_neg = source[perm], target_neg[perm]
                neg_preds += [self.predictor(h[src], h[dst_neg]).squeeze().cpu()]
            neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

            mrr = evaluator.eval({
                'y_pred_pos': pos_pred,
                'y_pred_neg': neg_pred,
            })['mrr_list'].mean().item()
            
            return mrr

        valid_mrr = test_split('valid')
        test_mrr = test_split('test')

        # NO AUC
        return valid_mrr, test_mrr 
    
    def log_results(self, results, epoch):
        try:
            with open(self.results_file, 'a') as file:
                file.write(f"Epoch: {epoch}\n")
                for key, value in results.items():
                    file.write(f"{key}: {value}\n")
                file.write("\n")
            print(f"Results saved to {self.results_file}")
        except Exception as e:
            print(f"Failed to save results: {e}")


    def train(self):
        print(f"Starting training for {self.epochs} epochs...")
        for epoch in tqdm(range(1, self.epochs + 1)):
            start_time = time.time()

            # CHECK Misalignment
            if self.opt['rewire_KNN'] and epoch % self.opt['rewire_KNN_epoch'] == 0 and epoch != 0:
                ei = apply_KNN(self.data, self.pos_encoding, self.model, self.opt)
                self.model.odeblock.odefunc.edge_index = ei
                # self.data.edge_index = ei
                
            loss = self.train_epoch()
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            if epoch % 1 == 0:
                _, test_mrr = self.test_epoch()
                self.log_results(test_mrr, epoch)

                current_metric = test_mrr
                if current_metric > self.best_metric:
                    self.best_epoch = epoch
                    self.best_metric = current_metric
                    self.best_results = test_mrr
                print(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s")
                print(f"Current Best {current_metric}: {self.best_metric:.4f} (Epoch {self.best_epoch})")
        print(f"Training completed. Best {current_metric}: {self.best_metric:.4f} (Epoch {self.best_epoch})")
        return self.best_results

    def finalize(self):
        if self.best_results:
            print(f"Final Best Results:")
            print(f"MRR: {self.best_metric}")
        else:
            print("No results to finalize.")
