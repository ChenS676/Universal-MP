import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import time
from tqdm import tqdm

from metrics.metrics import *
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from utils.utils import PermIterator
class Trainer_GRAND:
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
        self.batch_size = self.opt['batch_size']
        self.log_dir = log_dir

        self.results_file = os.path.join(log_dir, 'results.txt')
        os.makedirs(log_dir, exist_ok=True)

        self.best_epoch = 0
        self.best_metric = 0
        self.best_results = None

    def train_epoch(self):
        self.predictor.train()
        self.model.train()
        
        pos_encoding = self.pos_encoding.to(self.model.device) if self.pos_encoding is not None else None
        
        pos_train_edge = self.splits['train']['edge'].t().to(self.data.x.device)
        neg_edge = negative_sampling(self.data.edge_index.to(pos_train_edge.device), 
                                     self.data.adj_t.sizes()[0]).to(self.data.x.device)
        
        total_loss = total_examples = 0
        
        adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool) # mask for adj
        pos_weight = 1.0
        neg_weight = pos_train_edge.size(1) / neg_edge.size(1)
        for perm in PermIterator(adjmask.device, adjmask.shape[0], self.batch_size):
            self.optimizer.zero_grad()

            if self.opt['gcn']:
                h = self.model(self.data.x, self.data.adj_t.to_torch_sparse_coo_tensor())
            else:
                h = self.model(self.data.x, pos_encoding)
            
            edge = pos_train_edge[:, perm].to(self.device)
            
            pos_out = self.predictor(h[edge[0]], h[edge[1]])

            pos_loss = -torch.log(pos_out + 1e-15).mean()
            
            edge = neg_edge[:, perm]
            neg_out = self.predictor(h[edge[0]], h[edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            
            # TODO: Think about it
            # loss = pos_weight * pos_loss + neg_weight * neg_loss
            loss = pos_loss + neg_loss
            
            if self.opt['gcn'] == False:
                if self.model.odeblock.nreg > 0:
                    reg_states = tuple(torch.mean(rs) for rs in self.model.reg_states)
                    regularization_coeffs = self.model.regularization_coeffs

                    reg_loss = sum(
                        reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                    )
                    loss = loss + reg_loss

            # Update parameters
            if self.opt['gcn'] == False:
                self.model.fm.update(self.model.getNFE())
                self.model.resetNFE()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)

            self.optimizer.step()
            if self.opt['gcn'] == False:
                self.model.bm.update(self.model.getNFE())
                self.model.resetNFE()
            
            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

            # Update progress bar description with current loss
            # pbar.set_postfix({"Loss": loss.item()})

        return total_loss / total_examples
    
    @torch.no_grad()
    def test_epoch(self):
        self.model.eval()
        self.predictor.eval()
        
        if self.opt['dataset'].startswith('ogbl-'):
            evaluator = Evaluator(name=self.opt['dataset'])
        else:
            evaluator = Evaluator(name='ogbl-collab')
        
        if self.opt['gcn']:
            h = self.model(self.data.x, self.data.adj_t.to_torch_sparse_coo_tensor())
        else:
            h = self.model(self.data.x, self.pos_encoding)
        
        pos_train_edge = self.splits['train']['edge'].t().to(self.data.x.device)
        pos_valid_edge = self.splits['valid']['edge'].t().to(self.data.x.device)
        neg_valid_edge = self.splits['valid']['edge_neg'].t().to(self.data.x.device)
        pos_test_edge = self.splits['test']['edge'].t().to(self.data.x.device)
        neg_test_edge = self.splits['test']['edge_neg'].t().to(self.data.x.device)
        
        # pos_train_preds = []
        # for perm in DataLoader(range(pos_train_edge.size(0)), self.batch_size):
        #     edge = pos_train_edge[perm].t()
        #     pos_train_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        # pos_train_pred = torch.cat(pos_train_preds, dim=0)
        
        pos_train_pred = torch.cat([
        self.predictor(h[pos_train_edge[perm].t()[0]], h[pos_train_edge[perm].t()[1]]).squeeze().cpu()
        for perm in PermIterator(pos_train_edge.device,
                                 pos_train_edge.shape[0], self.batch_size, False)
        ],
                                dim=0)

        # pos_valid_preds = []
        # for perm in DataLoader(range(pos_valid_edge.size(0)), self.batch_size):
        #     edge = pos_valid_edge[perm].t()
        #     pos_valid_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        # pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
        pos_valid_pred = torch.cat([
        self.predictor(h[pos_valid_edge[perm][0]], h[pos_valid_edge[perm][1]]).squeeze().cpu()
        for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], self.batch_size, False)
        ],
                                dim=0)
        
        # neg_valid_preds = []
        # for perm in DataLoader(range(neg_valid_edge.size(0)), self.batch_size):
        #     edge = neg_valid_edge[perm].t()
        #     neg_valid_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        # neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        neg_valid_pred = torch.cat([
        self.predictor(h[neg_valid_edge[perm][0]], h[neg_valid_edge[perm][1]]).squeeze().cpu()
        for perm in PermIterator(neg_valid_edge.device,
                                 neg_valid_edge.shape[0], self.batch_size, False)
        ],
                                dim=0)
        
        # pos_test_preds = []
        # for perm in DataLoader(range(pos_test_edge.size(0)), self.batch_size):
        #     edge = pos_test_edge[perm].t()
        #     pos_test_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        # pos_test_pred = torch.cat(pos_test_preds, dim=0)

        pos_test_pred = torch.cat([
        self.predictor(h[pos_test_edge[perm][0]], h[pos_test_edge[perm][1]]).squeeze().cpu()
        for perm in PermIterator(pos_test_edge.device,
                                 pos_test_edge.shape[0], self.batch_size, False)
        ],
                                dim=0)
        
        # neg_test_preds = []
        # for perm in DataLoader(range(neg_test_edge.size(0)), self.batch_size):
        #     edge = neg_test_edge[perm].t()
        #     neg_test_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        # neg_test_pred = torch.cat(neg_test_preds, dim=0)

        neg_test_pred = torch.cat([
        self.predictor(h[neg_test_edge[perm][0]], h[neg_test_edge[perm][1]]).squeeze().cpu()
        for perm in PermIterator(neg_test_edge.device,
                                 neg_test_edge.shape[0], self.batch_size, False)
        ],
                                dim=0)
        
        results = {}
        for K in [1, 3, 10, 20, 50, 100]:
            evaluator.K = K
            train_hits = evaluator.eval({
                'y_pred_pos': pos_train_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
            valid_hits = evaluator.eval({
                'y_pred_pos': pos_valid_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
            test_hits = evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']

            results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
        
        print(f"Shape of pos_val_pred: {pos_test_pred.shape}")
        print(f"Shape of neg_val_pred: {neg_test_pred.shape}")

        result_mrr_test = evaluate_mrr(pos_test_pred, neg_test_pred, self.opt)  
        
        for name in ['MRR', 'mrr_hit1', 'mrr_hit3', 'mrr_hit10', 'mrr_hit20', 'mrr_hit50', 'mrr_hit100']:
            results[name] = (result_mrr_test[name])
        
        test_pred = torch.cat([pos_test_pred, neg_test_pred])
        test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                                torch.zeros(neg_test_pred.size(0), dtype=int)])
        
        result_auc_test = evaluate_auc(test_pred, test_true)
        for name in ['AUC', 'AP']:
            results[name] = (result_auc_test[name])

        result_acc_test = acc(pos_test_pred, neg_test_pred)
        results['ACC'] = (result_acc_test)
        
        return results
    
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
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            loss = self.train_epoch()
            
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            if epoch % 5 == 0:
                results = self.test_epoch()

                self.log_results(results, epoch)

                current_metric = results['Hits@100'][2]
                if current_metric > self.best_metric:
                    self.best_epoch = epoch
                    self.best_metric = current_metric
                    self.best_results = results

                print(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s")
                print(f"Current Best {current_metric}: {self.best_metric:.4f} (Epoch {self.best_epoch})")

        print(f"Training completed. Best {current_metric}: {self.best_metric:.4f} (Epoch {self.best_epoch})")
        return self.best_results

    def finalize(self):
        if self.best_results:
            print(f"Final Best Results:")
            for key, value in self.best_results.items():
                print(f"{key}: {value}")
        else:
            print("No results to finalize.")
