import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import time
from tqdm import tqdm

from metrics.metrics import *
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader

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

    @torch.no_grad()
    def test_epoch_OGB(self):
        self.model.eval()
        self.predictor.eval()

        evaluator = Evaluator(name=self.opt['dataset'])
        
        if self.opt['gcn']:
            h = self.model(self.data.x, self.data.adj_t.to_torch_sparse_coo_tensor())
        else:
            h = self.model(self.data.x, self.pos_encoding)
        
        pos_train_edge = self.splits['train']['pos_edge_label_index'].to(self.data.x.device)
        neg_train_edge = self.splits['train']['neg_edge_label_index'].to(self.data.x.device)
        pos_valid_edge = self.splits['valid']['pos_edge_label_index'].to(self.data.x.device)
        neg_valid_edge = self.splits['valid']['neg_edge_label_index'].to(self.data.x.device)
        pos_test_edge = self.splits['test']['pos_edge_label_index'].to(self.data.x.device)
        neg_test_edge = self.splits['test']['neg_edge_label_index'].to(self.data.x.device)
        
        pos_train_preds = []
        for perm in DataLoader(range(pos_train_edge.size(1)), self.batch_size):
            edge = pos_train_edge[:, perm]
            pos_train_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_train_pred = torch.cat(pos_train_preds, dim=0)

        neg_train_preds = []
        for perm in DataLoader(range(neg_train_edge.size(1)), self.batch_size):
            edge = neg_train_edge[:, perm]
            neg_train_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_train_pred = torch.cat(neg_train_preds, dim=0)

        pos_valid_preds = []
        for perm in DataLoader(range(pos_valid_edge.size(1)), self.batch_size):
            edge = pos_valid_edge[:, perm]
            pos_valid_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

        neg_valid_preds = []
        for perm in DataLoader(range(neg_valid_edge.size(1)), self.batch_size):
            edge = neg_valid_edge[:, perm]
            neg_valid_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(1)), self.batch_size):
            edge = pos_test_edge[:, perm]
            pos_test_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(1)), self.batch_size):
            edge = neg_test_edge[:, perm]
            neg_test_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

        results = {}
        for K in [1, 3, 10, 20, 50, 100]:
            evaluator.K = K
            train_hits = evaluator.eval({
                'y_pred_pos': pos_train_pred,
                'y_pred_neg': neg_train_pred,
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
  
  
    def train_epoch_OGB(self):
        self.predictor.train()
        self.model.train()

        pos_encoding = self.pos_encoding.to(self.model.device) if self.pos_encoding is not None else None
        
        pos_train_edge = self.splits['train']['pos_edge_label_index'].to(self.data.x.device)
        neg_train_edge = self.splits['train']['neg_edge_label_index'].to(self.data.x.device)
        
        total_loss = total_examples = 0
        data_loader = DataLoader(range(pos_train_edge.size(1)), self.batch_size, shuffle=True)
        
        with tqdm(data_loader, desc="Training Progress", unit="batch") as pbar:
            for perm in pbar:
                self.optimizer.zero_grad()
                
                if self.opt['gcn']:
                    h = self.model(self.data.x, self.data.adj_t.to_torch_sparse_coo_tensor())
                else:
                    h = self.model(self.data.x, pos_encoding)
                edge = pos_train_edge[:, perm]
                pos_out = self.predictor(h[edge[0]], h[edge[1]])
    
                pos_loss = -torch.log(pos_out + 1e-15).mean()
                
                edge = neg_train_edge[:, perm]
                
                neg_out = self.predictor(h[edge[0]], h[edge[1]])
                neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
                
                loss = pos_loss + neg_loss
                
                if self.opt['gcn'] == False:
                    if self.model.odeblock.nreg > 0:
                        reg_states = tuple(torch.mean(rs) for rs in self.model.reg_states)
                        regularization_coeffs = self.model.regularization_coeffs

                        reg_loss = sum(
                            reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                        )
                        print('LOSS and REG LOSS: ', loss, reg_loss)
                        loss = loss + reg_loss
                        print('NEW LOSS: ', loss)

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
                pbar.set_postfix({"Loss": loss.item()})
        
        return total_loss / total_examples


    def _train(self):
        self.predictor.train()
        self.model.train()
        
        pos_encoding = self.pos_encoding.to(self.model.device) if self.pos_encoding is not None else None
        
        pos_train_edge = self.splits['train']['edge'].to(self.data.x.device)
        
        total_loss = total_examples = 0
        data_loader = DataLoader(range(pos_train_edge.size(0)), self.batch_size, shuffle=True)
        
        with tqdm(data_loader, desc="Training Progress", unit="batch") as pbar:
            for perm in pbar:
                self.optimizer.zero_grad()
                if self.opt['gcn']:
                    h = self.model(self.data.x, self.data.edge_index)
                else:
                    h = self.model(self.data.x, pos_encoding)
                
                edge = pos_train_edge[perm].t()
                pos_out = self.predictor(h[edge[0]], h[edge[1]])
    
                pos_loss = -torch.log(pos_out + 1e-15).mean()
                
                edge = torch.randint(0, self.data.num_nodes, edge.size(), dtype=torch.long,
                    device=h.device)
                
                neg_out = self.predictor(h[edge[0]], h[edge[1]])
                neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
                
                loss = pos_loss + neg_loss
                
                if self.opt['gcn'] == False:
                    if self.model.odeblock.nreg > 0:
                        reg_states = tuple(torch.mean(rs) for rs in self.model.reg_states)
                        regularization_coeffs = self.model.regularization_coeffs

                        reg_loss = sum(
                            reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                        )
                        print('LOSS and REG LOSS: ', loss, reg_loss)
                        loss = loss + reg_loss
                        print('NEW LOSS: ', loss)

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
                pbar.set_postfix({"Loss": loss.item()})
        
        return total_loss / total_examples
  
  
    @torch.no_grad()
    def test_epoch(self):
        self.model.eval()
        self.predictor.eval()
        
        if self.opt['dataset'].startswith('ogbl-'):
            evaluator = Evaluator(name=self.opt['dataset'])
        else: # keep the same as ncnc
            evaluator = Evaluator(name='ogbl-ppa')
        
        if self.opt['gcn']:
            h = self.model(self.data.x, self.data.adj_t.to_torch_sparse_coo_tensor())
        else:
            h = self.model(self.data.x, self.pos_encoding)
        
        pos_train_edge = self.splits['train']['edge'].cpu()
        pos_valid_edge = self.splits['valid']['edge'].cpu()
        neg_valid_edge = self.splits['valid']['edge_neg'].cpu()
        pos_test_edge = self.splits['test']['edge'].cpu()
        neg_test_edge = self.splits['test']['edge_neg'].cpu()
        
        pos_train_preds = []
        for perm in DataLoader(range(pos_train_edge.size(0)), self.batch_size):
            edge = pos_train_edge[perm].t()
            pos_train_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_train_pred = torch.cat(pos_train_preds, dim=0)

        pos_valid_preds = []
        for perm in DataLoader(range(pos_valid_edge.size(0)), self.batch_size):
            edge = pos_valid_edge[perm].t()
            pos_valid_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

        neg_valid_preds = []
        for perm in DataLoader(range(neg_valid_edge.size(0)), self.batch_size):
            edge = neg_valid_edge[perm].t()
            neg_valid_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), self.batch_size):
            edge = pos_test_edge[perm].t()
            pos_test_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), self.batch_size):
            edge = neg_test_edge[perm].t()
            neg_test_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

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

        result_mrr_test = evaluate_mrr(pos_test_pred, neg_test_pred)  
        
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


    def train(self): # Cora, Citeseer, Pubmed, Computer, Photo
        print(f"Starting training for {self.epochs} epochs...")
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            loss = self.train_epoch()
            
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            if epoch % 5 == 0:
                results = self.test_epoch()

                # TODO log system should be the same as before.
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
