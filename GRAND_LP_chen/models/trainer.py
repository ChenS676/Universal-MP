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
        
        #DEBUG CHECK THE CODE DIMENSION OF THE DATA
       
        pos_encoding = self.pos_encoding.to(self.model.device) if self.pos_encoding is not None else None
        pos_train_edge = self.splits['train']['edge'].to(self.data.x.device)
        neg_train_edge = negative_sampling(
            self.data.edge_index.to(pos_train_edge.device),
            num_nodes=self.data.num_nodes,
            num_neg_samples=pos_train_edge.size(0)
        ).t().to(self.data.x.device)

        total_loss = total_examples = 0
        indices = torch.randperm(pos_train_edge.size(0), device=pos_train_edge.device)

        for start in tqdm(range(0, pos_train_edge.size(0), self.batch_size)):
            
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

            # Germa 's update Update parameters
            # self.model.fm.update(self.model.getNFE())
            # self.model.resetNFE()
            # loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)

            # self.optimizer.step()
            # self.model.bm.update(self.model.getNFE())
            # self.model.resetNFE()
            ######################## original code
            
            self.model.fm.update(self.model.getNFE())
            self.model.resetNFE()
            loss.backward()
            self.optimizer.step()
            self.model.bm.update(self.model.getNFE())
            self.model.resetNFE()
            #######################
            
        return total_loss / total_examples
    
    @torch.no_grad()
    def test_epoch(self):
        self.model.eval()
        self.predictor.eval()
        
        if self.opt['dataset'].startswith('ogbl-'):
            evaluator = Evaluator(name=self.opt['dataset'])
        else:
            evaluator = Evaluator(name='ogbl-collab')
        
        h = self.model(self.data.x, self.pos_encoding)
        
        pos_train_edge = self.splits['train']['edge'].to(self.data.x.device)
        pos_valid_edge = self.splits['valid']['edge'].to(self.data.x.device)
        neg_valid_edge = self.splits['valid']['edge_neg'].to(self.data.x.device)
        pos_test_edge = self.splits['test']['edge'].to(self.data.x.device)
        neg_test_edge = self.splits['test']['edge_neg'].to(self.data.x.device)
        
        predict = []
        for perm in PermIterator(pos_train_edge.device,
                                 pos_train_edge.shape[0], self.batch_size, False):
            predict.append(self.predictor(h[pos_train_edge[perm, 0]], h[pos_train_edge[perm, 1]]).squeeze().cpu().tolist()[0])
        pos_train_pred = torch.Tensor(predict)


        predict = []
        for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], self.batch_size, False):
            predict.append(self.predictor(h[pos_valid_edge[perm, 0]], h[pos_valid_edge[perm, 1]]).squeeze().cpu().tolist()[0])
        pos_valid_pred = torch.Tensor(predict)


        predict = []
        for perm in PermIterator(neg_valid_edge.device,
                                 neg_valid_edge.shape[0], self.batch_size, False):
            predict.append(self.predictor(h[neg_valid_edge[perm, 0]], h[neg_valid_edge[perm, 1]]).squeeze().cpu().tolist()[0])
        neg_valid_pred = torch.Tensor(predict)

        predict = []
        for perm in PermIterator(pos_test_edge.device,
                                 pos_test_edge.shape[0], self.batch_size, False):
            predict.append(self.predictor(h[pos_test_edge[perm, 0]], h[pos_test_edge[perm, 1]]).squeeze().cpu().tolist()[0])
        pos_test_pred = torch.Tensor(predict)

        predict = []
        for perm in PermIterator(neg_test_edge.device,
                                 neg_test_edge.shape[0], self.batch_size, False):
            predict.append(self.predictor(h[neg_test_edge[perm, 0]], h[neg_test_edge[perm, 1]]).squeeze().cpu().tolist()[0])
        neg_test_pred = torch.Tensor(predict)

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
    
        result_mrr_test = evaluate_mrr(pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1), self.opt)  
        
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
