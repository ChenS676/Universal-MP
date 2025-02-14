## Baselines 

#TODO 
pretest trainer for ogb dataset to measure the difference with reported performance in paper 
implement one random sampling method to reduce development period for ogb
integrate grand_lp 
transfer differnet split methods 
start some experiment with synthetic graphs
```
cd /hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/baselines
python3 main.py  --data Cora --device cuda:0 --epochs 300 --model GCN_Variant 
```

Cora
```
python  main_gnn.py  --data_name Cora  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024
```


Citeseer 
```
python  main_gnn.py  --data_name Citeseer  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 
```


Pubmed
```
python  main_gnn.py  --data_name Pubmed  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024
```

ogb-datasets:  ogbl-collab, ogbl-ppa, and ogbl-citation2
```
python  ogb_gnn.py  --data_name ogbl-ppa  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024
```

```
 python  ogb_gnn.py  --data_name ogbl-collab  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 
```
```
 python  ogb_gnn.py  --data_name ogbl-citation2  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 
```

ogbl-ddi
```
python ddi_gnn.py --data_name ogbl-ddi --gnn_model GCN  --lr 0.01 --dropout 0.5  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536 
```

````
python ogb_gnn.py --data_name ogbl-ddi --gnn_model GCN  --lr 0.01 --dropout 0.5  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536 --name_tag ddi_ogb_trainer --runs 2
````
