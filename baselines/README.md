## Baselines 

#TODO 
pretest trainer for ogb dataset to measure the difference with reported performance in paper 
implement one random sampling method to reduce development period for ogb dataset except ddi 

integrate grand_lp 
transfer differnet split methods 
create one result 
start some experiment with synthetic graphs
implement one script to fill the latex table automatically 
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
``` tested
python  main_gnn.py  --data_name Citeseer  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 --debug 
```


Pubmed
```
python  main_gnn.py  --data_name Pubmed  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 --debug
```

ogb-datasets:  ogbl-ppa, ddi, vessel and ogbl-citation2, ogbl-collab, 
```
python  ogb_gnn.py  --data_name ppa  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024  --random_sampling
python  ogb_gnn.py  --data_name collab  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024  --random_sampling
python  ogb_gnn.py  --data_name ddi  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024  --random_sampling
python  ogb_gnn.py  --data_name citation2  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024  --random_sampling
python  ogb_gnn.py  --data_name vessel  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024  --random_sampling

```

```
 python  ogb_gnn.py  --data_name ogbl-collab  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 
```
```
 python  ogb_gnn.py  --data_name ogbl-citation2  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 
```

ogbl-ddi
```
python ddi_gnn.py --data_name ogbl-ddi --gnn_model GCN  --lr 0.01 --dropout 0.5  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 7 --eval_steps 1 --kill_cnt 100 --batch_size 65536 --debug
```

```
python ogb_gnn.py --data_name ogbl-ddi --gnn_model GCN  --lr 0.01 --dropout 0.5  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 200 --eval_steps 1 --kill_cnt 100 --batch_size 65536 --name_tag ddi_ogb_trainer --runs 2
```


To reduce development period, we use a downsampling ratio documented as follows: 
To ensure all datasets have a similar number of edges as **ogbl-ddi** (1,334,889 edges), we can compute the **sampling ratio** for each dataset as:  

\[
\text{sampling\_ratio} = \frac{\text{Target Number of Edges (ogbl-ddi)}}{\text{Original Number of Edges}}
\]

Now, let's compute the sampling ratios:  

| Dataset           | Original # Edges | Sampling Ratio |
|------------------|----------------|----------------|
| **ogbl-ppa**      | 30,326,273      | **0.0440** |
| **ogbl-collab**   | 1,285,465       | **1.0386** ≈ 1.0 (keep all) |
| **ogbl-ddi**      | 1,334,889       | **1.0000** |
| **ogbl-citation2** | 30,561,187     | **0.0437** |
| **ogbl-wikikg2**  | 17,137,181      | **0.0779** |
| **ogbl-biokg**    | 5,088,434       | **0.2623** |
| **ogbl-vessel**   | 5,345,897       | **0.2497** |

Thus, the recommended **sampling ratios** are:  

```python
sampling_ratio = {
    "ogbl-ppa": 0.044,
    "ogbl-collab": 1.0,  # No downsampling needed
    "ogbl-ddi": 1.0,  # Reference dataset
    "ogbl-citation2": 0.044,
    "ogbl-wikikg2": 0.078,
    "ogbl-biokg": 0.262,
    "ogbl-vessel": 0.25
}
```

This ensures that all datasets are **downsampled to around 1.33M edges**, making them more comparable. 