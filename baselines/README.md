## Baselines 

```
cd /hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/baselines
python3 main.py  --data Cora --device cuda:0 --epochs 300 --model GCN_Variant 
```


```
python  main_gnn.py  --data_name Cora  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024
```