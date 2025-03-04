## debug GCNs 
    - [ ] save result file has a bug: name of the model is lost change the name_tag
    - [ ] submit the job and report the result to table

## debug on ppa  UPGRADE    
    - it never ends -> 1 epoch up to 2 hours why? with slightly improvement considering assing common neighbors

## debug on vessel  UPGRADE
    - [ ] no default parameters solved promptly, reference to statistic  -> use ddi 
    - [ ] function_laplacian_diffusion has OOM memory when running `ax = self.sparse_multiply(x)`
            - reimplement it using sparse tensor reduce epoch training time from 2.5 mins to 0.5min, but somehow some metrics have the decay.

###  citation2 UPGRADE
    - default parameter commented, no reason why
    - what to do -  uncomment
    - finalize the implementation of trainer with slight change in test_epoch only by copying the script from  NCNC OverlapCommonNeighbor
    -
### Temporary task
    - 
### Paper writing, 
    - automatic table generation in new pipeline result - xlsx - script - latex table -overleaf

# Systematic Issue of High Complexity 

- [ ] measure the training and inference time of all gcns and save into txt
- [ ] measure the training and inference time of grands
- [ ] ### double check the evaluator of grand and the one from heart 
- [ ] provide table draft and fill the result in paper 

- [ ] Implement one random sampling method to deal with the long developement period 
- [x] Submit jobs to run ppa, vessel and citation2 for 4 epochs first, 
        - ppa epoch 1, 2, 3
        - vessel epoch 1, 2, 4
        - citation2 epoch 1, 2, 3

# How to choose hyperparameter for citation2, vessel 
    - I have an over-completed solution to calc the graph statistic with corresponding downsampling rate. 
    - While in the same time, just use parameter of ddi, it has until now the best performance.
    - for them we set the beltrami as False first #TODO beltrami true 
    - We indeed have no clue these statistic helps choosing a better hyper-parameters. 

# Report data statistic in paper draft
    Done: 
    Findings are: 
        - 1. citation network are more hierarchial Cora, Citesser
        - 2. ddi are very connected (function groups)
        - 3. vessel and citation2 are non-local, vessel is a mouse brain, citation2 is a collaboration network

## double check the evaluator of grand and the one from heart 

- [x]  first formatten the best parameters for all datasets 
- [ ]  generate nodevec for all listed dataset ogbl-ddi, ppa, vessel, citation2, collab, Cora, Citeseer, Pubmed 
- [ ]  create link
- [ ]  check the input and data dimension 
- [x]  TODO check the overlapping params betweena argparse and cfg_file for further analysis - use default value for ogbl-collab as test

- [x] 1. debug on ogbl-collab, 
    - download new version it is runnable. german has done a good job....
    - 
- [x] 2. debug on ogbl-ddi 
    - killed -> estimation is the hidden dimension is too high wrong
             -> check parameter integration method is only dopri5 for ddi  why? change back to dopri5
             -> PermIterator looks to be a bad dataloder
    - > identify as the overload of function torch.cat([]) -> change into list solve the problem 

- [x] 3. debug on ppa  UPGRADE
    - it never ends -> 1 epoch up to 2 hours why? with slightly improvement considering assing common neighbors 
    - slightly outperform

- [ ] 4. debug on vessel  UPGRADE
    - no default parameters solved promptly see below

- [ ] 5. citation2 UPGRADE
    - default parameter commented, no reason why 
    - what to do -  uncomment


Running Command:
```
python3 main_grand.py --dataset ogbl-ppa --device 0 --no_early --beltrami 
python3 main_grand.py --dataset ogbl-collab --device 0 --no_early --beltrami 
python3 main_grand.py --dataset ogbl-ddi --device 0 --no_early --beltrami  
# take very long several hours for one epoch
python main_grand.py  --dataset ogbl-vessel --device 0 --no_early --beltrami

python main_grand.py  --dataset Citeseer --device 0 --no_early --beltrami
python main_grand.py  --dataset Pubmed --device 0 --no_early --beltrami
python main_grand.py  --dataset Cora --device 0 --no_early --beltrami
```

CUDA_LAUNCH_BLOCKING=1 python allin_grand_ncnc.py --data_name ogbl-collab --device 0  --beltrami --predictor cn1

python allin_grand_ncnc.py   --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --predp 0.3 --gnndp 0.1  --probscale 2.5 --proboffset 6.0 --alpha 1.05  --gnnlr 0.0082 --prelr 0.0037  --batch_size 65536  --ln --lnnn --predictor cn1 --dataset collab  --epochs 100 --runs 2 --hidden_dim 128 --mplayers 3  --testbs 131072  --maskinput --use_valedges_as_input   --res  --use_xlin  --tailact 

python allin_grand_ncnc.py   
Here is the table formatted in Markdown:


| **Keyword**               | **Interpretation** |
|--------------------------|------------------|
| `use_valedges_as_input`  | Whether to add validation edges to the input adjacency matrix of GNN. |
| `epochs`                | Number of training epochs. |
| `dataset`               | Name of the dataset used. |
| `testbs`                | Batch size for testing. |
| `maskinput`             | Whether to remove target links. |
| `mplayers`              | Number of message passing layers in GNN. |
| `nnlayers`              | Number of MLP layers. |
| `hidden_dim`            | Hidden dimension size. |
| `ln`                    | Whether to use layer normalization in MPNN. |
| `lnnn`                  | Whether to use layer normalization in MLP. |
| `res`                   | Whether to use residual connections. |
| `jk`                    | Whether to use Jumping Knowledge connections. |
| `gnndp`                 | Dropout ratio for GNN. |
| `xdp`, `tdp`            | Dropout ratios for different parts of GNN. |
| `gnnedp`                | Edge dropout ratio for GNN. |
| `predp`                 | Dropout ratio for predictor. |
| `preedp`                | Edge dropout ratio for predictor. |
| `prelr`                 | Learning rate for predictor. |
| `beta`                  | A detailed hyperparameter (purpose unspecified). |
| `splitsize`             | Splitting operations inside the model for memory and speed optimization. |
| `probscale`, `proboffset`, `pt`, `learnpt` | Parameters for calibrating edge existence probability in NCNC. |
| `trndeg`                | Maximum number of sampled neighbors during training (-1 means no sampling). |
| `cndeg`                 | Maximum number of sampled common neighbors (generally not used). |
| `depth`                 | Number of completion steps in NCNC. |
| `save_gemb`             | Whether to save GNN-generated node embeddings. |
| `load`                  | Path to load precomputed node embeddings. |
| `loadmod`, `savemod`    | Whether to load or save trained models. |
| `savex`, `loadx`        | Whether to save or load trained node embeddings. |
| `use_xlin`              | (Unspecified function, likely a linear transformation on `x`). |
| `cnprob`                | Common neighbor probability (unused in experiments). |
| `mlp_num_layers`        | Number of layers in MLP. |
| `batch_size`            | Training batch size. |
| `gcn`                   | Whether to use GCN (Graph Convolutional Network). |
| `num_layers`            | Number of layers in GCN. |
| `runs`                  | Number of independent runs for experiments. |
| `eval_steps`            | Frequency of evaluation during training. |
| `predictor`             | Type of predictor used (e.g., NCN, NCNC). |
| `tstdeg`                | Maximum number of sampled neighbors during testing. |
| `tailact`               | Whether to use a specific tail activation function. |
| `twolayerlin`           | Whether to use a two-layer linear transformation. |
| `increasealpha`         | Whether to increase the alpha parameter. |
| `gnnlr`                 | Learning rate for GNN. |

You can copy and paste this into a Markdown editor or a GitHub README, and it will render properly. Let me know if you need any refinements!

`
 python grand_ncnc_tune.py   --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --predp 0.3 --gnndp 0.1  --probscale 2.5 --proboffset 6.0 --alpha 1.05  --gnnlr 0.0082 --prelr 0.0037  --batch_size 65536  --ln --lnnn --predictor cn1 --dataset collab  --runs 1 --hidden_dim 128 --mplayers 3  --testbs 131072  --maskinput --use_valedges_as_input   --res  --use_xlin  --tailact 
`
