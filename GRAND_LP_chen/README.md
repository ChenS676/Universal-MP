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

python allin_grand_ncnc.py   --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --predp 0.3 --gnndp 0.1  --probscale 2.5 --proboffset 6.0 --alpha 1.05  --gnnlr 0.0082 --prelr 0.0037  --batch_size 65536  --ln --lnnn --predictor cn1 --dataset collab  --epochs 100 --runs 2 --hidden_dim 128 --mplayers 1  --testbs 131072  --maskinput --use_valedges_as_input   --res  --use_xlin  --tailact 


with this mail I would like to submit my code to you:
my latest implementation is /hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/GRAND_LP_chen/allin_grand_ncnc.py
my modifed implementation of your grand is /hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/GRAND_LP_chen/allin_grand_ori.py

allin_old is a backup file (ignore)

I have several best params files, to balance the human readability best_params.py is required.

main_grand.py is your original implementation. 
model.py NeighborOverlap.py includes necessary modules from NCNC. 

The useful commands are:

`
CUDA_LAUNCH_BLOCKING=1 python allin_grand_ncnc.py --data_name ogbl-collab --device 0  --beltrami --predictor cn1

`

My current unusual situation is that performance does not change with different hyperparameters. This strongly suggests a possible bug where the parameters are not actually being updated according to the configuration. Therefore, the first step should be a quick check to confirm whether the parameters are properly passed and applied.

After that, other possible explanations include:
	1.	The hyperparameters may not be relevant to performance (though they should be).
	2.	The number of epochs may be too small for the model to converge, preventing noticeable differences.
	3.	The internal mechanism might be too complex, requiring substantial effort to analyze. The model involves two modalities: one based on a diffusion mechanism and another relying on structural features, such as common neighbors. These two components may exhibit vastly different behaviors, necessitating distinct hyperparameter ranges. This is similar to transfer learning—where earlier layers capture general features and should be fine-tuned with a small learning rate, whereas later layers require a higher learning rate to optimize performance on a custom dataset.

I have implemented allin_grand_ncnc.py along with a README.md file, where you can tune parameters and find descriptions for each of them.

Steps to Proceed
	1.	Visualize the model structure in a diagram and ensure you fully understand it so that we are aligned.
	2.	When something doesn’t change, your general approach should be:
	•	First, eliminate basic errors, such as failing to pass the parameters correctly.
	3.	Force a change by trying extreme learning rates, e.g., 0.1, 1, and 0.000001. Any proper implementation should exhibit noticeable differences. If not, thoroughly investigate how parameters are being passed.
	4.	If no basic implementation errors are found and the performance stagnation is real, the key question is: why?
	•	This is where your ability to apply the knowledge you’ve gained comes into play. You need to re-evaluate the algorithmic framework and list all possible causes.
	•	Consider this as a technical interview question: If your loss function remains unchanged, list ten possible reasons and send them to me in a message.
	5.	Systematically rule out each possibility through careful testing and debugging.

Strengthening Your Problem-Solving Approach
To enhance your problem-solving skills, I highly recommend reading the following documentation on best practices for academic implementation:
Implementation Guidelines

Make sure to read every line carefully instead of repeatedly generating ideas with limited depth.

A Note on Your Approach
From your personal notes, I need to emphasize something important. Simply running a script and waiting the whole day for a temporary result is not an effective way to tackle this issue.

At this stage—being a final-year researcher with one year of academic experience—you should not rely on passive trial and error. Instead, you must develop a clear strategy based on a deep understanding of what is happening inside the model.

This is generally referred to as problem-solving skills. It requires a comprehensive grasp of strategy, tactics, and details, much like career planning—spanning months, years, or even a decade. The real challenge is ensuring that each decision you make today aligns with your long-term objectives, even amid uncertainty.

A Practical Debugging Strategy
One effective way to solve such problems is to first write down your thought process—as I did in the first paragraph—then systematically execute each step one by one.

Research is a continuous process of decision-making, and you need a structured approach to ensure your decisions are logical and methodical.