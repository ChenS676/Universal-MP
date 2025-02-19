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

- [x] 3. debug on ppa 
    - it never ends -> 1 epoch several hours 

- [ ] 4. debug on vessel 
    - no default parameters solved promptly

- [ ] 5. citation2 
    - default parameter commented, no reason why
    - what to do -  uncomment

# Systematic Issue of High Complexity 

- [ ] measure the training and inference time of all gcns and save into txt
- [ ] measure the training and inference time of grands
- [ ] double check the evaluator of grand and the one from heart 
- [ ] provide table draft and fill the result in paper 

- [ ] Implement one random sampling method to deal with the long developement period 


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


Running Command:
```
python3 main_german.py --dataset ogbl-ppa --device 0 --no_early --beltrami 
python3 main_german.py --dataset ogbl-collab --device 0 --no_early --beltrami 
python3 main_german.py --dataset ogbl-ddi --device 0 --no_early --beltrami  
# take very long several hours for one epoch
python main_german.py  --dataset ogbl-vessel --device 0 --no_early --beltrami
```
