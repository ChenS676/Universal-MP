- [x]  first formatten the best parameters for all datasets 
- [ ]  generate nodevec for all listed dataset ogbl-ddi, ppa, vessel, citation2, collab, Cora, Citeseer, Pubmed 
- [ ]  create link
- [ ]  check the input and data dimension 
- [ ]  TODO check the overlapping params betweena argparse and cfg_file for further analysis - use default value for ogbl-collab as test

- [ ] 1. debug on ogbl-collab, 
    - download new version it is runnable. german has done a good job....
    - 
- [ ] 2. debug on ogbl-ddi 
    - killed -> estimation is the hidden dimension is too high wrong
             -> check parameter integration method is only dopri5 for ddi  why? change back to dopri5
             -> PermIterator looks to be a bad dataloder
    - > identify as the overload of function torch.cat([]) -> change into list solve the problem 
- [ ] 3. debug on ppa 
    - it never ends -> 1 epoch several hours 

- [ ] 4. debug on vessel 
    - no default parameters 

- [ ] 5. citation2 
    - default parameter commented, no reason why
    - what to do

# How to choose hyperparameter for citation2, vessel 
    - I have an over-completed solution to calc the graph statistic with corresponding downsampling rate. 
    - While in the same time, just use
    - for them we set the beltrami as False first #TODO beltrami true 

Running Command:
```
python3 main_german.py --dataset ogbl-ppa --device 0 --no_early --beltrami 
python3 main_german.py --dataset ogbl-collab --device 0 --no_early --beltrami 
python3 main_german.py --dataset ogbl-ddi --device 0 --no_early --beltrami  
# take very long several hours for one epoch
python main_german.py  --dataset ogbl-vessel --device 0 --no_early --beltrami
```
