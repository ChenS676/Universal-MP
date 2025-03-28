#!/bin/bash

# Define the hyperparameters
INTER_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
INTRA_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
TOTAL_EDGES=(1000 2000 3000 4000 5000)

# Create a log file to monitor the process
LOG_FILE="process_log.txt"

# Use GNU Parallel to run the Python script in parallel and log output
parallel --jobs 4 python3 real_syn_automorphic.py --inter_ratio {1} --intra_ratio {2} --total_edges {3} >> $LOG_FILE 2>&1 ::: ${INTER_RATIOS[@]} ::: ${INTRA_RATIOS[@]} ::: ${TOTAL_EDGES[@]}


