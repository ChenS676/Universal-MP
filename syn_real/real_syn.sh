

#!/bin/bash

# Define the hyperparameters
INTER_RATIOS=(0.7)
INTRA_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
TOTAL_EDGES=(1000 2000 3000 4000 5000)

# Create a log file to monitor the process
LOG_FILE="process_log.txt"
> $LOG_FILE  # Clear the log file if it exists

# Loop over the combinations of hyperparameters
for inter_ratio in "${INTER_RATIOS[@]}"; do
    for intra_ratio in "${INTRA_RATIOS[@]}"; do
        for total_edges in "${TOTAL_EDGES[@]}"; do
            # Run the Python script with the current hyperparameters
            echo "Running with inter_ratio=$inter_ratio, intra_ratio=$intra_ratio, total_edges=$total_edges" | tee -a $LOG_FILE
            python3 real_syn_automorphic.py --inter_ratio $inter_ratio --intra_ratio $intra_ratio --total_edges $total_edges >> $LOG_FILE 2>&1
            echo "Finished with inter_ratio=$inter_ratio, intra_ratio=$intra_ratio, total_edges=$total_edges" | tee -a $LOG_FILE
        done
    done
done