#!/bin/bash

# Define the dataset array
datasets=("Cora" "Citeseer" "Pubmed" "Photo" "Computers")

# Loop through each dataset
for data in "${datasets[@]}"; do
    echo "Running heuristic_plaintoid.py on dataset: $data"
    python heuristic_plaintoid.py --data_name "$data"

    # Check for errors
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing $data. Exiting."
        exit 1
    fi
done

echo "All tasks completed successfully."