#!/bin/bash

datasets=("Cora" "Citeseer" "Pubmed" "Computers" "Photo")

for dataset in "${datasets[@]}"; do
    echo "Running experiment for dataset: $dataset"
    python heuristic_run.py --data_name "$dataset"
    echo "Completed experiment for dataset: $dataset"
done

echo "All experiments completed."
