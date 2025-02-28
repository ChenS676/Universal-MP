#!/bin/bash

datasets=("ogbl-collab" "ogbl-ddi" "ogbl-citation2" "ogbl-ppa")

for dataset in "${datasets[@]}"; do
    python heuristic_ogb.py --data_name "$dataset"
done
