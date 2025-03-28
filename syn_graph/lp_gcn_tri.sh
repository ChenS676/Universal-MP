for p in $(seq 0.1 0.01 0.2); do 
    p_formatted=$(printf "%.3f" $p)
    python lp_gcn_tri.py --p $p --gnn_model GIN
done
