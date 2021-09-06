python compare_difference.py -i "data/sampled_data_row.pickle" -o "results"
python compare_cw_attacks.py -i "data/sampled_data_grid.pickle" -o "results"
python compare_levels.py -i "data/sampled_data_row.pickle" -o "results"
python main.py -p "experiment.json" -i "data/sampled_data_grid.pickle" -o "results"