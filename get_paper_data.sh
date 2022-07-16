#!/bin/bash

# get individual controllers
python noise_analysis.py --nspin 4 --outspin 2 \
--fid_threshold 0.1 --run_until_told_to_stop True \
--run_until_completion_its 1000000 --num_controllers 1000 \

python noise_analysis.py --nspin 5 --outspin 2 \
--fid_threshold 0.1 --run_until_told_to_stop True \
--run_until_completion_its 1000000 --num_controllers 1000

python noise_analysis.py --nspin 5 --outspin 4 \
--fid_threshold 0.1 --run_until_told_to_stop True \
--run_until_completion_its 1000000 --num_controllers 1000

python noise_analysis.py --nspin 6 --outspin 3 \
--fid_threshold 0.1 --run_until_told_to_stop True \
--run_until_completion_its 1000000 --num_controllers 1000

python noise_analysis.py --nspin 6 --outspin 5 \
--fid_threshold 0.1 --run_until_told_to_stop True \
--run_until_completion_its 1000000 --num_controllers 1000

python noise_analysis.py --nspin 7 --outspin 3 \
--fid_threshold 0.1 --run_until_told_to_stop True \
--run_until_completion_its 1000000 --num_controllers 1000

python noise_analysis.py --nspin 7 --outspin 6 \
--fid_threshold 0.1 --run_until_told_to_stop True \
--run_until_completion_its 1000000 --num_controllers 1000

# run the arim scaling experiments
# run stochastic ham experiments
python run_stoch_experiments.py --nspin 5 --outspin 2  \
--run_until_told_to_stop True --run_until_completion_its 40000000 \
--num_controllers 100 --records_update_rate 100000 --fid_threshold 0.0 \
--ham_noisy True

# run nonstochastic ham experiments
python run_stoch_experiments.py --nspin 5 --outspin 2  \
--run_until_told_to_stop True --run_until_completion_its 40000000 \
--num_controllers 100 --records_update_rate 100000 --fid_threshold 0.0 \
--use_fixed_ham True --fixed_ham_train_size 100


