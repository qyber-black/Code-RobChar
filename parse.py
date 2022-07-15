#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 19:52:18 2021

@author: irtazakhalid
"""

import argparse

def get_noise_analysis_args():
    """Get arguments needed in noise_analysis.py."""
    parser = argparse.ArgumentParser('Start collecting spin transition data.')

    add_common_args(parser)

    parser.add_argument('--algo_name',
                        type=str,
                        choices=('ppo', 'lbfgs', 'snob', 'nmplus'), 
                        default=None,
                        help='Name of the algo for which the statistics will be recorded.')
    parser.add_argument('--topo',
                        type=str,
                        default='chain',
                        choices=('chain', 'ring'),
                        help='topology of the system: limited to 2 choicesfor now.')
    parser.add_argument('--num_controllers',
                        type=int,
                        default=1000,
                        help='number of controllers to be found.')
    parser.add_argument('--fid_threshold',
                        type=float,
                        default=0.0,
                        help='fidelity threshold of the controller.')
    parser.add_argument('--max_noise',
                        type=float,
                        default=0.1,
                        help='maximum value of added noise.')
    parser.add_argument('--noise_res',
                        type=int,
                        default=11,
                        help='noise resolution/steps: e.g. 0, 0.1, 0.2, ..., 0.1 is 11 steps.')
    parser.add_argument('--fid_noisy',
                        type=bool,
                        default=False,
                        help='coarse grained fidelity with default `draws`=100 shots.')
    parser.add_argument('--ham_noisy',
                        type=bool,
                        default=True,
                        help='add MC noise to Hamiltonian during training.')
    parser.add_argument('--draws',
                        type=int,
                        default=100,
                        help='# of coarse grained shots.')
    parser.add_argument('--respawn_from_checkpoint',
                        type=bool,
                        default=False,
                        help='Continue where you left of during some prior data collection.')
    
    parser.add_argument('--verbose',
                        type=bool,
                        default=False,
                        help='Debugging verbosity to print stuff for qualitative okay-signalling')
    
    parser.add_argument('--run_until_told_to_stop',
                        type=bool,
                        default=False,
                        help='homogenize iteration count')
    
    parser.add_argument('--run_until_completion_its',
                    type=int,
                    default=600000,
                    help='homogenize-able iteration count')
    parser.add_argument('--run_stoch_arimscale', type=bool, 
                        default=False, 
                        help='run an asymptotic test with multiple non-stochastic hamiltonians per objetive function call.')
    
    parser.add_argument('--records_update_rate', type=int, 
                        default=100000, 
                        help='update rate for checkpointing controllers')
    parser.add_argument('--use_fixed_ham', type=bool, 
                        default=False, 
                        help='update rate for checkpointing controllers')
    parser.add_argument('--fixed_ham_train_size', type=int, 
                        default=100, 
                        help='number of hamiltonians to average over in 1 objective function call for non stochastic sampling')
    args = parser.parse_args()
    
    # if not args.algo_name:
    #     raise argparse.ArgumentError
    return args

def add_common_args(parser):
    "args common to multiple scripts that need to be run"
    
    parser.add_argument('--exp_name',
                        type=str,
                        default='pipeline_nmplus2')
    parser.add_argument('--nspin',
                        type=int,
                        default=5,
                        help='Spin size/len of the qc system.')
    parser.add_argument('--inspin',
                        type=int,
                        default=0,
                        help='Input spin')
    parser.add_argument('--outspin',
                        type=int,
                        default=2,
                        help='Output spin')

def get_mcsim_args():
    """Get arguments needed in mcsim.py."""
    parser = argparse.ArgumentParser('Run a cachable Monte Carlo simulation')

    add_common_args(parser)

    parser.add_argument('--bootreps',
                        type=int,
                        default=100,
                        help='Number of bootstrap repititions.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=None,
                        help='Number of workers during the parallel bootstrap sampling step.')
    parser.add_argument('--training_noise',
                        type=str,
                        default='0.1',
                        help='Relevant if algo was trained on noise else pass')
    parser.add_argument('--parallel',
                        type=bool,
                        default=False,
                        help='Parallelize the bootstrapping for loop')
    parser.add_argument('--mc_max_noise',
                        type=float,
                        default=0.1,
                        help='Maximum simulation noise')
    parser.add_argument('--mc_noise_res',
                        type=float,
                        default=11,
                        help='MC noise resolution/steps: e.g. 0, 0.1, 0.2, ..., 0.1 is 11 steps.')
    
    args = parser.parse_args()

    return args



