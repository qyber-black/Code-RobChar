from noise_analysis import Experiment, get_noise_analysis_args
import numpy as np

def run_arim_scaling_experiments():
    args = get_noise_analysis_args()
    if args.use_fixed_ham:
        noises_for_paper = np.array([0.01, 0.05, 0.1])
    else:
        # adding the zero noise case for some reference algos later
        noises_for_paper = np.array([0.0, 0.01, 0.05, 0.1])

    exp = Experiment("pipeline_nonstoch_experiments_others_comp", 
                     Nspin=args.nspin, 
                     inspin=args.inspin,
                     outspin=args.outspin,
                     fid_threshold=args.fid_threshold,
                     fid_noisy=args.fid_noisy,
                     ham_noisy=args.ham_noisy,
                     noises=noises_for_paper, 
                     respawn_from_checkpoint=args.respawn_from_checkpoint, 
                     verbose=args.verbose,
                     run_until_told_to_stop=True,
                     run_until_completion_its=args.run_until_completion_its,
                     runs=args.num_controllers, 
                     records_update_rate=args.records_update_rate, # triggers multcontroller checkpointing
                     use_fixed_ham=args.use_fixed_ham, 
                     opt_train_size=args.fixed_ham_train_size)
    exp.singlerun_ccollector_nstoch_sampling()


if __name__ == '__main__':
    run_arim_scaling_experiments()