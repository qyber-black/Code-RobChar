#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:04:31 2021

@author: irtazakhalid

adapted from the openai spinningup github repo
"""


import numpy as np
import torch
from torch.optim import Adam
import ppo_core as core
from RLreinforceXXchain_actionedtime import Environment
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import logging
import json
import time as tt
from qnewton import LBFGS
from wd_sortof_fast_implementation import wd_from_ideal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



class PPO_en(object):
    def __init__(self, nspin=3, in_spin=0, out_spin=2, bmin=-10, bmax=10, 
                     max_time=30, repeats=100, fid_threshold=0.98, timestep_res=0.5, 
                     epochs=10000, rollouts=4000, log=False, ac_kwargs=dict(), 
                     save=False, timeout=1800, verbose=False, fid_noisy=False,
                     ham_noisy=False, draws=10, adaptive=False, adp_tol=0.05, 
                     testing=False, noise=0.05, transfer_learning=False, 
                     run_until_told_to_stop: bool = False,
                     run_until_completion_its: int = 6e5,
                     landscape_exploration: bool = False,
                     save_topc: int = 1000,
                     train_pi_iters=200, train_v_iters=200, clip_ratio=0.2,
                     lam=0.97, gamma=0.99,
                     pi_lr=3e-3,
                     vf_lr=1e-3, use_fixed_ham: bool =False, 
                     opt_train_size: int = 100, 
                     records_update_rate:float=None):
        
        #### hyperparameters to be optimized
        self.lam=lam
        self.gamma=gamma
        self.train_pi_iters=train_pi_iters 
        self.train_v_iters=train_v_iters
        self.clip_ratio = clip_ratio
        self.pi_lr=pi_lr
        self.vf_lr=vf_lr
        self.landscape_exploration = landscape_exploration
        self.save_topc = save_topc
            
        self.nspin = nspin
        self.In = in_spin
        self.Out = out_spin
        self.Tmin = 0
        self.Tmax = max_time
        self.Bmin = bmin
        self.Bmax = bmax
        self.repeats = repeats
        self.timestep_res = timestep_res 
        self.fid_noisy=fid_noisy
        self.draws=draws
        self.ham_noisy=ham_noisy
        self.verbose = verbose
        self.timeout = timeout
        self.adaptive = adaptive
        self.adp_func_calls_increment = self.draws
        self.adp_var_tol = adp_tol
        self.use_fixed_ham = use_fixed_ham
        self.train_size = opt_train_size

        self.env = Environment(nspin, self.In, self.Out, 
                               np.zeros(nspin), max_time=self.Tmax, 
                               bmin=self.Bmin, bmax=self.Bmax,
                               fid_noisy=self.fid_noisy, draws=self.draws,
                               ham_noisy=self.ham_noisy, noise=noise,
                               transfer_learning=transfer_learning,
                               use_fixed_ham=self.use_fixed_ham,
                               opt_train_size=self.train_size
                               )


        # Create actor-critic module
        self.ac = core.MLPActorCritic(self.nspin+1, self.nspin+1, **ac_kwargs) 
        self.epochs = epochs
        self.rollouts = rollouts
        self.repeats = repeats
        self.fid_threshold = fid_threshold
        self.total_rewards = []
        
        self.record = {"time_to_get_fid":None, "func_calls":None, "iterations":None, 
                       "repeats":None, "best_fid":None, "controller":None }
        self.filename = self.filename_generator()
        if log:
            self.logger = logging.basicConfig(filename=self.filename, encoding='utf-8', level=logging.DEBUG)
        self.save = save
        self.testing = testing
        
        self.Monte_env = LBFGS(nspin, self.In, self.Out, noise=noise)
        self.run_until_told_to_stop = run_until_told_to_stop
        self.run_until_completion_its=run_until_completion_its
        
        self.records = {}
        self.records_update_rate = records_update_rate # every 1e5 function calls
        self.update_counter=0

    def record_collector(self, fcalls, controller_dict):
        if fcalls>self.update_counter:
            self.records[fcalls] = controller_dict
            checkpoints = int(self.run_until_completion_its/self.records_update_rate)
            curr = int(fcalls/self.records_update_rate)
            if self.verbose:
                print(self.records)
                print(f"saving controller_dict {curr}/{checkpoints}")
            self.update_counter += self.records_update_rate
            
    def save_record(self):
        json.dump( self.record, open(self.filename, 'w'))
        
    def read_record(self):
        return json.load(open(self.filename))
        
    def filename_generator(self):
        return "ppo_en_record_s{}_o{}_t{}_b{}_r_{}.json".format(
            self.nspin, self.Out, self.Tmax, self.Bmax,self.repeats)
    
    def find_min_fid_index(self,controller_list):
        "in non-increasing order of fid"
        c2fid = lambda c: self.Monte_env.fidelity_ss(c)
        fids = list(map(c2fid, controller_list))
        # print(fids)
        return np.argmin(fids)
        
    

    def run(self, seed=0, epochs=1000000,
        steps_per_epoch=500, clip_ratio=0.2, pi_lr=3e-3,
        vf_lr=1e-3, max_ep_len=1000, train_pi_iters=200, train_v_iters=200,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
        lam=self.lam
        gamma=self.gamma
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        # setup_pytorch_for_mpi()

        # Set up logger and save configuration
        # train_pi_iters=self.train_pi_iters, train_v_iters=self.train_v_iters
        # steps_per_epoch=self.steps_per_epoch, 
        # gamma=0.99, 
        # lam = self.lam
        # pi_lr=self.pi_lr
        # vf_lr=self.vf_lr

        # max_ep_len=1000, 
        # train_pi_iters=200, 
        # train_v_iters=200,

        # Random seed
        #seed += 10000 * proc_id()
        if self.testing:
            torch.manual_seed(seed)
            np.random.seed(seed)

    
        # Sync params across processes
        # sync_params(ac)
    
        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.v])
        # print('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)
    
        # Set up experience buffer
        local_steps_per_epoch = int(steps_per_epoch)

        buf = PPOBuffer(self.nspin+1, self.nspin+1, local_steps_per_epoch, gamma, lam)
    
        # Set up function for computing PPO policy loss
        def compute_loss_pi(data):
            obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
    
            # Policy loss
            pi, logp = self.ac.pi(obs, act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
    
            # Useful extra info
            approx_kl = (logp_old - logp).mean().item()
            ent = pi.entropy().mean().item()
            clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
    
            return loss_pi, pi_info
    
        # Set up function for computing value loss
        from tqdm import tqdm
        def compute_loss_v(data, cond: bool=False):
            obs, ret = data['obs'], data['ret']
            if cond:
                wd_ret = torch.zeros_like(ret)
                for i,ob in enumerate(obs.numpy()):
                    wd_ret[i]=-1*self.Monte_env.wass_cost(ob, bootstrap_reps=30)
            else:
                wd_ret=ret

            return ((self.ac.v(obs) - wd_ret)**2).mean()
    
        # Set up optimizers for policy and value function
        pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
    
        # Set up model saving here::::: maybe....
    
    
        def update(cond):
            data = buf.get()
    
            pi_l_old, pi_info_old = compute_loss_pi(data)
            pi_l_old = pi_l_old.item()
            v_l_old = compute_loss_v(data).item()
    
            # Train policy with multiple steps of gradient descent
            for i in range(train_pi_iters):
                pi_optimizer.zero_grad()
                loss_pi, pi_info = compute_loss_pi(data)
                kl = pi_info['kl']
                # print(f"kl = {kl}")
                # raise AssertionError("cbp")
                if kl > 1.5 * target_kl:
                    #print('Early stopping at step %d due to reaching max kl.'%i)
                    break
                loss_pi.backward()
                #mpi_avg_grads(ac.pi)    # average grads across MPI processes
                pi_optimizer.step()

    
            #logger.store(StopIter=i)
    
            # Value function learning
            iters = tqdm(range(train_v_iters)) if cond else range(train_v_iters)
            for i in iters:
                vf_optimizer.zero_grad()
                loss_v = compute_loss_v(data, cond)
                loss_v.backward()
                # print(loss_v)
                # mpi_avg_grads(ac.v)    # average grads across MPI processes
                vf_optimizer.step()
    
            # Log changes from update
            # kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
            # logger.store(LossPi=pi_l_old, LossV=v_l_old,
            #               KL=kl, Entropy=ent, ClipFrac=cf,
            #               DeltaLossPi=(loss_pi.item() - pi_l_old),
            #               DeltaLossV=(loss_v.item() - v_l_old))
    
        # Prepare for interaction with environment

        o, ep_ret, ep_len = self.env.reset(), 0, 0
        o = np.concatenate((np.diag(o), [0]))  # matrix'ed
    
        # Main loop: collect experience in env and update/log each epoch
        max_fid_seen = 0; true_fid=0
        funcalls = 0
        iterations = 0
        start_time = tt.time()
        repeats = 0
        run_until_completion_criterion = False
        running_controllers = {}
        for epoch in range(epochs):
            for t in range(local_steps_per_epoch):
                #print(torch.as_tensor(o, dtype=torch.float32))
                
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
    
                
                action_and_time = a
                action, time = action_and_time[:-1], action_and_time[-1]
                action = np.diag(action)
                self.env.timestep += time
                self.env.tres = self.env.timestep 
                self.env.final_time = self.env.timestep

                next_o, r, d = self.env.step(action)
                if not self.adaptive:
                    if self.use_fixed_ham:
                        funcalls += 1*self.train_size # number of times you call the environment
                    else:
                         funcalls += 1
                else:
                    funcalls += self.env.adp_func_calls_increment
                    self.env.adp_func_calls_increment  = self.draws
                    
                ep_ret += r
                ep_len += 1
    
    
                
                if self.ham_noisy or self.fid_noisy:
                    if max_fid_seen <= r:
                        if self.use_fixed_ham:
                            true_fid = None # self.env.true_fid(next_o, self.env.timestep)
                        else:
                            true_fid = self.env.tf
                        max_fid_seen = r
                else:
                    max_fid_seen = max(max_fid_seen, r)
                
                if self.verbose:
                    print(f"max_fid_obtained: {max_fid_seen}, true_fid: {true_fid}")
                    print(f"func calls {funcalls}")
                # print(max_fid_seen, r)
                next_store = np.concatenate((np.diag(next_o), [self.env.timestep])) 
                next_o = np.concatenate((np.diag(next_o), [self.env.timestep]))
                
                # save and log
                buf.store(o, a, r, v, logp)
                # logger.store(VVals=v)
                
                # Update obs (critical!)
                o = next_o
    
                ttimeout = ep_len == max_ep_len
                terminal = d or ttimeout
                epoch_ended = t==local_steps_per_epoch-1
                
                # if max_fid_seen > self.fid_threshold:
                #     return
    
                if terminal or epoch_ended:
                    #if epoch_ended and not(terminal):
                        # print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if ttimeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                    buf.finish_path(v)
                    # if terminal:
                    #     # only save EpRet / EpLen if trajectory finished
                    #     # logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0
                    o = np.concatenate((np.diag(o), [0]))  # matrix'ed
                    
                def save_controller_data_aux(): # auxilliary routine to save space
                    self.record["time_to_get_fid"] = tt.time()-start_time
                    self.record["func_calls"] = funcalls
                    self.record["iterations"] = iterations
                    self.record["repeats"] = repeats 
                    self.record["controller"] = next_store.tolist()
                    if self.landscape_exploration:
                        self.record["controllers"] = list(running_controllers.values())
                        if self.records_update_rate:
                            self.record_collector(funcalls, self.record["controllers"])
                        
                    # print("wd final soln: ", self.Monte_env.wass_cost(next_store, 1000))
                    if self.ham_noisy or self.fid_noisy:
                        self.record["best_fid"] = true_fid 
                    else:
                        self.record["best_fid"] = max_fid_seen
    
                if not self.run_until_told_to_stop: 
                    # premature stopping after crossing threshold
                    if max_fid_seen >= self.fid_threshold:
                        save_controller_data_aux()
                        if self.save:
                            self.save_record()
                        # print(self.record)
                        return max_fid_seen
                else:
                     # update current best until time out
                    if self.record["best_fid"] is None:
                        crit = r >= self.fid_threshold
                    else:
                        crit = r >= self.record["best_fid"]
                        if self.landscape_exploration:
                            crit = True # tautology: keep updating the list constantly: worried about case: 0.99 fid at iter < desired controllers will most likely never store up
                    if crit:
                        if self.landscape_exploration:
                            l=len(running_controllers.keys())
                            if l < self.save_topc:
                                running_controllers[r]=next_store.tolist()
                                # print("running_list: \n", running_controllers)
                            else:
                                #itopop=self.find_min_fid_index(running_controllers) # time to pop this ###
                                itopop=min(list(running_controllers.keys()))
                                running_controllers.pop(itopop)
                                running_controllers[r]=next_store.tolist() # maintain const size list
                                # print("running_list: \n", running_controller_list)
                        save_controller_data_aux()
              
                    if run_until_completion_criterion:
                        return max_fid_seen
                if tt.time()-start_time > self.timeout: # relegated to a fail-safe (extremely unlikely but don't want to wait all day for 1 run)
                    print(f"timed out! {self.filename}")
                    raise AssertionError("timeout")
                
                # run for a fixed number of iterations and then terminate
                run_until_completion_criterion = funcalls+1 >= self.run_until_completion_its
            # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs-1):
            #     logger.save_state({'env': env}, None)
    
            # Perform PPO update!
            update(max_fid_seen > 1)
            iterations += train_v_iters
            
    
            # Log info about epoch
if __name__ == '__main__':
    # import argpars
    trial = PPO_en(5, 0, 2, verbose=True, testing=False, fid_noisy=False, 
                    ham_noisy=True, noise=0.1, adp_tol=0.02, draws=100, timeout=7200,
                    fid_threshold=0.6, run_until_completion_its=6000000, run_until_told_to_stop=True,
                    max_time=70, landscape_exploration=True, save_topc=100, use_fixed_ham=False, records_update_rate=100)

    trial.run()

