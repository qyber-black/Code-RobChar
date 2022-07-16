#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:04:44 2022

@author: irtazakhalid
"""

from mcsim import MCDataSim
import numpy as np
import json
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn 
seaborn.set()

class NStochOpt(MCDataSim):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.c_dict_nsh = self.loadsimdata(self.get_controller_name+"_nsh")
            self.c_dict_sh = self.loadsimdata(self.get_controller_name+"_sh")
            self.lbfgs_no_noise_bench_nlvl="0.0"
        except:
            self.c_dict_nsh = self.loadsimdata(self.get_controller_name)
            self.c_dict_sh = self.loadsimdata(self.get_controller_name)
            self.lbfgs_no_noise_bench_nlvl=""
        
        self.colors = ["blue", "orange", "gold", "green"]
        self.set_fig_save_directory("gray_scale_adjusted_paperfigs")

        self.all_noises_combined_scaling_plot()

    def get_arims(self, algo="lbfgs", nlvl = "0.01", marker="", cdict=None):
        # algo -> nlvl -> fcalls checkpointed cont_dict
        save_fname = self.get_controller_name+"_arims_"+ algo+ nlvl + marker+ ".pickle"
        if os.path.exists(save_fname):
            return pickle.load(open(save_fname, "rb")), None
        # data cleaning
        if algo in cdict:
            fcall_dict = cdict[algo][nlvl]
            keys = list(fcall_dict.keys())
            for key in keys:
                if key in fcall_dict:
                    if len(fcall_dict[key]) < self.numcontrollers:
                        fcall_dict.pop(key)
            new_keys = list(fcall_dict.keys())
            if os.path.exists(save_fname):
                return pickle.load(open(save_fname, "rb")), new_keys
        else:
            # breakpoint()
            raise Exception("Unaccounted for case encountered.")

                    
        arims = np.zeros((len(fcall_dict.keys()), len(self.noises)))
        for j,fcall in enumerate(tqdm(fcall_dict)):
            conts = fcall_dict[fcall]
            rims_all = np.zeros((self.numcontrollers, len(self.noises)))
            # generate pdf for cont
            for i,cont in enumerate(conts):
                rims_all[i] = self.get_rims(cont) # get (self.noises,) rims

            arims[j] = rims_all.sum(axis=0) / len(conts)
            
        pickle.dump(arims, open(save_fname, "wb"))
        return arims, new_keys
        
    def combined_scaling_plot(self, ax, ind, nlvl=0.01):
        nlvl=str(nlvl)
        ax.tick_params(axis='both', which='major', labelsize=16)
    
        for marker, cdict in zip(["nonstoch",""], (self.c_dict_nsh, self.c_dict_sh)):
            for i,algo in enumerate(["lbfgs", "ppo", "snob","nmplus"]):
                algoname = algo
                if algo == "nmplus":
                    algoname = "nm"
                some_arims, _ = self.get_arims(algo, nlvl = nlvl, marker=marker, cdict=cdict)
                keys = np.arange(len(some_arims))*1e6                
                fcalls = list(map(lambda x: int(x), keys))
                mean_arim = np.average(some_arims, axis=-1)[:40]
                boot_std = self.bootstrap_resampling_std(np.mean,  mean_arim, 100)
                ax.set_ylim(0,0.8)
                if marker == "" and algo != "ppo":
                    label = None
                elif marker == "" and algo == "ppo":
                    label = "stoch ppo and others" 
                else:
                    label = algoname+" "+marker
                ax.plot(fcalls[:40], mean_arim, label=label, 
                         color=self.colors[i], linestyle="--" if marker=="" else "-")
                ax.fill_between(fcalls[:40],  mean_arim - 2*boot_std,  mean_arim + 2*boot_std, 
                                alpha=0.2, color=self.colors[i])
                
        lbfgs_no_noise_ref, _ = self.get_arims("lbfgs",nlvl=self.lbfgs_no_noise_bench_nlvl,
                                                marker="", cdict=self.c_dict_sh)

        keys = np.arange(len(lbfgs_no_noise_ref))*1e6                
        fcalls = list(map(lambda x: int(x), keys))
        ax.plot(fcalls[:40], np.average(lbfgs_no_noise_ref, axis=-1)[:40], label="lbfgs no-noise bench", 
                  color="gray", linestyle="dotted")
            
        ax.set_title(self.figlabels[ind]+" "+r" $\sigma_{\rm{train}}$"+f"={nlvl}", fontsize=15)
        
    def all_noises_combined_scaling_plot(self):
        fig, ax = plt.subplots(ncols=3, figsize=(13,4))
        ax = ax.ravel()
        ax[1].set_xlabel("function calls", fontsize=15)
        ax[0].set_ylabel("average ARIM across all " r"$\sigma_{\rm{sim}}$", fontsize=15)
        for i,noise in enumerate([0.01, 0.05, 0.1]):
            self.combined_scaling_plot(ax[i], i, nlvl=noise)
        
        ax[i].legend()
        save_fname = "fig8_arim_scaling"+ "_all_.pdf"
        self.save_fig(fig, save_fname, keepsimple=True)
            

            
    def get_rims(self,cont):
        rims = np.zeros(len(self.noises))
        for i,nlvl in enumerate(self.noises):
            self.noise_model.rng(scale=nlvl)
            f = 0
            for b in range(self.bootreps):
                val = self.noise_model.evaluate_noisy_fidelity(cont, ham_noisy=True)

                f += val
            f = f/self.bootreps
            rims[i]=1-f
        return rims
        

if __name__ == '__main__':
    y = NStochOpt(experiment_name="pipeline_nonstoch_experiments_others_comp", Nspin=5, outspin=2,
                  bootreps=100, parallel=False, numcontrollers=100, filemarker=".le", #None,
                  noises=np.linspace(0,0.1,11)[:])
        
        
    
