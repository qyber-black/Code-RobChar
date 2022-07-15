#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 20:32:33 2021

@author: irtazakhalid
"""

from mcsim import MCDataSim
from wd_sortof_fast_implementation import wd_from_ideal_zero
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
def sum_pcolalldata_():
    noises=np.linspace(0,0.1,11)
    plt.figure()
    nspins=[4,4,5,5,6,6,7,7,8,8,9,9]
    outspins=[2,3,2,4,3,5,3,6,4,7,4,8]
    zero_noise_beats_lbfgs = 0
    for Nspin,outspin in zip(nspins, outspins):
        mcobj = MCDataSim(experiment_name="pipeline_beta_for_real", Nspin=Nspin, outspin=outspin,
                          bootreps=100, parallel=False, numcontrollers=1000, filemarker=".le", 
                          noises=noises, topk=100)
        
        data=np.array(mcobj.get_wd_data_c())
        data = data.sum(axis=-1).sum(axis=-1) # crude crude
        # data = data.sum(axis=-1)
        # print(np.argmin(data), data.shape)
        # val = data[np.argmin(data)]
        # data[np.arange(len(data))!= val] = np.nan
        bool = (data[9] < data[-1])
        zero_noise_beats_lbfgs += bool
        print(zero_noise_beats_lbfgs, f"prob Nspin, outspin: {Nspin},{outspin} ppo beats l: {bool}")
        plt.plot(range(12), data, label=f"Nspin={Nspin}, outspin={outspin}", linestyle="-", marker="o",)
    
    plt.xlabel("algo", fontsize=30)
    plt.ylabel("sum WRB across $\sigma_sim$", fontsize=30)
    
    plt.legend(fontsize=20)

class ARIM_generator(MCDataSim):
    "read: algorithm robustness infidelity measure"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ncolors = ["blue", "green", "purple", "gold", "orange", "red", "brown", 
                        "gray", "mediumseagreen", "olive", "cyan"]
        self.lbfgscol = "darkgreen"
        self.lbfgsmarker = "D"
        # styles ["loosely dashdotted",  "densely -1", "loosely dotted,]
        self.linestyles = [(0, (3, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 10)), "solid", "dotted", "dashed"]
        self.set_fig_save_directory("gray_scale_adjusted_paperfigs")
        

    def get_ARIM(self, algo: str = None, plot_noises = None, 
                 noise_keys = None, plot_error: bool = False, ax2=None, ylim=None):
        
        
        specific_noise_keys = noise_keys 
        if plot_noises is None:
            plot_noises=self.noises
        
        if isinstance(algo, str): # assuming only one algo in training
            algo = [algo]
        elif algo is None:
            algo = self.algos 

        if ax2 is None:
            fig2, ax2 = plt.subplots(nrows=1)

        def get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, topk):
            filmask = self.get_ranks(wd_data_c[0]) <= topk-1
            idx = np.ix_(np.ones(wd_data_c.shape[0], dtype=bool), filmask)
            wd_data_c = np.array(wd_data_c)[idx] 
            wd_data_u = np.array(wd_data_u)[idx]
            wd_data_l = np.array(wd_data_l)[idx]
            return wd_data_c, wd_data_u, wd_data_l


        for alg in algo:
            # print(algo)
            if noise_keys is None:
                noise_keys = list(self.controllers[alg].keys())


            elif specific_noise_keys is not None:
                r = len(algo)-1
                if r == 0:
                    r=1
                pltrows = (r*len(specific_noise_keys))//2
                pltcols = len(algo)

                strspecific_noise_keys = [str(i) for i in specific_noise_keys]
                noise_keys = [str(i) for i in list(self.controllers[alg].keys()) if i in strspecific_noise_keys]

                specific_noise_keys = None
                # raise Exception
                def save_fig(fig, name="noiseless_comp"):
                    cname = self.get_controller_name.split("/")[-1]
                    fig.savefig(f"paperfigs2/{name}_c{pltcols}_r{pltrows}_{cname}.pdf", dpi=1000)
                
            if alg == "lbfgs":
                wd_data = self.get_metrics_dict(None, plot_noises, algoname=alg)[alg]
                wd_data_c = wd_data[r'$W(.,\delta(x-1))$']
                wd_data_u, wd_data_l = wd_data[r'$W(.,\delta(x-1))$'+ ' upper'], wd_data[r'$W(.,\delta(x-1))$'+ ' lower'] 
                
                # filter out to look at unsorted but topk controllers
                wd_data_c = np.array(wd_data_c) # shape: (plot_noise_res, controller_counts)
                wd_data_u = np.array(wd_data_u)
                wd_data_l = np.array(wd_data_l)
                
                if self.topk:
                    wd_data_c, wd_data_u, wd_data_l = get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, self.topk) 

              
                wdd = wd_data_c[~np.isnan(wd_data_c)].reshape((len(plot_noises),-1))
                
                wddl = wd_data_l[~np.isnan(wd_data_l)].reshape((len(plot_noises),-1))
                wddu = wd_data_u[~np.isnan(wd_data_u)].reshape((len(plot_noises),-1))
                ps_c = [wd_from_ideal_zero(wdd[j]) for j in range(len(wd_data_c))]
                

                
                # 4. nonparametric bootstrap resampling
                ps_l = np.array([self.bootstrap_resampling_std(wd_from_ideal_zero, wdd[j], 100) 
                                 for j in range(len(wd_data_c))])
                ps_u = ps_l
                
                lin_model = sp.linregress(plot_noises*10, y=ps_c)
                slope=round(lin_model.slope,3)
                label=f"{alg} "
                ax2.plot(plot_noises, ps_c, label=label, linewidth=2,
                         marker=self.lbfgsmarker, color=self.lbfgscol, ms=5, alpha=0.7, linestyle="solid")

                if plot_error:
                    ax2.fill_between(plot_noises, ps_c - 2*ps_l, ps_c + 2*ps_u, alpha=0.2, color=self.lbfgscol)
                
            else:
                ps_cs = []
                algoname = "nm" if alg == "nmplus" else alg
                for i in range(len(noise_keys)):
                    print(alg, i)
                    if alg =="snob":
                        nalgomarker = "^"  
                    elif alg == "nmplus":
                        nalgomarker = "v"
                    else:
                        nalgomarker = "o"
                        
                    wd_data = self.get_metrics_dict(noise_keys[i], plot_noises, algoname=alg)[alg]
                    # print(alg, i, wd_data)
                    # print(wd_data)
                    wd_data_c = wd_data[r'$W(.,\delta(x-1))$']
                    wd_data_u, wd_data_l = wd_data[r'$W(.,\delta(x-1))$'+ ' upper'], wd_data[r'$W(.,\delta(x-1))$'+ ' lower'] 
                    wd_data_c = np.array(wd_data_c); wd_data_u = np.array(wd_data_u)
                    wd_data_l = np.array(wd_data_l)
                    
                    
                    if self.topk:
                        wd_data_c, wd_data_u, wd_data_l = get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, self.topk) 
                    
                    
                    wdd = wd_data_c[~np.isnan(wd_data_c)].reshape((len(plot_noises),-1))
                    
                    wddl = wd_data_l[~np.isnan(wd_data_l)].reshape((len(plot_noises),-1))
                    wddu = wd_data_u[~np.isnan(wd_data_u)].reshape((len(plot_noises),-1))
                    ps_c = np.array([wd_from_ideal_zero(wdd[j]) for j in range(len(wd_data_c))])
                    ps_cs.append(ps_c)
                    
                    ps_l = np.array([self.bootstrap_resampling_std(wd_from_ideal_zero, wdd[j], 100) 
                            for j in range(len(wd_data_c))])
                    ps_u = ps_l
                    
                    # lin_model = sp.linregress(plot_noises*10, y=ps_c)
                    # lin_modell = sp.linregress(plot_noises*10, y=ps_l)
                    # lin_modelu = sp.linregress(plot_noises*10, y=ps_u)
                    # slope=round(lin_model.slope,3)
                    label=f"{algoname} "+"$\sigma_{{train}}$="+f"{noise_keys[i]}"
                    if alg != "ppo" and alg != "lbfgs":
                        if i != 0:
                            label=None
                        else:
                            label = f"{algoname} various"
                        
                    ax2.plot(plot_noises, ps_c, 
                             label=label, linewidth=2, marker=nalgomarker, ms=6, alpha=0.7, color=self.ncolors[i], linestyle=self.linestyles[i] )
                    if plot_error:
                        ax2.fill_between(plot_noises, ps_c - 2*ps_l, ps_c + 2*ps_u, alpha=0.2, color=self.ncolors[i], linestyle=self.linestyles[i])
        
        # ax2.set_xlabel("$\sigma_{sim}$", fontsize=20)
        altlabel="Wasserstein robustness measure"
        # ax2.set_ylabel("ARIM", fontsize=20) # alternative label: r"$W(W(P_\delta(f), \delta(f-1)),\delta(W-0)$"
        # ax2.tick_params(axis='both', which='major', labelsize=15)
        # ax2.legend(fontsize=15)
        if ylim is None:
            ylim = 0.6
        ax2.set_ylim(0, ylim)
def get_ARIM_plot(pltns, pipeline_name="pipeline_snob"):
    figlabelindex = 0
    pltrows=2
    pltcols=4
    fig, ax = plt.subplots(pltrows,pltcols, figsize=(17, 7))
    for i in range(pltrows):
        for j in range(pltcols):
            ax[i][j].tick_params(axis='both', which='major', labelsize=16)
            if i != pltrows-1:
                ax[i][j].set_xticks([])
            if j != 0:
                ax[i][j].set_yticks([])
    # _.supylabel("ARIM", fontsize=20, )
    fig.text(-0.02, 0.55, "ARIM", va='center', rotation='vertical', fontsize=30)
    fig.text(0.5, -0.04, r"$\sigma_{\rm sim}$", va='center', fontsize=30)
    # fig.supxlabel(r"$\sigma_{\rm sim}$", fontsize=30)
    fig.tight_layout(pad=0.01)
    i=0
    flag=True
    ax = ax.ravel()
    for nspin, outspin in zip([4,5,6,7,4,5,6,7], [2,2,3,3,3,4,5,6]):
        
        if i > 3:
            ylim = 0.6
        else:
            ylim=None
        y = ARIM_generator(experiment_name=pipeline_name, Nspin=nspin, outspin=outspin,
                  bootreps=100, parallel=False, numcontrollers=1000, filemarker=".le", #None,
                  noises=np.linspace(0,0.1,11))
        y.get_ARIM(noise_keys = np.linspace(0,0.1,11)[:pltns], plot_error=True, ax2=ax[i], ylim=ylim)
        if i < 4:
            ax[i].set_title(y.figlabels[figlabelindex]+" "+ f"M={nspin}", fontsize=16)
        else:
            ax[i].set_title(y.figlabels[figlabelindex], fontsize=13)
        i+=1
        figlabelindex += 1
    box = ax[0].get_position()
    # ax[0][0].set_position([box.x0, box.y0 + box.height * 0.1,
    #                   box.width, box.height * 0.9])
    
    # Put a legend below current axis
    ax[0].legend(loc='upper center', bbox_to_anchor=(2., +1.35),
              fancybox=True, shadow=True, ncol=8, fontsize=13.7)
    y.save_fig(fig, name="fig5_arim_all", keepsimple=True)
        
if __name__ == '__main__':
    # import seaborn as sns
    # sns.set()
    # sum_pcolalldata_()
    get_ARIM_plot(6, pipeline_name="pipeline_nmplus2")
    # y = ARIM_generator(experiment_name="pipeline_nmplus2", Nspin=5, outspin=2,
    #           bootreps=100, parallel=False, numcontrollers=1000, filemarker=".le", #None,
    #           noises=np.linspace(0,0.1,11))
    # fig, ax = plt.subplots(figsize=(6,4))
    # y.get_ARIM(noise_keys=np.linspace(0,0.1,11)[:1], plot_error=True, ax2=ax, ylim=0.65)
    # ax.legend(fontsize=15)
    # ax.set_xlabel(r"$\sigma_{\rm sim}$", fontsize=20)
    # ax.set_ylabel('ARIM', fontsize=20)
    # ax.tick_params(axis='both', which='major', labelsize=15)
    # fig.tight_layout(pad=0.001)
    # y.save_fig(fig, name="ARIM example fig")