from mcsim import MCDataSim, remove_redundant_ticks, vn_test
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sp
from wd_sortof_fast_implementation import wd_from_ideal_zero
import seaborn as sns
from multiprocessing import Pool
import matplotlib
from scipy.stats import kendalltau

class Individual_cont_comparisons(MCDataSim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_fig_save_directory("gray_scale_adjusted_paperfigs")
    def plot_figs_3_6_10_11_12(self, algo: str = None, plot_noises = None, 
                    noise_keys = None, remove_final_axis=False,
                    fid_thres: float=0.95, figname="poster_noisycomp"):
        
        specific_noise_keys = noise_keys 
        if plot_noises is None:
            plot_noises=self.noises
        
        if isinstance(algo, str): # assuming only one algo in training
            algo = [algo]
        elif algo is None:
            algo = self.algos 

        self.figlabels = ["({})".format(i) for i in "abcdefghijklmnopqrstuvwxyz"]   # PRA customs
        figlabelindex = 0
        
        def pcolorwrm(wd_data_c, alg, pfig7, pax7, pltcolbar=False, sigma_sims=self.noises, fontsize=20):
            
            idx = np.ix_(np.ones(wd_data_c.shape[0], dtype=bool), wd_data_c[0].argsort())
            coo=pax7.pcolor(np.log(wd_data_c[idx]), norm=matplotlib.colors.Normalize(vmin=-5, vmax=0), cmap="viridis")
            from matplotlib import ticker
            ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/(10*(len(self.noises)-1))))
            pax7.yaxis.set_major_formatter(ticks_y)
            altRIMlabel=r"$W(P^{(i)}_{\sigma_{\rm sim}}(\mathcal{I}),\delta(\mathcal{I}))$"
            if pltcolbar:
                pfig7.subplots_adjust(right=0.90)
                cbar_ax = pfig7.add_axes([0.91, 0.15, 0.03, 0.8])
                pfig7.colorbar(coo, ax=pax7, cax=cbar_ax)
                for t in cbar_ax.get_yticklabels():
                    t.set_fontsize(fontsize)
                cbar_ax.set_ylabel(r'$\log{\rm{RIM}}$', fontsize=20)
                pax7.set_title(alg, fontsize=fontsize-5)
                pax7.tick_params(axis='both', which='major', labelsize=15)

            pax7.set_title(alg, fontsize=fontsize-5)
            pax7.tick_params(axis='both', which='major', labelsize=15)


        plti = 0
        for alg in algo:
            # print(algo)
            
            if noise_keys is None:
                noise_keys = list(self.controllers[alg].keys())
                fig7, ax7 = plt.subplots(nrows=(len(noise_keys)+2)//2, ncols=len(algo), figsize=(10,5))
                fig7.tight_layout()
                if isinstance(ax7, np.ndarray):
                    ax7=ax7.ravel()
                else:
                    ax7 = np.array([ax7])

            elif specific_noise_keys is not None:
                r = len(algo)-1
                if r == 0:
                    r=1
                # pltrows = (r*len(specific_noise_keys))//2-2
                pltrows = (r*len(specific_noise_keys))//2-4 if len(noise_keys) != 1 else 1
                pltcols = len(algo)
                fig7, ax7 = plt.subplots(nrows=pltrows, ncols=pltcols, figsize=(13,7))
                fontsize=20
                if len(ax7.shape) == 1:
                    ax7 = ax7[None,: ]
                remove_redundant_ticks(ax7, pltrows, pltcols)

                fig7.supxlabel("controller", fontsize=fontsize)
                fig7.supylabel(r"$\sigma_{sim}$", fontsize=fontsize)
                fig7.tight_layout()
                if not isinstance(ax7, np.ndarray):
                    ax7 = np.array([ax7])
                ax7=ax7.ravel()
                strspecific_noise_keys = [str(i) for i in specific_noise_keys]
                noise_keys = [str(i) for i in list(self.controllers[alg].keys()) if i in strspecific_noise_keys]
                print(list(self.controllers[alg].keys()))
                print(noise_keys)
                specific_noise_keys = None
                # raise Exception
                
            if alg == "lbfgs":
                # continue
                wd_data = self.get_metrics_dict(None, plot_noises, algoname=alg)[alg]
                wd_data_c = wd_data[r'$W(.,\delta(x-1))$']
                wd_data_u, wd_data_l = wd_data[r'$W(.,\delta(x-1))$'+ ' upper'], wd_data[r'$W(.,\delta(x-1))$'+ ' lower'] 
                
                # filter out to look at unsorted but topk controllers
                wd_data_c = np.array(wd_data_c) # shape: (plot_noise_res, controller_counts)
                wd_data_u = np.array(wd_data_u)
                wd_data_l = np.array(wd_data_l)
                
                if self.topk:
                    wd_data_c, wd_data_u, wd_data_l = self.get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, self.topk, None) 
                    wd_data_c2, wd_data_u2, wd_data_l2 = self.get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, self.topk, fid_thres) 
                
                print("plti", plti)
                pcolorwrm(wd_data_c, self.figlabels[figlabelindex]+" "+alg, fig7, ax7[plti], pltcolbar=True)
                if remove_final_axis:
                    fig7.delaxes(ax7[plti+1])
                # plt.show()
                # raise AssertionError
                self.save_fig(fig7, name=figname, keepsimple=True)
                return

            else:
                for i in range(len(noise_keys)):

                    wd_data = self.get_metrics_dict(noise_keys[i], plot_noises, algoname=alg)
                    wd_data=wd_data[alg]

                    wd_data_c = wd_data[r'$W(.,\delta(x-1))$']
                    wd_data_u, wd_data_l = wd_data[r'$W(.,\delta(x-1))$'+ ' upper'], wd_data[r'$W(.,\delta(x-1))$'+ ' lower'] 
                    wd_data_c = np.array(wd_data_c); wd_data_u = np.array(wd_data_u)
                    wd_data_l = np.array(wd_data_l)
                    
                    
                    if self.topk:
                        wd_data_c, wd_data_u, wd_data_l = self.get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, self.topk, None) 
                    
                    algoname = alg
                    if alg == "nmplus":
                        algoname = "nm"                            
                    
                    alglabel=self.figlabels[figlabelindex]+" "+algoname+r" $\sigma_{train}$="+f"{noise_keys[i]}"
                    figlabelindex += 1
                    pltcolbar = True if alg=="ppo" and noise_keys[i] == noise_keys[-1] else False
                    pcolorwrm(wd_data_c, alglabel, fig7, ax7[plti], pltcolbar=pltcolbar)
                    plti+=1

        

    def plot_fig3e(self, algo: str = None, plot_noises = None, 
                   noise_keys = None, fid_thres: float=0.95, 
                   best_and_gt_fid_thres=False, figname="indvid_cont_comp"):
        
        if plot_noises is None:
            plot_noises=self.noises
        
        if isinstance(algo, str): # assuming only one algo in training
            algo = [algo]
        elif algo is None:
            algo = self.algos 

        fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
        lw4=5 # linewidth for a subplot in ax4 
        
        self.figlabels = ["({})".format(i) for i in "abcdefghijklmnopqrstuvwxyz"]   # PRA customs
        figlabelindex = 0

        plti = 0
        for alg in algo: 
            if alg == "lbfgs":
                # continue
                wd_data = self.get_metrics_dict(None, plot_noises, algoname=alg)[alg]
                wd_data_c = wd_data[r'$W(.,\delta(x-1))$']
                wd_data_u, wd_data_l = wd_data[r'$W(.,\delta(x-1))$'+ ' upper'], wd_data[r'$W(.,\delta(x-1))$'+ ' lower'] 
                
                # filter out to look at unsorted but topk controllers
                wd_data_c = np.array(wd_data_c) # shape: (plot_noise_res, controller_counts)
                wd_data_u = np.array(wd_data_u)
                wd_data_l = np.array(wd_data_l)
                
                
                if self.topk:
                    wd_data_c, wd_data_u, wd_data_l = self.get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, self.topk, None) 
                    wd_data_c2, wd_data_u2, wd_data_l2 = self.get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, self.topk, fid_thres) 


                figlabelindex += 1
                plti += 1

                label=f"{alg} "
                bcperf, avcperf, bco, avo, bestpernoise = self.get_best_controller_perf(wd_data_c, label, contcount=self.topk)
                bcperf2, avcperf2, bco2, avo2,corrfac = self.get_best_controller_perf(wd_data_c2, label, contcount=self.topk)

                # thresholded best
                color=ax4.get_lines()[-1].get_color()

                ax4.semilogy(plot_noises, bco, label=f"{alg} best", linestyle="-", linewidth=lw4, marker="D", ms=15, alpha=0.7)
                if best_and_gt_fid_thres:
                    ax4.semilogy(plot_noises, bco2, label="indicates best & "+r"$\mathcal{F}>$"+f"{fid_thres}", 
                                linestyle="dotted", linewidth=lw4-1, marker="D", ms=10, alpha=0.6, c="red")

                ax4.semilogy(plot_noises, avo, 
                         label="indicates average", linestyle="-.", linewidth=lw4-1, color=color, alpha=0.5, marker="D", ms=10)

                # raise AssertionError

            else:
                for i in range(len(noise_keys)):
                    print(alg, i)
                    if alg =="snob":
                        nalgomarker = "^"  
                    elif alg == "nmplus":
                        nalgomarker = "v"
                    else:
                        nalgomarker = "o"
                    wd_data = self.get_metrics_dict(noise_keys[i], plot_noises, algoname=alg)
                    wd_data=wd_data[alg]
                    # print(alg, i, wd_data)
                    # print(wd_data)
                    wd_data_c = wd_data[r'$W(.,\delta(x-1))$']
                    wd_data_u, wd_data_l = wd_data[r'$W(.,\delta(x-1))$'+ ' upper'], wd_data[r'$W(.,\delta(x-1))$'+ ' lower'] 
                    wd_data_c = np.array(wd_data_c); wd_data_u = np.array(wd_data_u)
                    wd_data_l = np.array(wd_data_l)
                    
                    
                    if self.topk:
                        wd_data_c, wd_data_u, wd_data_l = self.get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, self.topk, None) 
                        wd_data_c2, wd_data_u2, wd_data_l2 = self.get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, self.topk, fid_thres) 
                    
                    algoname = alg
                    if alg == "nmplus":
                        algoname = "nm"                            
                    
                    # jkt_or_ordinaltau_pairwise(wd_data_c, alpha=alpha)
                    # for alpha in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]:

                    figlabelindex += 1

                    plti+=1

                    label=f"{algoname} "+"$\sigma_{{train}}$="+f"{noise_keys[i]}"

                    bcperf, avcperf, bco, avo, bestpernoise = self.get_best_controller_perf(wd_data_c, label, contcount=self.topk)
                    bcperf2, avcperf2, bco2, avo2,corrfac = self.get_best_controller_perf(wd_data_c2, label, contcount=self.topk)

                    ax4.semilogy(plot_noises, bco, 
                              label=label+" best", linewidth=lw4, marker=nalgomarker, ms=15, alpha=0.7)
                    color=ax4.get_lines()[-1].get_color()

                    ax4.semilogy(plot_noises, avo, 
                              #label=f"{alg} "+"$\sigma_{{train}}$="+f"0.0{i} average", 
                              linewidth=lw4, 
                              linestyle="-.", color=color, alpha=0.5,  marker=nalgomarker, ms=15)

                    # indicates best and >F plot below
                    if best_and_gt_fid_thres:
                        ax4.semilogy(plot_noises, bco2, 
                                label=None, linewidth=lw4-1, marker=nalgomarker, ms=10, alpha=0.6, c="red", linestyle="dotted")
                    


        ax4.set_xlabel("$\sigma_{sim}$", fontsize=30)

        ax4.set_title(self.figlabels[figlabelindex], fontsize=30)
        figlabelindex += 1

        ax4.set_ylabel(r"${\rm RIM}_{c}$", fontsize=30)
        ax4.tick_params(axis='both', which='major', labelsize=30)

        ax4.legend(fontsize=20)
        
        fig4.tight_layout()
        self.save_fig(fig4, name=figname, keepsimple=True)



if __name__ == '__main__':
    y = Individual_cont_comparisons(experiment_name="pipeline_nmplus2", Nspin=5, outspin=2,
                  bootreps=100, parallel=False, numcontrollers=1000, filemarker=".le", #None,
                  noises=np.linspace(0,0.1,11))
    # fig 3
    y.plot_figs_3_6_10_11_12(noise_keys=np.linspace(0,0.1,11)[:1], figname="fig3")
    y.plot_fig3e(noise_keys=np.linspace(0,0.1,11)[:1], figname="fig3e")
    # # fig 6
    y.plot_figs_3_6_10_11_12(noise_keys=np.linspace(0,0.1,11)[:6], figname="fig6", remove_final_axis=True)
    # # fig 10
    y2 = Individual_cont_comparisons(experiment_name="pipeline_nmplus2", Nspin=5, outspin=4,
                  bootreps=100, parallel=False, numcontrollers=1000, filemarker=".le", #None,
                  noises=np.linspace(0,0.1,11))
    y2.plot_figs_3_6_10_11_12(noise_keys=np.linspace(0,0.1,11)[:1], figname="fig10")
    y2.plot_fig3e(noise_keys=np.linspace(0,0.1,11)[:1], figname="fig10e", best_and_gt_fid_thres=True)

    # # fig 11
    y2.plot_figs_3_6_10_11_12(noise_keys=np.linspace(0,0.1,11)[:6], figname="fig11", remove_final_axis=True)
    # fig 12
    y3 = Individual_cont_comparisons(experiment_name="pipeline_nmplus2", Nspin=6, outspin=5,
                  bootreps=100, parallel=False, numcontrollers=1000, filemarker=".le", #None,
                  noises=np.linspace(0,0.1,11))
    y3.plot_figs_3_6_10_11_12(noise_keys=np.linspace(0,0.1,11)[:6], figname="fig12", remove_final_axis=True)

    # fig 13 extra
    y3 = Individual_cont_comparisons(experiment_name="pipeline_nmplus2", Nspin=6, outspin=3,
                  bootreps=100, parallel=False, numcontrollers=1000, filemarker=".le", #None,
                  noises=np.linspace(0,0.1,11))
    y3.plot_figs_3_6_10_11_12(noise_keys=np.linspace(0,0.1,11)[:6], figname="fig13", remove_final_axis=True)
