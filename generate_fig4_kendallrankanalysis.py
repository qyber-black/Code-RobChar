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

class KTRConsitency(MCDataSim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_fig_save_directory("gray_scale_adjusted_paperfigs")

    def plot_kendalltaus(self, algo: str = None, plot_noises = None, 
                                  noise_keys = None, taufigname="fig4",
                                  taumatrix_plt_flag: bool =False, 
                                  grouped_boxplot: bool = False, fid_thres: float=0.95, 
                                  altfigname="alternative_fig9"):
        
        specific_noise_keys = noise_keys 
        if plot_noises is None:
            plot_noises=self.noises
        
        if isinstance(algo, str): # assuming only one algo in training
            algo = [algo]
        elif algo is None:
            algo = self.algos 
        
        self.figlabels = ["({})".format(i) for i in "abcdefghijklmnopqrstuvwxyz"]   # PRA customs
        figlabelindex = 0
        
        if taumatrix_plt_flag:
            taucols=len(specific_noise_keys)*2+1
            if len(algo) == 1 and len(algo) < len(noise_keys):
                ncols=len(noise_keys)
            elif len(noise_keys) == 1:
                ncols = len(algo)
            else:
                raise Exception("Modify the number of cols manually for ax6... See below. Edge case detected")
            fig6, ax6 = plt.subplots(ncols=ncols, figsize=(12,3), gridspec_kw={'width_ratios': [1]*(ncols-1)+[1.25]})
            fig_alt, ax_alt = plt.subplots(figsize=(10,10))
        else:
            nrows = 1 if len(noise_keys) <=3 else 2
            ncols = len(noise_keys) if len(noise_keys) <=3 else 3
            fig_alt, ax_alt = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,10))
            ax_alt = ax_alt.ravel()
                
        if grouped_boxplot:
            figgb, axgb = plt.subplots(nrows=(len(noise_keys)+1)//2, ncols=2, figsize=(10,10))
            axgb = axgb.ravel()
            pdstore = {j:{} for j in range(10)}
            for j in pdstore:
                pdstore[j]["algo"] = np.array([])
                pdstore[j]["noise"] = np.array([])
                pdstore[j]["wd"] = np.array([])
            figlabelindexbp = 0
        
        def get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, topk, fid_thres=0.8):
            filmask = self.get_ranks(wd_data_c[0]) <= topk-1
            if fid_thres:
                filmask &= wd_data_c[0] <= 1-fid_thres
            idx = np.ix_(np.ones(wd_data_c.shape[0], dtype=bool), filmask)
            wd_data_c = np.array(wd_data_c)[idx] 
            wd_data_u = np.array(wd_data_u)[idx]
            wd_data_l = np.array(wd_data_l)[idx]
            
            return wd_data_c, wd_data_u, wd_data_l
        
        def jkt_or_ordinaltau(wd_data_c, r=1e-3):
            infid_ranks = get_ranks_clustered_little(wd_data_c[0], r=r)

            corrs = []
            inv_tol = 3
            invalids = 0
            printed=False
            for wdi in range(wd_data_c.shape[0]):
                wd_ranks = self.get_ranks(wd_data_c[wdi])+1
                # VN non-parametric rank test of independence of samples following bartels et. al.
                from scipy.signal import detrend
                try: # accept with a tolerance of `inv_tol` failed tests
                    assert vn_test(detrend(wd_ranks), bartels=True, verbose=False)[0] == True, "VN test of independence for RIM samples fails"
                except:
                    invalids += 1
                if invalids == inv_tol and not printed:
                    print("Number of VN tests exceeded tolerance")
                    printed=True
                test = kendalltau(infid_ranks, wd_ranks)
                corrs.append(test.correlation)
            return corrs
        
        def jkt_or_ordinaltau_pairwise(wd_data_c, alpha=0.05, debug_vn_bartlet_test=False):
            allcorrs = []
            for wdj in range(wd_data_c.shape[0]):
                r = alpha*(max(wd_data_c[wdj])-min(wd_data_c[wdj]))
                rim_ranks = get_ranks_clustered_little(wd_data_c[wdj], r=r)
                corrs = []
                invalids = 0
                inv_tol = 1
                printed=False
                for wdi in range(wd_data_c.shape[0]):
                    wd_ranks = self.get_ranks(wd_data_c[wdi])+1
                    if debug_vn_bartlet_test:
                        from scipy.signal import detrend
                        try: # accept with a tolerance of `inv_tol` failed tests
                            vntestout = vn_test(detrend(wd_ranks), bartels=True,verbose=False)
                            vntestvals.append(vntestout[1])
                            assert vntestout[0] == True, "VN test of independence for RIM samples fails"
                        except:
                            invalids += 1
                        if invalids == inv_tol and not printed:
                            print("Number of VN tests exceeded tolerance")
                            printed=True
                    
                    test = kendalltau(rim_ranks, wd_ranks)
                    corrs.append(test.correlation)
                allcorrs.append(corrs)
            return allcorrs

            
        def pcolortaus(allcorrs, ylabel="algo", title=None, colorbar=False, figax=None):
            if figax is None:
                fig, ax = plt.subplots()
            else:
                fig, ax = figax
            coo=ax.pcolor(np.array(allcorrs), norm=matplotlib.colors.Normalize(vmin=0, vmax=1), edgecolors="k", linewidth=3, cmap="viridis")
            from matplotlib import ticker
            ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/(10*(len(self.noises)-1))))
            ax.yaxis.set_major_formatter(ticks_y)
            ax.xaxis.set_major_formatter(ticks_y)
            ax.tick_params(axis='both', which='major', labelsize=12)
            if colorbar:
                #fig6.tight_layout(pad=0.001)
                # fig.subplots_adjust(right=0.90)
                # cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.8])
                fig.colorbar(coo, ax=ax, label=r"$\tilde{\tau}$")
            
            ax.set_xlabel(r"$\sigma_{sim}^{\rm (i)}$", fontsize=15)
            ax.set_ylabel(r"$\sigma_{sim}^{\rm (j)}$", fontsize=15)
            if title:
                ax.set_title(title)
            # ax.legend(fontsize=20)
            
        def get_ranks_clustered_little(infids: np.array, r: float =-1e-15):
            " returns 1d cluster ranks with discrepancy radius r "
            x = infids.copy()
            ucranks = np.argsort(x)
            x0 = min(x)
            x.sort()
            rank = 0
            unsorted_ranks = np.zeros_like(infids)
            for i, ucrank in zip(x, ucranks):
                if i-x0 > r:
                    rank += 1
                    unsorted_ranks[ucrank] = rank
                    x0 = i
                else:
                    unsorted_ranks[ucrank] = rank
            # tests the simple case where r=0 and everything is ranked with an earlier function that was written
            # assert np.allclose(unsorted_ranks - (self.get_ranks(infids)+1), 0)
            # assert int(max(unsorted_ranks)) == len(infids)
            return unsorted_ranks
            
        lbfgstaus=None
        plti = 0
        allcorrs = []
        alpha=0.05; taumatindex = 0
        indii = 0
        vntestvals = []
        for alg in algo:
            # print(algo)
            
            if noise_keys is None:
                noise_keys = list(self.controllers[alg].keys())

            elif specific_noise_keys is not None:
 
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
                    wd_data_c, wd_data_u, wd_data_l = get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, self.topk, None) 
                    wd_data_c2, wd_data_u2, wd_data_l2 = get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, self.topk, fid_thres) 
                
                
                if taumatrix_plt_flag:
                    lbfgstaus = jkt_or_ordinaltau(wd_data_c, 0.05*(max(wd_data_c[0])-min(wd_data_c[0])))#r=-1e-15)
                    
                    # for alpha in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]:
                    lbfgstausall = jkt_or_ordinaltau_pairwise(wd_data_c, alpha=alpha)
                    
                    ax_alt.plot(np.linspace(0,0.1,11), np.array(lbfgstausall)[0], label="lbfgs"+r" $\sigma_{\rm train}=$"+f"{noise_keys[i]}", marker="D", ms=15, lw=5)
                    ax_alt.set_ylabel(r"$\tilde{\tau}_{0,j}$", fontsize=30)
                    ax_alt.set_xlabel(r"$\sigma_{sim}^{(j)}$", fontsize=30)
                    ax_alt.set_title(r" $\sigma_{\rm train}=$"+f"{noise_keys[i]} "+r"$\alpha=$ "+ f"{alpha}", fontsize=30)
                    ax_alt.legend(fontsize=20)
                    ax_alt.tick_params(axis='both', which='major', labelsize=30)
                    fig_alt.tight_layout()
                    self.save_fig(fig_alt, "alternative_fig4", keepsimple=True)
                    pcolortaus(lbfgstausall, ylabel=r"$\sigma_{sim}$", title=self.figlabels[indii]+" "+alg+r" $\alpha=$ "+ f"{alpha}", 
                               colorbar=True, figax=(fig6, ax6[taumatindex]))
                    taumatindex += 1
                    
                    # remove_redundant_ticks(ax6[None,:], pltrows=1, pltcols=taucols, remove_titles=True)
                    fig6.tight_layout()
                    self.save_fig(fig6, name=taufigname, keepsimple=True)
                else:
                    lbfgstausall = jkt_or_ordinaltau_pairwise(wd_data_c, alpha=alpha)
                    for ind, ax in enumerate(ax_alt):
                        ax.plot(np.linspace(0,0.1,11), np.array(lbfgstausall)[0], label="lbfgs"+r" $\sigma_{\rm train}=$"+f"{0}", marker="D", ms=15, lw=5)
                        ax.set_ylabel(r"$\tilde{\tau}_{0,j}$", fontsize=30)
                        ax.set_xlabel(r"$\sigma_{sim}^{(j)}$", fontsize=30)
                        ax.set_title(self.figlabels[ind]+" "r" $\sigma_{\rm train}=$"+f"{noise_keys[ind]} "+r"$\alpha=$ "+ f"{alpha}", fontsize=30)
                        
                        ax.tick_params(axis='both', which='major', labelsize=30)
                    ax_alt[-1].legend(fontsize=20)
                    remove_redundant_ticks(ax_alt.reshape(nrows,-1), pltrows=nrows, pltcols=ncols)
                    fig_alt.tight_layout()
                    self.save_fig(fig_alt, altfigname, keepsimple=True)
                

                figlabelindex += 1
                plti += 1
                
            else:

                if alg =="snob":
                    nalgomarker = "^"  
                elif alg == "nmplus":
                    nalgomarker = "v"
                else:
                    nalgomarker = "o"
                alti=0
                for i in range(len(noise_keys)):
                    wd_data = self.get_metrics_dict(noise_keys[i], plot_noises, algoname=alg)
                    wd_data=wd_data[alg]
                    # print(alg, i, wd_data)
                    # print(wd_data)
                    wd_data_c = wd_data[r'$W(.,\delta(x-1))$']
                    wd_data_u, wd_data_l = wd_data[r'$W(.,\delta(x-1))$'+ ' upper'], wd_data[r'$W(.,\delta(x-1))$'+ ' lower'] 
                    wd_data_c = np.array(wd_data_c); wd_data_u = np.array(wd_data_u)
                    wd_data_l = np.array(wd_data_l)
                    
                    
                    if self.topk:
                        wd_data_c, wd_data_u, wd_data_l = get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, self.topk, None) 
                        wd_data_c2, wd_data_u2, wd_data_l2 = get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, self.topk, fid_thres) 
                    
                    algoname = alg
                    if alg == "nmplus":
                        algoname = "nm"                            

                    corrs = jkt_or_ordinaltau(wd_data_c, r=0.05*(max(wd_data_c[0])-min(wd_data_c[0])))#r=-1e-15)
                    allcorrs.append(corrs)
                    # jkt_or_ordinaltau_pairwise(wd_data_c, alpha=alpha)
                    # for alpha in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]:
                    if taumatrix_plt_flag:
                        
                        tausall = jkt_or_ordinaltau_pairwise(wd_data_c, alpha=alpha)
                        colbar = True if taumatindex == len(noise_keys)-1 and len(algo)==1 else False
                        # colbar=False
                        pcolortaus(tausall, ylabel=r"$\sigma_{sim}^{(i)}$", 
                                   title=self.figlabels[indii]+" "+algoname+r" $\sigma_{\rm train}=$"+f"{noise_keys[i]} "+r"$\alpha=$ "+ f"{alpha}", 
                                   colorbar=colbar, figax=(fig6, ax6[taumatindex]))
                        ax_alt.plot(np.linspace(0,0.1,11), np.array(tausall)[0], label=algoname+r" $\sigma_{\rm train}=$"+f"{noise_keys[i]}", marker=nalgomarker, ms=15, lw=5)
                        if colbar:
                            self.save_fig(fig6, name=taufigname, keepsimple=True)
                            ax_alt.set_ylabel(r"$\tilde{\tau}_{0,j}$", fontsize=30)
                            ax_alt.set_xlabel(r"$\sigma_{sim}^{(j)}$", fontsize=30)
                            ax_alt.set_title(r"$\alpha=$ "+ f"{alpha}", fontsize=30)
                            ax_alt.legend(fontsize=20)
                            ax_alt.tick_params(axis='both', which='major', labelsize=30)
                            fig_alt.tight_layout()
                            self.save_fig(fig_alt, "alternative_fig9", keepsimple=True)
                        taumatindex += 1
                        indii += 1
                    else:
                        tausall = jkt_or_ordinaltau_pairwise(wd_data_c, alpha=alpha)

                        ax_alt[alti].plot(np.linspace(0,0.1,11), np.array(tausall)[0], label=algoname+r" $\sigma_{\rm train}=$"+f"{noise_keys[i]}", marker=nalgomarker, ms=15, lw=5)
                        alti += 1

                    figlabelindex += 1
                    plti+=1

                    if grouped_boxplot:
                        
                        for j in range(len(wd_data_c)):

                            pdstore[i]["wd"] = np.append(pdstore[i]["wd"], wd_data_c[j])
                            pdstore[i]["noise"] = np.append(pdstore[i]["noise"], [j/100]*len(wd_data_c[j]))
                            pdstore[i]["algo"] = np.append(pdstore[i]["algo"],[alg]*self.topk)
                            
                        if alg=="ppo":
                            if i==0: # only add the noiseless training case box plot else seems like a disingenous comparison
                                wd_data = self.get_metrics_dict(None, self.noises, algoname="lbfgs")["lbfgs"]
                                wd_data_c = wd_data[r'$W(.,\delta(x-1))$']
                                wd_data_u, wd_data_l = wd_data[r'$W(.,\delta(x-1))$'+ ' upper'], wd_data[r'$W(.,\delta(x-1))$'+ ' lower'] 
                                wd_data_c = np.array(wd_data_c) 
                                wd_data_u = np.array(wd_data_u)
                                wd_data_l = np.array(wd_data_l)
                                
                                if self.topk:
                                    wd_data_c, wd_data_u, wd_data_l = get_top_k_by_fid(wd_data_c, wd_data_u, wd_data_l, self.topk,  None) 
                                for j in range(len(wd_data_c)):
                                    pdstore[i]["wd"] = np.append(pdstore[i]["wd"], wd_data_c[j])
                                    pdstore[i]["noise"] = np.append(pdstore[i]["noise"], [j/100]*len(wd_data_c[j]))
                                    pdstore[i]["algo"] = np.append(pdstore[i]["algo"],["lbfgs"]*self.topk)
                                
                            
                            pdstoretoplot = pd.DataFrame(pdstore[i])
                            p = sns.boxplot(data=pdstoretoplot, x=pdstoretoplot["noise"], 
                                            y=pdstoretoplot["wd"], hue=pdstoretoplot["algo"], ax=axgb[i], 
                                            width=0.6, whis=1.7)
                            p.set_title(self.figlabels[figlabelindexbp]+" "+r"$\sigma_{\rm train}=$"+f"{noise_keys[i]}", fontsize=20)
                            figlabelindexbp += 1
                            axgb[i].set_ylabel("RIM", fontsize=18)
                            axgb[i].set_xlabel(r"$\sigma_{\rm sim}$", fontsize=20)
                            plt.setp(axgb[i].get_legend().get_texts(), fontsize=18)
                            axgb[i].tick_params(axis='both', which='major', labelsize=18, )
                            axgb[i].tick_params(axis='x', which='major', labelsize=18, rotation=45)
                            
                            
                        # print(pdstore)
                        # #ax[a][c].legend([],[])
                            if i != 0:
                                p.get_legend().remove()
                            from matplotlib import ticker
                            ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/(10*(len(self.noises)-1))))
                            axgb[i].xaxis.set_major_formatter(ticks_x)
                        
                    if taumatrix_plt_flag:
                        if len(noise_keys)==1:
                            break


        if grouped_boxplot:
            axgb = axgb.reshape((len(noise_keys)+1)//2, 2)
            remove_redundant_ticks(axgb, pltrows=(len(noise_keys)+1)//2, pltcols=2, remove_titles=True, remove_x_title_too=True)
            figgb.tight_layout(pad=0.001)
            self.save_fig(figgb, name="fig7_grouped", keepsimple=True)
            

        if lbfgstaus is not None:
            allcorrs.append(lbfgstaus)
        pcolortaus(allcorrs)
        

        figlabelindex += 1
if __name__ == "__main__":
    y = KTRConsitency(experiment_name="pipeline_nmplus2", Nspin=5, outspin=2,
                  bootreps=100, parallel=False, numcontrollers=1000, filemarker=".le", #None,
                  noises=np.linspace(0,0.1,11))
    # # fig 4
    y.plot_kendalltaus(noise_keys=np.linspace(0,0.1,11)[:1], taumatrix_plt_flag=True, taufigname="fig4")
    # # fig 7
    y.plot_kendalltaus(noise_keys=np.linspace(0,0.1,11)[:6], taumatrix_plt_flag=False, grouped_boxplot=True)
    # # # fig 9 tau matrix
    y.plot_kendalltaus("ppo", noise_keys=np.linspace(0,0.1,11)[4:6], taumatrix_plt_flag=True, taufigname="fig9")
    y = KTRConsitency(experiment_name="pipeline_nmplus2", Nspin=5, outspin=4,
                  bootreps=100, parallel=False, numcontrollers=1000, filemarker=".le", #None,
                  noises=np.linspace(0,0.1,11))
    # fig 10+ alternative tau plots
    y.plot_kendalltaus(noise_keys=np.linspace(0,0.1,11)[:6], altfigname="alternative_fig9")

    y = KTRConsitency(experiment_name="pipeline_nmplus2", Nspin=5, outspin=4,
                  bootreps=100, parallel=False, numcontrollers=1000, filemarker=".le", #None,
                  noises=np.linspace(0,0.1,11))

    y.plot_kendalltaus(noise_keys=np.linspace(0,0.1,11)[:6], altfigname="alternative_fig10")