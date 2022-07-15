from mcsim import MCDataSim
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns
from mcsim import remove_redundant_ticks

class ExploringRIMK(MCDataSim):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    def exploring_rim_k(self, noise_index: int = 3, topk=10, p=3, save_dir=None, arim=True, algo="ppo"):
        fs=25; algname1=""
        if algo == "lbfgs":
            ni = None
        else:
            ni = self.noises[noise_index]
        pdf_dict = json.load(open(self.get_mcname(ni, self.noises), "rb"))
        pdf_dict = np.array(pdf_dict[algo]) # shape (noise, cont, samples)
        #idx = self.get_top_k_by_fid_idx(pdf_dict[0].mean(axis=-1), topk)
        pdf_dict = pdf_dict[np.ix_(np.ones(pdf_dict.shape[0], dtype=bool), 
                            self.get_ranks(-1*pdf_dict[0].mean(axis=-1))<=topk)] # filter by observed fid
        from wd_sortof_fast_implementation import RIM_p
        def get_rim_function(k):
            from scipy.stats import skew, kurtosis
            if k=="var":
                rim = lambda cont_dist: cont_dist.var()
            elif k=="skewness":
                rim = lambda cont_dist: 0 #skew(cont_dist)
            elif k=="kurtosis":
                rim = lambda cont_dist: 0# kurtosis(cont_dist)
            else:
                rim = lambda cont_dist : RIM_p(cont_dist, p=k)
            return rim

        def rim_k(k):
            rimks = np.array([list(map(get_rim_function(k), pdf_dict[i])) for i in range(len(pdf_dict))])
            # idxes1 = self.get_top_k_by_fid_idx(rimks, topk=topk)
            # rimks = rimks[idxes1]
            return rimks
        rim_ks = [rim_k(k) for k in range(1,p+1)] # (k, noise, cont)
        rim_ks.append(rim_k("var"))
        rim_ks.append(rim_k("skewness"))
        rim_ks.append(rim_k("kurtosis"))
        rim_ks=  np.array(rim_ks)
        reg_coeffs = np.zeros((p+1+3, topk)) # (k, cont)
        
        # arim moments
        if arim:
            fig, ax = plt.subplots()
            for k in list(range(1,len(rim_ks)-2))+["var", "skewness", "kurtosis"]:
                if isinstance(k, int):
                    label=f"ARIM {k+1}"
                else:
                    label=k
                ax.plot(self.noises, list(map(get_rim_function(k), 1-rim_ks[0])), label=label)
            ax.set_title(f"algo {algo} nlevel opt. {noise_index*0.01} top-k={topk}")
            ax.set_xlabel("noise")
            ax.set_ylabel("ARIM_p")
            ax.legend()
            if save_dir:
                fig.savefig(save_dir+"/"+"arim_p_"+ algo + f"_noise_opt{ni}" +f"_L{self.Nspin}_O{self.outspin}.png",
                             dpi=1000, bbox_inches="tight")
            return
        fig, ax = plt.subplots(1,1)
        for cont in range(topk):
            # for i in range(len(self.noises)):
                # plt.figure()
                # label=""
                # for k in range(len(rim_ks)):
                #     z="rim k={}: {} ".format(k+1, round(rim_ks[k][i][cont],6))
                #     label+= z
                # label += f"noise lvl: {i}"
                # plt.hist(pdf_dict[i][cont], bins=np.linspace(0,1,50))
                # plt.title(label)
            for k in range(len(rim_ks)):
                color=self.colors[k]
                if cont==0:
                    label=f"rim {k+1}"
                    if k==p:
                        label="var"
                    elif k==p+1:
                        label="skewness"
                    elif k==p+2:
                        label="kurtosis"
                else:
                    label=None
                from scipy.stats import linregress 
                assert rim_ks[k][:,cont].shape[-1]==11, f"Not in the noise level index yet {rim_ks[k][:,cont].shape[-1]}"
                if k==0:
                    reg_coeff = linregress(self.noises, rim_ks[k][:,cont])[0]
                    reg_coeffs[k][cont] = reg_coeff
                    reg_coeffs[k+1][cont] = rim_ks[k][:,cont][1]
                else:
                    if k < p:
                        reg_coeffs[k+1][cont] = rim_ks[k][:,cont][1]-rim_ks[0][:,cont][1] # at noise level 1
                    else:
                        reg_coeffs[k+1][cont] = rim_ks[k][:,cont][1]
                ax.plot(self.noises, rim_ks[k][:,cont], label=label,color=color)
                ax.set_xlabel("noise")
                ax.set_ylabel("RIM_k")
        ax.legend()
        cols = []
        for k in range(len(rim_ks)-3):
            if k==0:
                cols.append(f"RIM_1 growth factor {k+1}")
                cols.append(f"RIM {k+1}")
            else:
                cols.append(f"RIM {k+1}")
        cols.append(f"Var")
        cols.append(f"Skew")
        cols.append(f"Kurt")
        df = pd.DataFrame(reg_coeffs.T, columns=cols)
        print(df.corr())
        plt.figure()
        g = sns.pairplot(df, corner=True)
        def corrfunc(x, y, **kws):
            from scipy.stats import kendalltau
            r, _ = kendalltau(x, y)
            ax = plt.gca()
            ax.annotate("tau = {:.2f}".format(r),
                        xy=(.1, .9), xycoords=ax.transAxes)
        g.map_lower(corrfunc)

        raise AssertionError
        lbfgs_wd_data = self.get_metrics_dict(None, self.noises, algoname="lbfgs")["lbfgs"]
        wd_data_c1 = np.array(lbfgs_wd_data[r'$W(.,\delta(x-1))$'])
        print(wd_data_c1.shape)
        idxes1 = self.get_top_k_by_fid_idx(wd_data_c1, topk=topk)
        wd_data_c1 = wd_data_c1[idxes1]
        # wd_data_u, wd_data_l = wd_data[r'$W(.,\delta(x-1))$'+ ' upper'], wd_data[r'$W(.,\delta(x-1))$'+ ' lower'] 
        
        q951 = np.array(lbfgs_wd_data['Q th. 0.95'])[idxes1]
        q981 = np.array(lbfgs_wd_data['Q th. 0.98'])[idxes1]
        
        

        fig, ax = plt.subplots(figsize=(7,7))
        # ax.scatter(-1*q951[noise_index], wd_data_c1[noise_index], alpha=0.5, c="blue", 
        #            label=r"$\mathcal{F}_{\rm Th}$"+"=0.95"+f" \n Spearman={spearman1}")
        # # plt.scatter(-1*q952[2], wd_data_c2[2], alpha=0.5, c="orange")
        # ax.scatter(-1*q981[noise_index], wd_data_c1[noise_index], alpha=0.5, marker="o", 
        #            label=r"$\mathcal{F}_{\rm Th}$"+"=0.98"+f" \n Spearman={spearman2}")
        for k in range(1,p+1):
            ax.scatter(rim_k(2)[noise_index], rim_k(k)[noise_index], alpha=0.5, marker=r"${}$".format(k), 
                       label=f"k={k}", s=100)
        # plt.scatter(-1*q982[3], wd_data_c2[3], alpha=0.5, marker="o")
        ax.set_xlabel(r"$\rm{RIM}_1$", fontsize=fs)
        ax.set_ylabel(r"$\rm{RIM}_k$", fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.legend(fontsize=15)
        ax.set_title(r"$\sigma_{\rm sim}=$"+f"{self.noises[noise_index]}, {algname1}", fontsize=fs)
        # savename="qfactorintuition_N"+str(self.Nspin)+"to"+str(self.outspin)
        # fname = self.save_fig(fig, name=savename, copyto=self.poster_fig_save_folder)
        return
    
    def exploring_metrics(self, noise_index: int = 2, topk=200, allnoisesplot=False):
        fs=25; algname1=""
        lbfgs_wd_data = self.get_metrics_dict(None, self.noises, algoname="lbfgs")["lbfgs"]
        ppo_wd_data = self.get_metrics_dict(self.noises[noise_index], self.noises, algoname="ppo")["ppo"]
        wd_data_c1 = np.array(lbfgs_wd_data[r'$W(.,\delta(x-1))$'])
        idxes1 = self.get_top_k_by_fid_idx(wd_data_c1, topk=topk)
        wd_data_c1 = wd_data_c1[idxes1]
        # wd_data_u, wd_data_l = wd_data[r'$W(.,\delta(x-1))$'+ ' upper'], wd_data[r'$W(.,\delta(x-1))$'+ ' lower'] 
        
        q951 = np.array(lbfgs_wd_data['Q th. 0.95'])[idxes1]
        q981 = np.array(lbfgs_wd_data['Q th. 0.98'])[idxes1]
        

        # savefile = {"rim":wd_data_c1.tolist(), "q_98":q981.tolist(), "q_95":q951.tolist()}
        # json.dump(savefile, open("rim_fig1_data.json", "w"))
        
        ######## ppo controllers ################
        # wd_data_c2 = np.array(ppo_wd_data[r'$W(.,\delta(x-1))$'])
        # idxes2 = self.get_top_k_by_fid_idx(wd_data_c2, topk=topk)
        # wd_data_c2 = wd_data_c2[idxes2]
        # q952 = np.array(ppo_wd_data['Q th. 0.95'])[idxes2]
        # q982 = np.array(ppo_wd_data['Q th. 0.98'])[idxes2]
        
        
        from scipy.stats import spearmanr
        import seaborn as sns
        sns.set()
        spearman1 = round(spearmanr(-1*q951[noise_index], wd_data_c1[noise_index])[0],3)
        spearman2 = round(spearmanr(-1*q981[noise_index], wd_data_c1[noise_index])[0],3)
        if not allnoisesplot:
            fig, ax = plt.subplots(figsize=(7,7))
            ax.scatter(-1*q951[noise_index], wd_data_c1[noise_index], alpha=0.5, c="blue", 
                       label=r"$\mathcal{F}_{\rm Th}$"+"=0.95"+f" \n Spearman={spearman1}")
            # plt.scatter(-1*q952[2], wd_data_c2[2], alpha=0.5, c="orange")
            ax.scatter(-1*q981[noise_index], wd_data_c1[noise_index], alpha=0.5, marker="o", 
                       label=r"$\mathcal{F}_{\rm Th}$"+"=0.98"+f" \n Spearman={spearman2}")
            # plt.scatter(-1*q982[3], wd_data_c2[3], alpha=0.5, marker="o")
            ax.set_xlabel(r"$Y(\mathcal{F}_{\rm Th})$", fontsize=fs)
            ax.set_ylabel("RIM", fontsize=fs)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            ax.legend(fontsize=15)
            ax.set_title(r"$\sigma_{\rm sim}=$"+f"{self.noises[noise_index]}, {algname1}", fontsize=fs)
            savename="qfactorintuition_N"+str(self.Nspin)+"to"+str(self.outspin)
            fname = self.save_fig(fig, name=savename, copyto=self.poster_fig_save_folder)
            return
            
        else:
            fig, ax = plt.subplots(nrows=5, ncols=2)
            ax = ax.ravel()
            fs=15
            for noise_index in range(1,len(self.noises)):
                spearman1 = round(spearmanr(-1*q951[noise_index], wd_data_c1[noise_index])[0],3)
                spearman2 = round(spearmanr(-1*q981[noise_index], wd_data_c1[noise_index])[0],3)
                ax[noise_index-1].scatter(-1*q951[noise_index], wd_data_c1[noise_index], alpha=0.5, c="blue", 
                           label=r"$\mathcal{F}_{\rm Th}$"+"=0.95"+f" \n Spearman={spearman1}")
                # plt.scatter(-1*q952[2], wd_data_c2[2], alpha=0.5, c="orange")
                ax[noise_index-1].scatter(-1*q981[noise_index], wd_data_c1[noise_index], alpha=0.5, marker="o", 
                           label=r"$\mathcal{F}_{\rm Th}$"+"=0.98"+f" \n Spearman={spearman2}")
                # plt.scatter(-1*q982[3], wd_data_c2[3], alpha=0.5, marker="o")
                ax[noise_index-1].set_xlabel(r"$Y(\mathcal{F}_{\rm Th})$", fontsize=fs)
                ax[noise_index-1].set_ylabel("RIM", fontsize=fs)
                ax[noise_index-1].tick_params(axis='both', which='major', labelsize=fs)
                ax[noise_index-1].legend(fontsize=fs-5)
                ax[noise_index-1].set_ylim(0,1)
                ax[noise_index-1].set_xlim(0,1)
                ax[noise_index-1].set_title(r"$\sigma_{\rm sim}=$"+f"{self.noises[noise_index]}, {algname1}", fontsize=fs)
            ax = ax.reshape((5,2))
            remove_redundant_ticks(ax, pltrows=5, pltcols=2, remove_x_title_too=True)
            # plt.tight_layout()


for n,o in zip([4,5,6,7,4,5,6,7], [2,2,3,3,3,4,5,6]):
    y = MCDataSim(experiment_name="pipeline_snob", Nspin=n, outspin=o,
                    bootreps=100, parallel=False, numcontrollers=1000, filemarker=".le", #None,
                    noises=np.linspace(0,0.1,11))
    for algo in ["snob", "ppo", "lbfgs"]:
        for i in range(10):
            try:
                y.exploring_rim_k(noise_index=i, save_dir="rim_p_figs", topk=50, algo=algo)
            except Exception as e:
                print(e)