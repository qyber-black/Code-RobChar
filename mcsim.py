"""
Copyright 2022 Irtaza Khalid

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author: irtazakhalid
"""
import json 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
from wd_sortof_fast_implementation import wd_from_ideal, compute_dkw_error
from noise_model import structured_perturbation
from noise_analysis import ExperimentNamer, DirectoryDoesNotExistError
from typing import List
from multiprocessing import Pool
from typing import Callable, List
from sklearn.manifold import TSNE
from scipy.stats import norm
import glob

def check_numpytype(f):
    def method(arrays, *args, **kwargs):
        if type(arrays)==np.ndarray and len(arrays.shape) == 1:   
            return f(arrays,*args, **kwargs)
        else:
            raise TypeError("make sure arg is a numpy array")
    return method

@check_numpytype
def get_cdf(arrays):
    "return output list of cdfs or one cdf"
    sarrays = np.sort(arrays)
    out = sarrays.cumsum() / sarrays.sum()
    return out, sarrays


@check_numpytype
def get_supcdf(cdf):
    sup = np.zeros_like(cdf)
    len_cdf = len(cdf)
    for i in range(len_cdf):
        supq = sum(cdf[i:]) / (len_cdf-i)
        sup[i]=supq
    return sup

@check_numpytype
def vn_test(obs_v, alpha=0.95, verbose=True, bartels=True):
    """
    Von Neumann successive difference test for randomness (unbiased version)

    Parameters
    ----------
    obs_v : np.1darray
        1-D Vector of random samples 
    alpha : float
        rejection region. The default is 0.01.
    verbose : bool, optional
        Print the statistics. The default is False.
    bartels : bool, optional
        Use the beta approximation rank version proposed by Bartels (1982)

    Raises
    ------
    Exception
        Insufficient sample size (obs_v.size).

    Returns
    -------
    bool
        Test outcome.
    interval : \alpha-interval around the gaussian approximate VN statistic. or nothing
    
    """
    n = obs_v.size
    if n < 40: # TODO: choose a suitable nobs threshold as I exploit asymptotic normality later
        raise Exception("{} nobs are insufficient for the test.".format(n))
    
    mean = 2*n/(n-1) # first moment
    sigma = 4*n*n*(n-2) / ((n+1)*pow((n-1),3)) # second moment
    sdiff = np.diff(obs_v)
    sdiff = sdiff*sdiff
    VN_statistic = sdiff.mean() / obs_v.var()
    if bartels:
        # bartels standardization
        # VN_statistic = (VN_statistic - 2) * np.sqrt(n) / 2
        if verbose:
            print(VN_statistic)
        # bartels' (1982) rank version based on the empirical beta value (1.49,1.54,1.67, 1.71 for 0.005, 0.025, 0.05, 0.1 alpha)
        # 
        if VN_statistic > 1.1: # grid searched satisfiable by all algos
            return True, VN_statistic
        else:
            return False, VN_statistic
    
    # compute the p-value interval using Gaussian approximation
    phi = norm.ppf(1-alpha, loc=mean, scale=np.sqrt(sigma))
    if verbose:
        print("sigma is", sigma)
        print("mean is", mean)
        print("unstd VN is", VN_statistic)
        print("std VN is", (VN_statistic-mean)/np.sqrt(sigma))
        print("thresh is", phi)
        print("="*20)

    # check if you fall in the interval
    # if you do, then data are random else not
    if VN_statistic > phi:
        return True, phi
    else:
        return False, phi

# little test
def test_vn_test():
    x = np.random.normal(0, 1, 500000)
    assert vn_test(x)[0] == True, "failed on random data"
    x = np.arange(1000)
    assert vn_test(x)[0] == False, "failed on non-random data"
    

def ovlen(obj):
    "overloaded `len` finder"
    if isinstance(obj, list) or isinstance(obj, np.ndarray) or isinstance(obj, pd.Series):
        return len(obj)
    elif isinstance(obj, dict):
        return len(obj.keys())
    elif isinstance(obj, int) or isinstance(obj, float):
        return 1
    else:
        raise TypeError("unknown data type encountered")

@check_numpytype
def Q(fid_array, threshold):
    return len(fid_array[fid_array >= threshold]) / len(fid_array)

def wc_fids(fids):
    return map(lambda x: -x,map(min, fids))

def std_fids(fids):
    return map(np.std, fids)

def Q_fids(fids, threshold=0.95):
    def _Q(fids, threshold=threshold):
        return -1*Q(fids, threshold)
    return map(_Q, fids)
def wd_from_ideal_fids(fids):
    return map(wd_from_ideal, fids)

def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)


from dataclasses import dataclass
@dataclass
class Q_partial:
    qthres: float = 0.95   
    def Q_fids(self, fids) -> Callable[[List[float]], List[float]]:
        def _Q(fids, threshold=self.qthres):
            return -1*Q(fids, self.qthres)
        return map(_Q, fids)

__metric_name_to_metric__ = {r'$W(.,\delta(x-1))$':wd_from_ideal_fids, 
                         "Q th. 0.95": Q_partial(qthres=0.95).Q_fids,
                         "Q th. 0.98": Q_partial(qthres=0.98).Q_fids,
                         "std": std_fids, 
                         "worst case fid": wc_fids,
                         }

def remove_redundant_ticks(ax, pltrows, pltcols, remove_titles=False, remove_x_title_too=False):
    for i in range(pltrows):
        for j in range(pltcols):
            if i != pltrows-1:
                ax[i][j].set_xticks([])
                if remove_x_title_too:
                    ax[i][j].set_xlabel(None)
            if j != 0:
                ax[i][j].set_yticks([])
                if remove_titles:
                    ax[i][j].set_ylabel(None)
    
    


class MCDataSim:
    "A class for MC data generation with structured perturbations of XX-controllers."
    def __init__(self, experiment_name: str = "pipeline_alpha", Nspin: int = 5, 
                 inspin: int = 0, outspin: int = 2, 
                 noises: np.ndarray = np.linspace(0,0.1,11), 
                 bootreps: int = 100, training_noise: float=None,
                 numcontrollers: int = 100, parallel: bool = False,
                 num_workers: int = None, 
                 dkw_conflvl: float = 0.95, 
                 filemarker: str = None, 
                 topk: int = 100):
        self.global_experiments_directory = "experiments/"
        self.filemarker = filemarker 
        self.experiment_name=experiment_name
        self.topk = topk
        self.args = dict(Nspin=Nspin, 
                         inspin=inspin, 
                         outspin=outspin)
        self.bootreps = bootreps
        self.alpha = 1-dkw_conflvl
        self.training_noise = training_noise
        self.Nspin= Nspin
        self.inspin = inspin
        self.outspin = outspin
        self.noises = noises
        self.numcontrollers = numcontrollers # ovlen(self.controllers)
        
        self.get_controller_name = self.get_experiment_name(experiment_name)()
        if self.filemarker is not None:
            self.get_controller_name += self.filemarker
        print(self.get_controller_name)
        try:
            self.controllers = self.load_controllers()
            self.algos = self.ctrlnames(self.controllers)
        except FileNotFoundError as e:
            print("flagging: ", e)
            self.controllers = None
            self.algos = None

        self.noise_model = structured_perturbation(**self.args)
        self.parallel = parallel
        self.num_workers = num_workers
        self.colors = ["blue", "orange", "gold", "purple", "pink", "brown", 
                        "red", "cyan", "gray", "mediumseagreen", "olive"]
        self.figlabels = ["({})".format(i) for i in "abcdefghijklmnopqrstuvwxyz"]   # PRA customs

    def set_fig_save_directory(self, cur_save_folder):
        self.cur_save_folder = cur_save_folder
        if not os.path.exists(cur_save_folder):
            os.mkdir(cur_save_folder)

    def get_all_algo_controllers(self):
        "combine all algo controllers"
        cs = []
        for alg in list(self.controllers.keys()):
            if alg=="lbfgs":
                conts = np.array(self.controllers[alg][str(self.Nspin)]["controller"])
                if self.numcontrollers-len(conts) > 0:
                    conts = np.pad(conts, [(self.numcontrollers-len(conts),0), (0,0)])
                cs.append(conts)
            else:   
                for noise in list(self.controllers[alg].keys()):
                    cs.append(np.array(self.controllers[alg][noise]["controller"]))
        
        cs = np.array(cs).reshape(-1, self.Nspin+1) # ambiguate controller algos for now
        return cs

    @staticmethod 
    def bootstrap_resampling_std(summarystatistic, l, bootsamples):
        bootsss = np.zeros(bootsamples)
        for i in range(bootsamples):
            randi = np.random.randint(0, len(l), size=len(l))
            bootstrappedl = l[randi]
            bootss = summarystatistic(bootstrappedl)
            bootsss[i] = bootss
        return bootsss.std()

    def tsneconts(self):
        names2nkeys = []
        for alg in list(self.controllers.keys()):
            for noise in list(self.controllers[alg].keys()):
                names2nkeys.append((alg,noise))
        if not os.path.exists(self.get_controller_name+".tsne"):
            cs = self.get_all_algo_controllers()
            X_embedded = TSNE(n_components=2, perplexity=50, n_iter=500).fit_transform(cs)
            algs = len(cs)
            X_embedded = X_embedded.reshape(algs,-1, 2)
            json.dump(X_embedded.tolist(), open(self.get_controller_name+".tsne", "w"))
        else:
            X_embedded = np.array(self.loadsimdata(self.get_controller_name+".tsne"))
            algs = len(X_embedded)
        plt.figure()
        # noise_keys = list(self.controllers["ppo"].keys())
        for alg in range(algs):
            if alg == algs-1: # lbfgs case
                algoname=names2nkeys[alg][0]
                nkey="0.00"
                wd_data = self.get_metrics_dict(None, self.noises, algoname="lbfgs")["lbfgs"]
                # wd_data_c = wd_data[r'$W(.,\delta(x-1))$']
                # print(np.isnan(np.array(wd_data_c)[0]))
                # print(self.get_ranks(wd_data_c[0]) <= self.topk-1)
            else:
                algoname=names2nkeys[alg][0]
                nkey = names2nkeys[alg][1]
                if float(nkey) > 0.06:
                    continue
                wd_data = self.get_metrics_dict(nkey, self.noises, algoname=algoname)[algoname]
            wd_data_c = wd_data[r'$W(.,\delta(x-1))$']
            topk_idx = self.get_ranks(wd_data_c[0]) <= self.topk-1 #self.get_top_k_by_fid_idx(wd_data_c, self.topk) 
            
            plt.scatter(X_embedded[alg][:,0][topk_idx],X_embedded[alg][:,1][topk_idx], 
                        label=algoname+" "+r"$\sigma_{\rm{train}}$="+str(nkey) if alg!=algs-1 else "lbfgs", 
                        color="k" if alg==algs-1 else None, marker=r"${}$".format(algoname[0]),
                        alpha=0.5, s=100)
        plt.legend()
        raise AssertionError
        
    def get_wd_data_c(self ):
        "TODO: requires generalization to other algos"
        noise_keys = list(self.controllers["ppo"].keys())
        algs = len(noise_keys)+1 # algo count, can generalize later
        # for alg in list(self.controllers.keys()):
        #     algs += len(list(self.controllers[alg].keys()))
        all_wd_data_c = []
        for alg in range(algs):
            if alg == 11:
                wd_data = self.get_metrics_dict(None, self.noises, algoname="lbfgs")["lbfgs"]
            else:
                wd_data = self.get_metrics_dict(noise_keys[alg], self.noises, algoname="ppo")["ppo"]
            wd_data_c = wd_data[r'$W(.,\delta(x-1))$']
            wd_data_c = np.array(wd_data_c)
            if self.topk:
                wd_data_c_idx = self.get_top_k_by_fid_idx(wd_data_c, self.topk) 
                wd_data_c = np.array(wd_data_c)[wd_data_c_idx]
            all_wd_data_c.append(wd_data_c)
        return all_wd_data_c

    def ctrlnames(self, ctrlcontainer) -> List:
        if isinstance(ctrlcontainer, dict):
            prospective_keys = list(ctrlcontainer.keys())
            # purge empty containers
            for key in prospective_keys:
                if ctrlcontainer[key] == {}:
                   ctrlcontainer.pop(key)
            return list(ctrlcontainer.keys())
                    
        elif isinstance(ctrlcontainer, list) or isinstance(ctrlcontainer, np.ndarray):
            return ["unnamed"]
        else:
            raise TypeError("need controller container either as a list or a dict")
        
    def get_mcname(self, training_noise = None, noises = None) -> str:
        if training_noise is None:
            training_noise = self.training_noise
        if noises is None:
            noises = self.noises
        return self.get_controller_name+"_tn{}_br_{}_nlvl{}.mc".format(training_noise, self.bootreps, noises)
        
    def load_controllers(self, controllers = None):
        if controllers is None:
            return json.load(open(self.get_controller_name, "rb"))
        elif isinstance(controllers, str):
            return json.load(open(controllers, "rb"))
        elif isinstance(controllers, list) or isinstance(controllers, np.ndarray):
            return controllers
    
    def loadsimdata(self, simname: str):
        return json.load(open(simname, "rb"))
    
    def get_controller_fid_dist_boot(self, x=None):
        if self.controller is not np.nan:
            return self.noise_model.evaluate_noisy_fidelity(
                        self.controller, ham_noisy=True) # bootstrap
        else:
            return np.nan
        
    def get_experiment_name(self, experiment_name: str) -> Callable[[str], ExperimentNamer]:
        return ExperimentNamer(experiment_name=experiment_name,
                               numcontrollers=self.numcontrollers, 
                               **self.args,)


    def get_fid_dists(self, training_noise: str = None, noises: np.ndarray = None, 
                      algoname = None)->dict:
        
        if isinstance(algoname, str):
            algos = [algoname]
        elif algoname is None:
            algos = self.algos
        
        if noises is None:
            noises = self.noises
        
        if training_noise is None:
            training_noise = self.training_noise
            
        if os.path.exists(self.get_mcname(training_noise, noises)):
            simdict = self.loadsimdata(self.get_mcname(training_noise, noises))
            for algoname in algos:
                if algoname not in simdict:
                    self.get_algo_fid_dist(algoname, simdict, noises, training_noise)
            
            for algoname in simdict.keys():
                if algoname not in algos:
                    raise Exception(f"Fid distribution generation for {algoname} was unsuccessful.")
                
            return simdict
        else:
            # do mc sims from scratch and return fid distributions of shape ((noise_res, controllers, bootreps))
            allalgoallfids = {}
            for algoname in algos:
                if algoname=="lbfgs":
                    training_noise=None
                self.get_algo_fid_dist(algoname, allalgoallfids, noises, training_noise)
                
            for algoname in allalgoallfids.keys():
                if algoname not in algos:
                    raise Exception(f"Fid distribution generation for {algoname} was unsuccessful.")
    
            return allalgoallfids
                

    def get_algo_fid_dist(self, algoname: str,allalgoallfids: dict, noises, training_noise):
        allfids = np.zeros((noises.size, self.numcontrollers, self.bootreps))
        for j, noise in enumerate(tqdm(noises[:])):
            self.noise_model.rng(scale=noise) # change sim noise lvl
            if algoname!="lbfgs":
                print(algoname, training_noise)

                num_ctrls = len(self.controllers[algoname][str(training_noise)]["controller"])
            else:
                print(algoname, training_noise)
                num_ctrls = len(self.controllers[algoname][str(self.Nspin)]["controller"])

            for ctrli in range(self.numcontrollers):
                # some design quirks coming back to bite 
                if ctrli < num_ctrls:
                    if algoname!="lbfgs":

                        controller = self.controllers[algoname][str(training_noise)]["controller"][ctrli]
                    else:
                        controller = self.controllers[algoname][str(self.Nspin)]["controller"][ctrli]
                else:
                    controller = np.nan
                
                self.controller = controller
                if not self.parallel:
                    fids=np.zeros(self.bootreps)
                    for i in range(self.bootreps):
                        fids[i] = self.get_controller_fid_dist_boot()
                    
                else:
                    # parallelize the matrix exponentials: doesn't seem very fastL TODO: fix this!
                    pool = Pool() if self.num_workers is None else Pool(processes=self.num_workers)
                    fids = pool.map(self.get_controller_fid_dist_boot, range(self.bootreps))
                    pool.close()
                allfids[j][ctrli] = fids
        allalgoallfids[algoname] = allfids.tolist()
        
        json.dump(allalgoallfids, open(self.get_mcname(training_noise, noises), "w"))
        return allalgoallfids


    def get_metrics_dict(self, training_noise: str = None, 
                         noises: np.ndarray = None, algoname=None):
        "generate a dict of tuples with wd, wc, (qt_i) metrics for specific algos and save it aptly"
        if training_noise is None:
            training_noise = self.training_noise
        if noises is None:
            noises = self.noises
        
        
        if isinstance(algoname, str):
            algos = [algoname]
        elif algoname is None:
            algos = self.algos
            
        def get_metric_dict_from_scratch(algos, algoname):
            algofiddists = self.get_fid_dists(training_noise, noises, algoname)
            allalgos_metrics_dict = {}
            for algo in algos:
                metrics_dict = {}
                dists_tensor = np.array(algofiddists[algo]) # shape: (noise_res, numcontrollers, bootreps)
                dkw_error = compute_dkw_error(self.alpha, self.bootreps) #  / np.sqrt(self.numcontrollers) does this make sense? need to think...
                dists_tensor_lower = np.clip(dists_tensor+dkw_error, 0, 1) # following convention ideal closer to 1 so must be an upper error
                dists_tensor_upper = np.clip(dists_tensor-dkw_error, 0, 1)
                
                for metric_name in __metric_name_to_metric__:
                    metric_func = __metric_name_to_metric__[metric_name]
                    allnoise_lvlsdata = []
                    allnoise_lvlsdata_upper = []
                    allnoise_lvlsdata_lower = []
                    for noise in range(noises.size):
                        allnoise_lvlsdata.append(list(metric_func(dists_tensor[noise])))
                        allnoise_lvlsdata_upper.append(list(metric_func(dists_tensor_upper[noise])))
                        allnoise_lvlsdata_lower.append(list(metric_func(dists_tensor_lower[noise])))
                    metrics_dict[metric_name] = allnoise_lvlsdata
                    metrics_dict[metric_name+" upper"] = allnoise_lvlsdata_upper
                    metrics_dict[metric_name+" lower"] = allnoise_lvlsdata_lower
                
                allalgos_metrics_dict[algo] = metrics_dict
            json.dump(allalgos_metrics_dict, open(self.get_mcname(training_noise, noises)+"m", "w"))
            return allalgos_metrics_dict
        
        if os.path.exists(self.get_mcname(training_noise, noises)+"m"):
            metric_dict = self.loadsimdata(self.get_mcname(training_noise, noises)+"m")
            return metric_dict
                
        else:
            allalgos_metrics_dict = get_metric_dict_from_scratch(algos=self.algos, algoname=None)
            return allalgos_metrics_dict
        
        
    @staticmethod
    def get_ranks(array):
        argranks = np.argsort(array) # ranked controllers
        ranks = np.zeros_like(argranks)  # small wd has the lowest rank. rank 0 is best for a noise axis
        ranks[argranks] = np.arange(len(argranks))
        return ranks
        
    def get_best_controller_perf(self, metric_data, algo=None, contcount=None):
        "assume metric is best when small"
        if contcount is None:
            contcount = self.numcontrollers
        argranks = np.argsort(metric_data, axis=1) # ranked controllers
        ranks = np.zeros_like(argranks)  # small wd has the lowest rank. rank 1 is best for a noise axis
        for i in range(argranks.shape[0]):
            ranks[i][argranks[i]] = np.arange(argranks.shape[-1])
        assert metric_data[-1][np.argmin(ranks[-1])]==np.min(metric_data[-1]), "rank order needs to be metric ascending"
        best_across_plot_noises = ranks.sum(axis=0) # rank sum
        # print(best_across_plot_noises)
        try: # soften as a flag for paper plot for now
            assert best_across_plot_noises.size == contcount, "summation axis is incorrect!"
        except:
            print("summation axis is incorrect!")

        bests_nranks = np.argsort(best_across_plot_noises)
        best_controller_index = bests_nranks[0] # 
        median_controller_index = bests_nranks[metric_data.shape[-1]//2]
        best_per_noise = np.min(metric_data, axis=1)
        best_controller_per_noise = metric_data[:, best_controller_index]
        median_controller_per_noise = metric_data[:,median_controller_index]
        assert best_controller_per_noise.size == best_per_noise.size == metric_data.shape[0], "sim noise vector shape consistency check violated"
        diff_median = (median_controller_per_noise - best_per_noise )
        diff = (best_controller_per_noise - best_per_noise )
        return diff, diff_median, best_controller_per_noise, median_controller_per_noise, best_per_noise
        
    
    def get_top_k_by_fid_idx(self, wd_data_c, topk, idx=0):
        filmask = self.get_ranks(wd_data_c[idx]) <= topk-1
        idx = np.ix_(np.ones(wd_data_c.shape[0], dtype=bool), filmask)
        return idx
    
    def save_fig(self, fig, name="noiseless_comp", pltrows=None, pltcols=None, copyto=None, keepsimple=False):
        cname = self.get_controller_name.split("/")[-1]
        if not keepsimple:
            fname = f"{self.cur_save_folder}/{name}_c{pltcols}_r{pltrows}_{self.Nspin}_-{self.outspin}.pdf"
        else:
            fname = f"{self.cur_save_folder}/{name}.pdf"
        fig.savefig(fname, dpi=1000, bbox_inches="tight")
        if copyto:
            import shutil
            shutil.copy(fname, copyto)
        return fname
        
        
    @staticmethod  
    def sort_fids_by(fids: np.ndarray, by_metric: np.ndarray, best_k: int=100):
        "in increasing order of `by_metric`"
        return fids[np.argsort(by_metric, axis=-1)[:best_k]]
    
    def get_path(self, directory_exportable, of:str ="controllers"):
        
        rootpath=self.global_experiments_directory+directory_exportable
        print(rootpath)
        if not os.path.exists(rootpath):
            raise DirectoryDoesNotExistError(self.global_experiments_directory)
        
        controller_dict_path = self.get_experiment_name(directory_exportable)()
        print(controller_dict_path)
        if self.filemarker is not None:
            controller_dict_path += self.filemarker
        if not os.path.exists(controller_dict_path):
            raise DirectoryDoesNotExistError(controller_dict_path)
        
        if of=="controllers":
            return controller_dict_path
        elif of == "mcm": # mc metrics dict:
            return glob.glob(controller_dict_path+"**.mcm")
        elif of == "mc": # mc fid dists dict
            return glob.glob(controller_dict_path+"**.mc") 
        else:
            raise Exception("No such object type exists. Please specify a correct .description.")
    
    def merge_mcdata(self, directory_exportable):
        local_path = self.experiment_name
        exportable_path = self.global_experiments_directory + directory_exportable
        currfidpaths = self.get_path(local_path, of="mc")
        currmetricpaths = self.get_path(local_path, of="mcm")

        for currfidpath, currmetricpath in zip(currfidpaths, currmetricpaths):
            currfiddata = self.loadsimdata(currfidpath)
            currmetdata = self.loadsimdata(currmetricpath)
            
            
            fiddata_path = exportable_path+"/"+currfidpath.split("/")[-1]
            metdata_path = exportable_path+"/"+currmetricpath.split("/")[-1]
            
            fiddata = self.loadsimdata(fiddata_path)
            metricdata = self.loadsimdata(metdata_path)
            for algo in fiddata:
                if algo not in currfiddata:
                    currfiddata[algo]=fiddata[algo]
            for algo in metricdata:
                if algo not in currmetdata:
                    currmetdata[algo]=metricdata[algo]

            print(currfiddata.keys())
            print(fiddata.keys())
            json.dump(currmetdata, open(currfidpath, "w"))
            json.dump(currfiddata, open(currmetricpath, "w"))
        print("files successfully merged")

    def load_controllers_in_dir(self, directory_exportable):
        controller_dict_path = self.get_path(directory_exportable, of="controllers")        
        alt_controllers = self.load_controllers(controller_dict_path)
        return alt_controllers
    
    def merge_controller_files(self, directory_exportable:str)->None:
        "file names must be identical but located in a different `directory_exportable`"
        alt_controllers = self.load_controllers_in_dir(directory_exportable)
        algos = self.ctrlnames(alt_controllers)
        for algo in algos:
            if algo == "lbfgs":
                # this is training without noise keys so by default is a special case
                if algo not in self.controllers:
                    self.controllers[algo] = alt_controllers[algo]
            else:
                # top level just add all the noises for this algo in cont_dict
                if algo not in self.controllers:
                    self.controllers[algo] = alt_controllers[algo]
                else:
                    # add only the noise diffs in the original data structure
                    noise_keys_alt = list(alt_controllers[algo].keys())
                    for noise in noise_keys_alt:
                        if noise not in self.controllers[algo]:
                            self.controllers[algo][noise] = alt_controllers[algo][noise]

        # save in og controller directory
        json.dump(self.controllers, open(self.get_controller_name, "w"))
        
    def get_top_k_by_fid(self, wd_data_c, wd_data_u, wd_data_l, topk, fid_thres=0.8):
        filmask = self.get_ranks(wd_data_c[0]) <= topk-1
        if fid_thres:
            filmask &= wd_data_c[0] <= 1-fid_thres
        idx = np.ix_(np.ones(wd_data_c.shape[0], dtype=bool), filmask)
        wd_data_c = np.array(wd_data_c)[idx] 
        wd_data_u = np.array(wd_data_u)[idx]
        wd_data_l = np.array(wd_data_l)[idx]
        
        return wd_data_c, wd_data_u, wd_data_l
        
 