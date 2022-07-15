import numpy as np
from wd_sortof_fast_implementation import wd_from_ideal, dkw_ecdf_bounds
import json
import matplotlib.pyplot as plt
from qnewton import LBFGS
from tqdm import tqdm
import os
import seaborn as sns
sns.set()

def Q(fid_array, threshold):
    return len(fid_array[fid_array >= threshold]) / len(fid_array)

def get_sd_results(spin: int =5, inspin: int =0, outspin: int =2, 
                    bootreps: int = 100, 
                    rlc_index: str = None, 
                    noises=np.linspace(0,1,11)):
    
    """
    Generate example area under the cdf average interpretation figures
    """
    # remove 0
    if abs(noises[0]-0) < 1e-7:
        noises = noises[1:]
    REPS = bootreps# boostrap reps
    CONTROLLERS=100
    results = json.load(open(f"noisy_analysis/lbfgs_spin_{spin}_{inspin}-{outspin}_in", "rb"))
    results2 = json.load(open(f"noisy_analysis/ppo_spin_{spin}_{inspin}-{outspin}_in", "rb"))
    
    
    assert len(results["lbfgs"].keys()) != 0 , "make sure you have the right qnewton file"
    
    lbfgs_controllers = results["lbfgs"]
    
    # results2 = json.load(open(f2, "rb"))
    
    ppo_controllers = results2["ppo"]
    keys = list(ppo_controllers.keys())
    
    
    if not rlc_index:
        rlc_index = keys[1] if spin!=6 else keys[0]  # which controllers to look at
    
    print([len(ppo_controllers[i]["controller"]) for i in keys])
    
    env = LBFGS(spin, inspin, outspin)
    
    print(f"file load: spin {spin} {inspin} -> {outspin} ==> all ok")
    
    
    allfidsl = np.zeros((len(noises),CONTROLLERS, REPS))
    allfidsp = np.zeros((len(noises),CONTROLLERS, REPS))
    
    
    for j, noise in enumerate(tqdm(noises)):
        print(noise)
        fidsl=[]; fidsp = [];
            
        for controller in range(CONTROLLERS):
            env.noise = noise
            fidsl=np.zeros(REPS); fidsp = np.zeros(REPS);
            for i in range(REPS):

                fidsl[i] = (env.fidelity_ss(lbfgs_controllers[str(spin)]["controller"][controller], ham_noisy=True)) # bootstrap
                if controller < len(ppo_controllers[rlc_index]["controller"]):
                    fidsp[i] = (env.fidelity_ss(ppo_controllers[rlc_index]["controller"][controller], ham_noisy=True))
                else:
                    fidsp[i]=np.nan

            allfidsl[j][controller] = fidsl
            allfidsp[j][controller] = fidsp

            l_sortedi = np.argsort(fidsl)
            p_sortedi = np.argsort(fidsp)
            combined = np.concatenate((fidsl, fidsp))
            combined.sort(kind="quicksort")
            
            sortedfidsl = fidsl[l_sortedi]
            sortedfidsp = fidsp[p_sortedi]
            c_fd = sortedfidsl.searchsorted(combined[:-1], side="right") / fidsl.size
            c_nfd = sortedfidsp.searchsorted(combined[:-1], side="right") / fidsp.size

            discrete_cdfl = c_fd
            discrete_cdfp = c_nfd
            
            intervals = np.arange(c_fd.size) / c_fd.size # [1/N, ..., (N-1)/N]
            discrete_cdfll, discrete_cdflu = dkw_ecdf_bounds(discrete_cdfl,conf_level=0.95)
            discrete_cdfpl, discrete_cdfpu = dkw_ecdf_bounds(discrete_cdfp,conf_level=0.95)
            # plt.style.use('grayscale')
            plt.figure(figsize=(10,10))
            rep1 = "; RIM={}"
            plt.plot(intervals,discrete_cdfl, 
label="$P^{(1)}_"+"{"+str(noise)+"}"+"(\mathcal{F}_1)$" + rep1.format(
round(wd_from_ideal(fidsl), 3)), linewidth=4, color="orange")
            delta = np.zeros_like(intervals)
            delta[-1] = 1
            rep2 = "; RIM=0"
            plt.plot(intervals, delta , color="green", 
label=r"$P^{(\delta)}_"+"{"+str(noise)+"}"+"(\mathcal{F}_{\delta_1})$" + rep2, linewidth=4, linestyle="-.")
            
           
            plt.plot(intervals,discrete_cdfp, 
label="$P^{(2)}_"+"{"+str(noise)+"}"+"(\mathcal{F}_2)$" + rep1.format(round(wd_from_ideal(fidsp), 3)), 
linewidth=4, color="blue")
            
            plt.fill_between(intervals, discrete_cdfll, discrete_cdflu, color="orange", alpha=0.5)
            plt.fill_between(intervals, discrete_cdfpl, discrete_cdfpu, color="blue", alpha=0.5)
            plt.legend(fontsize=30, loc="upper right")
            plt.xlim(0,1+0.01)
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            plt.ylabel(r"$P_" +"{"+str(noise)+"}"+"(\mathcal{F} \leq x)$", fontsize=30)
            plt.xlabel(r"$x$", fontsize=30)
            if not os.path.exists("example_cdf_area_figs"):
                os.mkdir("example_cdf_area_figs")
            plt.savefig("example_cdf_area_figs/examplefig_Ver2{}.pdf".format(np.random.randint(0, int(1e9))), dpi=800)
                                

if __name__ == '__main__':
    br=100
    qthresholds=0.95
    sns.set()
    x = get_sd_results(bootreps=br, outspin=2, spin=5, noises=[0.1])