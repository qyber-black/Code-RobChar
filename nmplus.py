import numpy as np
from qnewton import LBFGS
from typing import Tuple, List
import math
from scipy.stats import qmc

class NMPlus(LBFGS):
    "Nelder Mead standard and Accelerated Nelder-Mead B: with modifications"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_dim = self.Nspin+1
        self.isimp = self.init_simplex()
        self.alpha = 1
        self.beta = 2
        self.gamma = 0.5
        self.delta = 0.5
        self.planar_reflection=False
        

    def init_simplex(self, sampler=None):
        init_sm = np.zeros((self.x_dim+1, self.x_dim)) # s: (cont_guesses, cont_dim)
        for i in range(self.x_dim+1):
            for j in range(self.x_dim):
                # init differently for the time axis taken conventionally as -1th column
                if j == self.x_dim-1:
                    rng = self.rng(self.Tmin, self.Tmax, size=1, sampler=sampler)
                else:
                    rng = self.rng(self.Bmin, self.Bmax, size=1, sampler=sampler)
                if i==j+1 and i>0:
                    init_sm[i][j] = rng*(
                                            np.sqrt(self.x_dim+1)+self.x_dim-1)/np.sqrt(self.x_dim)
                elif i>0:
                    init_sm[i][j] = rng*(
                                        np.sqrt(self.x_dim+1)-1)/np.sqrt(self.x_dim)
        assert np.alltrue(init_sm[:,-1]>=0), "initial time guesses are not positive!"
        return init_sm
    
    def rng(self, low: float, high: float, size:Tuple, sampler=None) -> np.float64:
        "id. or random uniform pseudo number generator"
        if sampler and self.landscape_exploration:
            assert size == 1, "id-sampler configured for scalars only"
            x0 = sampler.random()[0]
            x0 = low + (high-low)*x0  
            return x0
        else:
            return np.random.uniform(low=low, high=high, size=size)

    def infidelity(self, x):
        if not self.use_fixed_ham:
            return 1-self.fidelity_ss(x, noisy=self.fid_noisy, ham_noisy=self.ham_noisy)
        else:
            return 1-self.fidelity_ss_av(x, noisy=self.fid_noisy, ham_noisy=self.ham_noisy, reps=self.train_size)

    @staticmethod
    def powell(x):
        "benchmark 1"
        return (((x[:-1]+x[1:])**2).sum() + 
                (5*(x[2:-1]-x[3:])**2).sum() + 
                ((x[1:-1]-2*x[2:])**4).sum()+ 
                (10*(x[:-3]-x[3:])**4).sum())
    @staticmethod
    def f(x):
        "benchmark 2"
        return math.sin(x[0]) * math.cos(x[1]) * (1./(abs(x[2]) + 2))

    def sort_simplex(self, simplex: np.ndarray, obj_f=None):
        "sort simplex points by fidelity evaluation"
        if obj_f is None:
            obj_f = self.infidelity
        infidelities = list(map(obj_f, simplex))
        sort_order = np.argsort(infidelities)
        infidelities.sort()
        return simplex[sort_order], infidelities
    

    def estimate_hyperplane(self, sorted_simplex: np.ndarray, infidelities: List):
        "get hyperplane coefficients for the simplex"
        X = np.ones((self.x_dim+1, self.x_dim+1))
        X[:,1:] = sorted_simplex
        # print("X \n",X)
        Y = infidelities
        # print("Y \n", Y)
        G = np.linalg.inv(X) @ Y 
        return G[1:]

    def update_simplex(self, sorted_simplex, infidelities, obj_f=None):
        if obj_f is None:
            obj_f = self.infidelity
        if self.planar_reflection:
            # reflection using the plane
            G = self.estimate_hyperplane(sorted_simplex, infidelities)
            rp = sorted_simplex[0] - self.alpha*G
            G = sorted_simplex[0]
        else:
            # using centroid
            G = sorted_simplex[:-1].mean(axis=0)
            rp = (1+self.alpha)*G - self.alpha*sorted_simplex[-1]
        if_rp = obj_f(rp)
        if_1 = infidelities[0]
        if_p = infidelities[-2]

        # case 1a: f_1 < f_r < f_p
        if if_1 <= if_rp < if_p:
            sorted_simplex[-1] = rp 
            infidelities[-1] = if_rp 
            
        
        # case 1b: f_1 > f_r
        elif if_rp < if_p and if_1 > if_rp:
            # expansion
            ep = (1-self.gamma)*G + self.gamma*rp
            if_ep = obj_f(ep) 
            # case 1bi: f_e <= f_r
            if if_ep < if_rp:
                sorted_simplex[-1] = ep 
                infidelities[-1] = if_ep
                 
            # case 1bii: f_e > f_r
            else:
                sorted_simplex[-1] = rp 
                infidelities[-1] = if_rp
                
        # case 1c: f_r > f_p
        elif if_rp >= if_p:
            # case 1ci: f_p < f_r < f_p+1
            if if_p <= if_rp < infidelities[-1]:
                # contraction outside
                cp = (1-self.beta)*G + self.beta*rp
                if_cp = obj_f(cp)
                # case 1cia: f_c < f_r 
                if if_cp <= if_rp:
                    sorted_simplex[-1] = cp 
                    infidelities[-1] = if_cp
                # case 1cib: f_c > f_r
                else: # shrink simplex
                    sorted_simplex[1:] = (1-self.delta)*np.tile(sorted_simplex[0], (self.x_dim,1)) + self.delta*sorted_simplex[1:]
            # case 1cii: f_r > f_p+1
            elif if_rp >= infidelities[-1]:
                # contraction inside
                cp = (1+self.beta)*G - self.beta*rp
                if_cp = obj_f(cp)
                # case 1ciia: f_c < f_r 
                if if_cp <= if_rp:
                    sorted_simplex[-1] = cp 
                    infidelities[-1] = if_cp
                else: # shrink simplex
                    sorted_simplex[1:] = (1-self.delta)*np.tile(sorted_simplex[0], (self.x_dim,1)) + self.delta*sorted_simplex[1:]

        return sorted_simplex, infidelities


    def _run(self, iterations, simplex=None, obj_f=None, improv_thres=1e-6):
        "in-house version: slower and accelerated version is a bit buggy still"
        init_simp = None
        if simplex is None:
            simplex = self.isimp
            init_simp = self.init_simplex
        else:
            def rng():
                return np.random.uniform(size=(self.x_dim+1,self.x_dim))
            init_simp = rng
        inf_best=np.inf
        prev_best = None; improv=0; max_tries=30
        tries = 0
        for i in range(iterations):
            if improv < improv_thres and tries < max_tries:
                tries += 1
            if improv < improv_thres and tries >= max_tries:
                simplex = init_simp()
                print("restarting simplex")
                tries = 0
            simplex, infidelities = self.sort_simplex(simplex, obj_f=obj_f)
            
            # try:
            simplex, infidelities = self.update_simplex(simplex, infidelities, obj_f=obj_f)
            # update the improvement
            if prev_best is None:
                improv = infidelities[0]
            else:
                improv = prev_best - infidelities[0]
            prev_best = infidelities[0]

            if infidelities[0] < inf_best:
                current_best = simplex[0]
                inf_best = infidelities[0]
            print(f"it {i} curr best {inf_best}", "\n", infidelities)
            # except:
            #     simplex = np.random.uniform(size=(5+1,5)) # self.init_simplex()
        return inf_best, current_best
    
    def run(self):
        "scipy nelder-mead: consistent with the overarching api from LBFGS"
        import time as tt
        from scipy.optimize import minimize
        funccalls = 0
        iters = 0
        start_time = tt.time()
        max_fid_seen = 0
        true = 0
        run_until_completion_criterion=False
        running_controllers = {}

        # if idsampling:
        #     sampler = qmc.Sobol(d=self.Nspin+1, scramble=False)
        if self.landscape_exploration:
            # sampler = qmc.Sobol(d=1, scramble=False)
            sampler = qmc.Sobol(d=self.Nspin+1, scramble=False)
        initx0 = None
        result=None
        for rep in range(self.repeats):

            if self.use_fixed_ham:
                fev=300
            else:
                fev=300
        
            if self.landscape_exploration:
                x0 = sampler.random()[0]
            else:
                x0 = np.random.rand(self.Nspin+1) 
                
            x0[0:self.Nspin] = self.Bmin + (self.Bmax-self.Bmin) * np.array(x0[0:self.Nspin])
            x0[self.Nspin] = self.Tmin + (self.Tmax-self.Tmin) * np.array(x0[self.Nspin])
            x = minimize(self.infidelity, x0=x0, 
                         options={'disp': False,
                                  #'initial_simplex': self.init_simplex(sampler=sampler),
                                  'maxfev': fev},
                         method='Nelder-Mead', bounds=self.val_bounds)
            if self.use_fixed_ham:    
                fi = 1-x.fun
                true_fid = 1-x.fun
            else:
                fi = self.fidelity_ss(x.x, noisy=self.fid_noisy, ham_noisy=self.ham_noisy)
                true_fid = self.fidelity_ss(x.x)
            

            if self.verbose:
                if max_fid_seen < fi:
                    max_fid_seen = fi
                    if self.use_fixed_ham:
                        true = None # self.fidelity_ss_av(x.x, noisy=self.fid_noisy, ham_noisy=self.ham_noisy, test=True)
                    else:
                        true = self.fidelity_ss(x.x)
                    # temp = self.noise 
                    # self.noise=0.1
                    # wd = self.wass_cost(x, bootstrap_reps=1000)
                    # self.noise=temp
                print(f"max_fid: {max_fid_seen}, true fid: {true} funccalls: {funccalls}")
                # print(f"wd: {wd}")

            if self.use_fixed_ham:
                funccalls += x.nfev*self.train_size
                iters += x.nit*self.train_size
            else:
                funccalls += x.nfev
                iters += x.nit
            
            
            def save_controller_data_aux(): # auxilliary routine to save space
                self.record["time_to_get_fid"] = tt.time()-start_time
                self.record["func_calls"] = funccalls
                self.record["iterations"] = iters
                self.record["repeats"] = rep 
                self.record["controller"] = x.x.tolist()
                # print("wd final soln: ", self.Monte_env.wass_cost(next_store, 1000))
                if self.landscape_exploration:
                    self.record["controllers"] = list(running_controllers.values())
                    if self.records_update_rate:
                        self.record_collector(funccalls, self.record["controllers"])
                if self.ham_noisy or self.fid_noisy:
                    self.record["best_fid"] = true_fid 
                else:
                    self.record["best_fid"] = fi
                    
            if not self.run_until_told_to_stop:  
                if fi > self.fid_threshold:
                    save_controller_data_aux()
                    if self.save:
                        self.save_record()
                    # print(self.record)
                    return fi
            
            else:
                 # update current best until time out
                if self.record["best_fid"] is None:
                    crit = fi >= self.fid_threshold
                else:
                    crit = fi >= self.record["best_fid"]
                    if self.landscape_exploration:
                        crit = True # is a tautology for max updating
                if crit:
                    if self.landscape_exploration:
                        l=len(list(running_controllers.keys()))
                        if l < self.save_topc:
                            running_controllers[fi]=x.x.tolist()
                            # print("running_list: \n", running_controller_list)
                        else:
                            #itopop=self.find_min_fid_index(running_controller_list) # time to pop this ###
                            itopop=min(list(running_controllers.keys()))
                            running_controllers.pop(itopop)
                            running_controllers[fi]=x.x.tolist() # maintain const size list
                            # print("running_list: \n", running_controllers)
                            
                    save_controller_data_aux()   
                if run_until_completion_criterion:
                    return self.record["best_fid"]
                if tt.time()-start_time > self.timeout: # relegated to a fail-safe (extremely unlikely e.g. don't want to wait all day for 1 run)
                    print(f"timed out! {self.filename}")
                    raise AssertionError("timeout")
                
                # run for a fixed number of iterations and then terminate
                run_until_completion_criterion = funccalls+1 >= self.run_until_completion_its
            
            if tt.time()-start_time > self.timeout:
                print(f"timed out! {self.filename}")
                raise AssertionError("timeout")
            elif funccalls > self.fun_call_limit:
                print("fun ceiling exceeded %s" %self.fun_call_limit)
                return

if __name__ == '__main__':
    # algo = NMPlus(7,0,6)
    # algo.x_dim = 5
    # algo.planar_reflection=True
    # print(algo.isimp)
    # print(algo._run(10000, simplex=np.random.uniform(size=(6,5)), obj_f=algo.powell))
# =============================================================================
#     TODO still broken, 
#     Relegated to a NOTE: planar reflection is hit and miss on some problems, 
#     especially quantum. still broken
# =============================================================================
    algo = NMPlus(5, 0, 2, -10, 10, 70, fid_noisy=False, draws=100, repeats=100000, 
                  fid_threshold=0.0, ham_noisy=True, verbose=True, adaptive=False, 
                  adp_tol=0.05, noise=0.05, run_until_completion_its=6000000, run_until_told_to_stop=True,
                  landscape_exploration=True, save_topc=10, use_fixed_ham=True, opt_train_size=100)
    algo.run()
