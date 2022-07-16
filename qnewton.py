
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 08:15:03 2021

@author: irtazakhalid
"""


import numpy as np
from scipy import optimize
import scipy as sp
import json
import logging
import time as tt
from scipy.stats import wasserstein_distance, qmc
from wd_sortof_fast_implementation import wd_from_ideal
import torch.nn as nn
from torch.optim import Adam, RMSprop, SGD
import torch
import skquant.opt as skq
from SQSnobFit import optset


class LBFGS(object):
       
    def __init__(self, nspin, in_spin, out_spin, bmin=-10, bmax=10, max_time=30, 
                 repeats=1000000, fid_threshold=0.98, log=False, topo="linear", 
                 save=False, noisy=False, timeout=1800000, fid_noisy=False, draws=10, 
                 ham_noisy=False, verbose=False, adp_tol=0.05, adaptive=False, 
                 noise=0.05, use_wass_cost=False, testing= None, 
                 run_until_told_to_stop=None, 
                 run_until_completion_its=None, 
                 landscape_exploration: bool = False,
                 save_topc: int = 1000, heisenberg_int: bool = False, 
                 use_fixed_ham: bool = False, opt_train_size: int = 100,
                 records_update_rate: float = None, 
                 ):
        "save_topc and runs are both redundant when landscape exploration is enabled!"
        self.landscape_exploration = landscape_exploration
        self.save_topc = save_topc
        
        self.topo=topo
        self.heisenberg_int = heisenberg_int
        self.Nspin = nspin
        self.In = in_spin
        self.Out = out_spin
        self.Tmin = 0
        self.Tmax = max_time
        self.Bmin = bmin
        self.Bmax = bmax
        self.repeats = repeats
        self.HH = self.sys_hamiltonian()
        # print(self.HH, np.diag(self.HH)+1)
        # print(np.linalg.eig(np.array(self.HH)+np.diag(np.diag(self.HH)+1)))
        # raise AssertionError
        self.CC = self.controls()
        self.fid_threshold = fid_threshold
        self.draws = draws
        self.ham_noisy = ham_noisy
        self.fid_noisy=fid_noisy
        self.timeout = timeout
        self.verbose = verbose
        self.adp_tol = adp_tol
        self.adaptive = adaptive
        self.adp_func_calls_increment = self.draws
        self.noise = noise
        self.fun_call_limit = 1e10
        self.use_wass_cost = use_wass_cost
        self.run_until_told_to_stop=run_until_told_to_stop
        self.run_until_completion_its=run_until_completion_its

        self.rho0 = np.zeros((self.Nspin,self.Nspin))
        self.rho0[self.In,self.In] = 1
        self.rho1 = np.zeros((self.Nspin,self.Nspin))
        self.rho1[self.Out,self.Out] = 1
        
        assert self.Tmax >= self.Tmin, "Tmin {} must be smaller than Tmax {}".format(self.Tmin, self.Tmax)
        assert self.Bmax >= self.Bmin, "Bmin {} must be smaller than Bmax {}".format(self.Bmin, self.Bmax)
        
        self.val_bounds = []
        for k in range(self.Nspin):
          self.val_bounds.append((self.Bmin,self.Bmax))
        self.val_bounds.append((self.Tmin,self.Tmax))
        
        ## generate training and test set hamilotniann
        self.use_fixed_ham = use_fixed_ham
        self.train_size = opt_train_size
        self.randH, self.randH_test = self.randHset_constructor(train_size=opt_train_size)
        # print(self.randH)
        # raise AssertionError

        if log:
            self.logger = logging.basicConfig(filename=self.filename, encoding='utf-8', level=logging.DEBUG)
        self.filename = self.filename_generator()
        
        ### admin stuff
        self.save = save
        self.record = {"time_to_get_fid":None, "func_calls":None, "iterations":None, "repeats":None, "best_fid":None, "controller":None }
        ### multiple records can also be stored
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
        
    
    def filename_generator(self):
        return "lbfgs_record_s{}_o{}_t{}_b{}_r_{}.json".format(
            self.Nspin, self.Out, self.Tmax, self.Bmax,self.repeats)
    
    def randHset_constructor(self, train_size=1000, test_size=10000):
        # TODO cache this and make this a universal training set
        np.random.seed(4)
        out_train = np.zeros((train_size, self.Nspin, self.Nspin), dtype="complex128")
        for i in range(train_size):
            H = self.HH.copy()
            H += self.structured_perturabation()
            out_train[i] = H
            
        out_test = np.zeros((test_size, self.Nspin, self.Nspin), dtype="complex128")
        for i in range(test_size):
            H = self.HH.copy()
            H += self.structured_perturabation()
            out_test[i] = H
            
        return out_train, out_test
        

    def sys_hamiltonian(self):
        HH = np.zeros((self.Nspin,self.Nspin), dtype=np.complex128);
        for l in range (1,self.Nspin):
            HH[l-1,l] = 1
            HH[l,l-1] = 1
        if self.topo =="ring":
            HH[self.Nspin-1,0] = 1
            HH[0,self.Nspin-1] = 1
        if self.heisenberg_int:
            t = 0.5*np.triu(HH).sum().sum()*np.ones(self.Nspin) - np.sum(HH, axis=1)
            HH += np.diag(t)
        return HH
    
    def controls(self):        
        CC = []
        for k in range(0,self.Nspin):
            CM = np.zeros((self.Nspin,self.Nspin))
            CM[k,k] = 1
            CC.append(CM)
        return CC


    def eval_static_fidelity_gradient(self, x):
        """
        x[0:], x[-1] -> biases, Time ( size: len(biases)+1)
        
        Notes
        -----
        1. assumption in the grad calculation is that the overlap is real!
        otherwise need to consider the complex conjugate case of the overlap 
        instead of the 2*(respective vals) for grad w.r.t bias/time.
        
        2. bias grad is optimized using a trick to get grad exp faster. 
           time grad is straightforward
        """
        T = abs(x[self.Nspin])
    
        H = self.HH.copy()
        for l in range(self.Nspin):
            H += x[l] * self.CC[l]
            
        if self.ham_noisy:          # experimental: what happens if I try to jitter the grad cost function?
            H += self.structured_perturabation()
            
        # Calculate propagator
        TH = -1j*T*H
        U = sp.linalg.expm(TH)
    
        # Derivatives w.r.t. biases (helper)
        dU = []
        nd = TH.shape[0]
        A = np.zeros((2*nd,2*nd)) + 1j* np.zeros((2*nd,2*nd))
        A[0:nd,0:nd] = TH
        A[nd:2*nd,nd:2*nd] = TH
        for l in range(0,self.Nspin):
            A[nd:2*nd,0:nd] = -1j*T*self.CC[l]
            PSI = sp.linalg.expm(A)
            dU.append(PSI[nd:2*nd,0:nd])
        # Derviatives w.r.t time (helper)
        HU = np.dot(H,U)
        # Calculate infidelity and gradient for all i/o maps
        grad = np.zeros(self.Nspin+1)
        phi = U[self.Out,self.In]
        # Infidelity
        err = (1 - (phi.real * phi.real + phi.imag * phi.imag))
        # Gradient w.r.t. biases
        for l in range(0,self.Nspin):
            z = np.matrix(dU[l])[self.Out,self.In] * phi.conjugate()
            grad[l] -= 2 * z.real
        # Gradeself.Int w.r.t. time
        z = HU[self.Out,self.In] * phi.conjugate()
        grad[self.Nspin] -= 2 * z.imag
        return err, grad

    def overlap_ss(self, x):
        # Hamiltonian construction
        H = self.HH.copy()
        for l in range(0,self.Nspin):
            H += x[l] * self.CC[l]
        # Steady state
        e, V = np.linalg.eigh(H);
        rho_ss = np.diag(np.transpose(np.conj(V)) @ self.rho0 @ V)
        rho_out = np.transpose(np.conj(V)) @ self.rho1 @ V
        # Overlap
        return np.trace(np.diag(rho_ss) @ rho_out)
    
    def ngd_torch(self, funcalls):
        w = torch.rand(self.Nspin+1)
        w[0:self.Nspin] = self.Bmin + (self.Bmax-self.Bmin) * w[0:self.Nspin]
        w[self.Nspin] = self.Tmin + (self.Tmax-self.Tmin) *w[self.Nspin]
        w.requires_grad=True
        # optimizer = Adam([w])
        # optimizer = SGD([w], lr=1e-3)
        optimizer = RMSprop([w], lr=1e-2)
        max_fid = 0
        for _ in range(funcalls):
          
            T = w[self.Nspin]
            H = torch.Tensor(self.HH.copy())
            # if ham_noisy:
            #     H += self.structured_perturabation()
            for l in range(0,self.Nspin):
                H += w[l] * torch.Tensor(self.CC[l])
            H += torch.Tensor(self.structured_perturabation())
            U = torch.matrix_exp(-1j*T*H)
            phi = U[self.Out,self.In]
        
            fid = (phi.real * phi.real + phi.imag * phi.imag)
            optimizer.zero_grad()
            loss=-fid 
            max_fid = min(max_fid, float(loss))
            loss.backward()
            optimizer.step()
            print(loss, f"max_fid: {-1*max_fid}")
        
    
    def adam(self, funcalls):
        "TODO: fix. quite slow and doesn't converge"
        m = np.random.rand(self.Nspin+1)
        v = np.random.rand(self.Nspin+1) 
        beta_1 = 0.9; beta_2=0.999; eta=0.008
        w = np.random.rand(self.Nspin+1) 
        w[0:self.Nspin] = self.Bmin + (self.Bmax-self.Bmin) * np.array(w[0:self.Nspin])
        w[self.Nspin] = self.Tmin + (self.Tmax-self.Tmin) * np.array(w[self.Nspin])
        its=0
        min_inf = 1
        err = 1
        restarts = 0
        grad=None
        for i in range(funcalls):
            # w *= np.abs(1+np.random.normal(scale=self.noise, size=self.Nspin+1))

            # batch grad update
            # batch_size = 50
            # err, grad = 0, np.zeros(self.Nspin+1)
            # for _ in range(batch_size):
            #     erri, gradi = self.eval_static_fidelity_gradient(w)
            #     err += erri
            #     gradi += gradi
            # grad /= batch_size
            # err /= batch_size
            
            # restart conditions: grad is small and have tried gradient descending hard enough
            if type(grad) is not type(None):
                grad_norm = np.linalg.norm(grad, ord=2)
            else:
                grad_norm=-1 # default true
            if (its+1)%5000==0 and grad_norm < 1e-4:
                while True:
                    w_temp = np.random.rand(self.Nspin+1) 
                    w_temp[0:self.Nspin] = self.Bmin + (self.Bmax-self.Bmin) * np.array(w_temp[0:self.Nspin])
                    w_temp[self.Nspin] = self.Tmin + (self.Tmax-self.Tmin) * np.array(w_temp[self.Nspin])
                    # restart if your current endeavour is looking fruitless
                    
                    # if self.fidelity_ss(w_temp) < err:
                    _,grad=self.eval_static_fidelity_gradient(w_temp)
                    restarts += 1
                    if np.linalg.norm(grad, ord=2) >1e-4:
                        w=w_temp
                        # restarts += 1
                        break
            
            # 1. exact analytic gradient with fidelity err <-> infidelity

            err, grad = self.eval_static_fidelity_gradient(w)
            
            # 2. or using finite differencing (for debugging)
            # eps = (np.random.rand(self.Nspin+1)**2 - 1)*1e-7
            # err = 1-self.fidelity_ss(w)
            # grad = (1-self.fidelity_ss(w+eps) - (err)) / eps
            
            # adam update
            m = beta_1*m+(1-beta_1)*grad
            v = beta_2*v+(1-beta_2)*grad*grad
            m_hat = m/(1-beta_1)
            v_hat = v/(1-beta_2)
            w -= eta*(m_hat)/(np.sqrt(v_hat)+1e-8)
            its += 1
            min_inf = min(min_inf, err)
            print(grad)
            print("infidelity: ", err, "its: ", its, "fid: ", 1-min_inf, "restarts: ", restarts)
            
        return w
    
    
    @staticmethod
    def whole_sphere_sampling(size, dim):
        """
        Sample on the whole n-ball using the box-muller method. 
        probably overkill as I've confirmed otherwise that I'm already 
        sampling from the whole ball using the structured perturbation method.
        """
        
        nrvs = np.random.normal(0,1,size=(size, dim))
        l2norm = np.sum(nrvs*nrvs, axis=1)**(0.5)      
        r = np.random.random(size=size)/dim
        r /= l2norm
    
        return r[:,None]*nrvs
    
    def directional_perturbation(self):
        """
        perturb 2 points in a hermitian matrix by a deterministic value
        given by the `self.noise` parameter.
        
        e.g. [[_,_,_]   ->   [[_,_+a-0.435j,_]
              [_,_,_]        [_+a+0.435j,_,_]]
              [_,_,_]]       [_,_,_]]   

        Returns
        -------
        None

        """
        diag_dir = np.random.randint(low=0, high=self.Nspin) # diagonal direction
        dir_offset = np.random.randint(low=-1, high=2) # diag offset (currently only nearest neighbours)
        pert_index = (diag_dir, diag_dir+dir_offset)
        pert_index2 = (diag_dir+dir_offset, diag_dir)
        
        z=np.zeros((self.Nspin,self.Nspin), dtype=np.complex128)
        nval = np.random.normal(scale=self.noise, size=2)
        z[pert_index] = nval[0]+1j*nval[1]
        z[pert_index2] = nval[0]-1j*nval[1]
        
        return z
     
    def structured_perturabation(self):
        z=np.zeros((self.Nspin,self.Nspin), dtype=np.complex128)
        
        for i in range(self.Nspin):
            z[i][i] = np.random.normal(scale=self.noise)
            nn, nnn = np.random.normal(scale=self.noise), 0 # np.random.normal(scale=0.05) # nearest neighbour and next nearest neighbour
            #nn2, nnn2 = np.random.normal(scale=self.noise), 0 # np.random.normal(scale=0.05) # nearest neighbour and next nearest neighbour
            if i >= 1:
                z[i][i-1]=nn#+1j*nn2
                z[i-1][i]=nn#-1j*nn2
            if i >=2:
                z[i][i-2]=nnn#+1j*nnn2
                z[i-2][i]=nnn#-1j*nnn2
        return z


    # Target functional for optimisation over all i/o maps, no gradient
    def fidelity_ss(self, x, noisy=False, ham_noisy=False, use_fixed_ham=False, rH=None):
        T = abs(x[self.Nspin])
        if use_fixed_ham:
            H = rH.copy()
            if H is None:
                raise AssertionError(f"H cannot be {type(H)}")
        else:
            H = self.HH.copy()
            if ham_noisy:
                H += self.structured_perturabation()
        
        for l in range(0,self.Nspin):
            H += x[l] * self.CC[l]
            
        U = sp.linalg.expm(-1j*T*H)
        phi = U[self.Out,self.In]
        
        fid = (phi.real * phi.real + phi.imag * phi.imag)
        
        if not noisy:
            return fid 
    
        else: # noisy fidelity functional
            if not self.adaptive:
                noisy_fid = np.random.binomial(self.draws, fid) / self.draws
                #print(fid, noisy_fid)
                return noisy_fid
            else:
                a,b = 0.5, 0.5 # jeffrey's prior for the conjugate dist. beta
                mean = a / (a+b)  # beta mean
                var = mean*(1-mean) / (a+b+1) # beta var
                # print(mean, np.sqrt(var))
                while np.sqrt(var) > self.adp_tol:
                    s = np.random.binomial(self.draws, fid)
                    a+=s
                    b+=(self.draws-s)
                    mean = (a+s) / (a+b+self.draws)
                    var = mean*(1-mean) / (a+b+self.draws+1) 
                    self.adp_func_calls_increment += self.draws 
                # print(mean, np.sqrt(var), self.adp_func_calls_increment)
                return mean
            
    def fidelity_ss_av(self, x, noisy=False, ham_noisy=False, reps=10, test=False):
        fids = np.zeros(reps)
        # idx = np.random.randint(0, len(self.randH), size=reps)
        if not test:
            for rep in range(reps):
                # print(self.randH[idx[rep]])
                fids[rep] = self.fidelity_ss(x, noisy=noisy, ham_noisy=ham_noisy, 
                                             use_fixed_ham=True, rH=self.randH[rep])
            out = fids.mean()
            # print(max(fids))
            return out
        else:
            size=len(self.randH_test)
            fids = np.zeros(size)
            for rep in range(size):
                # print(self.randH[idx[rep]])
                fids[rep] = self.fidelity_ss(x, noisy=noisy, ham_noisy=ham_noisy, 
                                             use_fixed_ham=True, rH=self.randH_test[rep])
            out = fids.mean()
            return out
            
    
    def wass_cost(self, x, bootstrap_reps=5):
        fid_dist = np.zeros(bootstrap_reps)
        #min_fid=1
        for i in range(bootstrap_reps):
            f = self.fidelity_ss(x, ham_noisy=True) # isotropically perturb/probe the surrounding region within a normal ball of radius env.noise
            fid_dist[i] = f    
        wdcost = wd_from_ideal(fid_dist)

        return wdcost
    
    def find_min_fid_index(self,controller_list):
        "in non-increasing order of fid"
        c2fid = lambda c: self.fidelity_ss(c)
        fids = list(map(c2fid, controller_list))
        # print(fids)
        return np.argmin(fids)

    def run(self):
        funccalls = 0
        iters = 0
        start_time = tt.time()
        max_fid_seen = 0
        true = 0
        run_until_completion_criterion=False
        running_controllers = {}
        idsampling = True if self.landscape_exploration else False
        if idsampling:
            sampler = qmc.Sobol(d=self.Nspin+1, scramble=False)
        initx0 = None
        result=None
        for rep in range(0,self.repeats):
            ### Optimisation
            # Initial value
            
            #x0 = seq.next();
    
            if self.landscape_exploration:
                x0 = sampler.random()[0]
            else:
                x0 = np.random.rand(self.Nspin+1) 
            
            x0[0:self.Nspin] = self.Bmin + (self.Bmax-self.Bmin) * np.array(x0[0:self.Nspin])
            x0[self.Nspin] = self.Tmin + (self.Tmax-self.Tmin) * np.array(x0[self.Nspin])
            # L-BFGS Optimisation (with or without exact gradient)
            logging.info("Optimisation run ", rep+1)

            logging.info(x0)
            
            logging.info(len(self.val_bounds), self.val_bounds, )
            if not self.fid_noisy and not self.ham_noisy:
                x,f,d = optimize.fmin_l_bfgs_b(self.eval_static_fidelity_gradient,x0,bounds=self.val_bounds)
                mul_fac = 1
                
            else:
                if self.use_fixed_ham:
                    mul_fac = self.train_size
                    def infidelity(x):
                        return 1-self.fidelity_ss_av(x, noisy=self.fid_noisy, ham_noisy=self.ham_noisy, reps=mul_fac)
                else:
                    mul_fac = 1
                    def infidelity(x):
                        return 1-self.fidelity_ss(x, noisy=self.fid_noisy, ham_noisy=self.ham_noisy)
                def grad(x):
                    _, gr = self.eval_static_fidelity_gradient(x)
                    return gr
                if not self.use_wass_cost:
                    x,f,d = optimize.fmin_l_bfgs_b(infidelity,x0, bounds=self.val_bounds, approx_grad=True, 
                                                   maxfun=500)
                    #options = {'qn_hist_size': 10}
                    # xs = bfgs_e(infidelity, 
                    #             grad, x0, eps_f = self.noise*100, eps_g=self.noise*100 )
                    # x, f, iters, f_evals, g_evals, flag, results = xs
                    # d = {"nit": iters, "funcalls": f_evals, "gevals": g_evals,"flag":flag, 
                    #      "results":results}
                    

                    

            logging.info(" x    =", x)
            logging.info(" f(x) =", f)
            if self.use_fixed_ham:
                ol = None
                fi = 1-f
                true_fid = 1-f
                    
            else:
                ol = self.overlap_ss(x)
                fi = self.fidelity_ss(x, noisy=self.fid_noisy, ham_noisy=self.ham_noisy)
                true_fid = self.fidelity_ss(x)
            

            if self.verbose:
                if max_fid_seen < fi:
                    max_fid_seen = fi
                    if self.use_fixed_ham:
                        true = None # self.fidelity_ss_av(x, noisy=self.fid_noisy, ham_noisy=self.ham_noisy, test=True)
                    else:
                        true = self.fidelity_ss(x)
                        
                    # temp = self.noise 
                    # self.noise=0.1
                    # wd = self.wass_cost(x, bootstrap_reps=1000)
                    # self.noise=temp
                print(f"max_fid: {max_fid_seen}, true fid: {true}, fcalls: {funccalls}")
                # print(f"wd: {wd}")

            logging.info(" fidelity =", fi)
            logging.info(" overlap =", ol)
            logging.info(" iter =", d['nit'])
            logging.info(" warn =", d['warnflag'])
            
            funccalls += d["funcalls"]*mul_fac
            iters += d["nit"]
            
            if not self.adaptive:
                funccalls += d["funcalls"]
                if self.verbose:
                      print(funccalls)
            else:
                funccalls += self.adp_func_calls_increment
                self.adp_func_calls_increment  = self.draws
                if self.verbose:
                    print(funccalls)
            
            def save_controller_data_aux(): # auxilliary routine to save space
                self.record["time_to_get_fid"] = tt.time()-start_time
                self.record["func_calls"] = funccalls
                self.record["iterations"] = iters
                self.record["repeats"] = rep 
                self.record["controller"] = x.tolist()
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
                            running_controllers[fi]=x.tolist()
                            # print("running_list: \n", running_controller_list)
                        else:
                            #itopop=self.find_min_fid_index(running_controller_list) # time to pop this ###
                            itopop=min(list(running_controllers.keys()))
                            running_controllers.pop(itopop)
                            running_controllers[fi]=x.tolist() # maintain const size list
                            # print("running_list: \n", running_controller_list)
                            
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
            
    def save_record(self):
        json.dump( self.record, open(self.filename, 'w'))
        
    def read_record(self):
        return json.load(open(self.filename))
    
    
class Adam(LBFGS): 
    def __init__(self, *listargs, **dictargs):
        super().__init__(*listargs, **dictargs)
        self.idsampling = True
        
    def run(self):
        if not self.run_until_told_to_stop or not self.landscape_exploration:
            raise Exception("alternative functionality isn't available yet.")
        
        funccalls = 0
        start_time = tt.time()

        run_until_completion_criterion=False
        running_controllers = {}
        idsampling = True if self.landscape_exploration else False
        m = np.random.rand(self.Nspin+1)
        v = np.random.rand(self.Nspin+1) 
        beta_1 = 0.9; beta_2=0.999; eta=0.008 if self.Nspin > 7 else 0.03
        if idsampling:
            sampler = qmc.Sobol(d=self.Nspin+1, scramble=False)
            w = sampler.random()[0]
        else:
            w = np.random.rand(self.Nspin+1) 
        w[0:self.Nspin] = self.Bmin + (self.Bmax-self.Bmin) * np.array(w[0:self.Nspin])
        w[self.Nspin] = self.Tmin + (self.Tmax-self.Tmin) * np.array(w[self.Nspin])
        # TODO: understand behavior of algo with id sampling to see if its worth a shot. ==> a priori feeling is that this is a first order fruit
        its=0
        min_inf = 1
        err = 1
        restarts = 0
        grad=None
        
        tot_its = 0
        while tot_its < self.run_until_completion_its:
            
            # restart conditions: grad is small and have tried gradient descending hard enough
            if type(grad) is not type(None):
                grad_norm = np.linalg.norm(grad, ord=2)
            else:
                grad_norm=-1 # default true
            if (its+1)%5000==0: #and grad_norm < 1e-4:
                while True:
                    if idsampling:
                        w_temp = sampler.random()[0]
                    else:
                        w_temp = np.random.rand(self.Nspin+1) 
                    w_temp[0:self.Nspin] = self.Bmin + (self.Bmax-self.Bmin) * np.array(w_temp[0:self.Nspin])
                    w_temp[self.Nspin] = self.Tmin + (self.Tmax-self.Tmin) * np.array(w_temp[self.Nspin])
                    # restart if your current endeavour is looking fruitless
                    
                    # if self.fidelity_ss(w_temp) < err:
                    _,grad=self.eval_static_fidelity_gradient(w_temp)
                    restarts += 1
                    tot_its += 1
                    funccalls += 1
                    th = 1e-4 if self.Nspin > 7 else 1e-2
                    if np.linalg.norm(grad, ord=2) >th:
                        w=w_temp
                        # restarts += 1
                        break
        
            err, grad = self.eval_static_fidelity_gradient(w)
            
            # adam update
            m = beta_1*m+(1-beta_1)*grad
            v = beta_2*v+(1-beta_2)*grad*grad
            m_hat = m/(1-beta_1)
            v_hat = v/(1-beta_2)
            w -= eta*(m_hat)/(np.sqrt(v_hat)+1e-8)
            its += 1
            tot_its +=1
            funccalls += 1
            
            min_inf = min(min_inf, err)
            if self.verbose:
                print(grad)
                print("infidelity: ", err, "its: ", its, "fid: ", 1-min_inf, "restarts: ", restarts)


            fi = self.fidelity_ss(w, noisy=self.fid_noisy, ham_noisy=self.ham_noisy)
        
            true_fid = self.fidelity_ss(w)
            
            
            def save_controller_data_aux(): # auxilliary routine to save space
                self.record["time_to_get_fid"] = tt.time()-start_time
                self.record["func_calls"] = funccalls
                self.record["iterations"] = tot_its
                self.record["repeats"] = restarts
                self.record["controller"] = w.tolist()
                # print("wd final soln: ", self.Monte_env.wass_cost(next_store, 1000))
                if self.landscape_exploration:
                    self.record["controllers"] = list(running_controllers.values())
                if self.ham_noisy or self.fid_noisy:
                    self.record["best_fid"] = true_fid 
                else:
                    self.record["best_fid"] = fi
                    
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
                        running_controllers[fi]=w.tolist()
                        # print("running_list: \n", running_controller_list)
                    else:
                        #itopop=self.find_min_fid_index(running_controller_list) # time to pop this ###
                        if funccalls%5000 == 0:   
                            itopop=min(list(running_controllers.keys()))
                            running_controllers.pop(itopop)
                            running_controllers[fi]=w.tolist() # maintain const size list
                            # print("running_list: \n", running_controllers)
                save_controller_data_aux()

            if run_until_completion_criterion:
                return self.record["best_fid"]
          
            # run for a fixed number of iterations and then terminate
            run_until_completion_criterion = funccalls+1 >= self.run_until_completion_its
        
            
        return w
    
class SNOB(LBFGS):
    def __init__(self, *listargs, **dictargs):
        super().__init__(*listargs, **dictargs)
    
    def run(self):
        funccalls = 0
        iters = 0
        start_time = tt.time()
        max_fid_seen = 0
        true = 0
        run_until_completion_criterion=False
        running_controllers = {}
        idsampling = True if self.landscape_exploration else False
        if idsampling:
            sampler = qmc.Sobol(d=self.Nspin+1, scramble=False)
        initx0 = None
        result=None
        for rep in range(0,self.repeats):
            logging.info("Optimisation run ", rep+1)
            ### Optimisation
            # Initial value
            
            #x0 = seq.next();

            if self.landscape_exploration:
                x0 = sampler.random()[0]
            else:
                x0 = np.random.rand(self.Nspin+1) 
            
            # x0 = np.random.rand(self.Nspin+1) 
            x0[0:self.Nspin] = self.Bmin + (self.Bmax-self.Bmin) * np.array(x0[0:self.Nspin])
            x0[self.Nspin] = self.Tmin + (self.Tmax-self.Tmin) * np.array(x0[self.Nspin])
            # L-BFGS Optimisation (with or without exact gradient)

            logging.info(x0)
            
            logging.info(len(self.val_bounds), self.val_bounds, )

                
            def infidelity(x):
                if not self.use_fixed_ham:
                    return 1-self.fidelity_ss(x, noisy=self.fid_noisy, ham_noisy=self.ham_noisy)
                else:
                    return 1-self.fidelity_ss_av(x, noisy=self.fid_noisy, ham_noisy=self.ham_noisy, reps=self.train_size)
        
            def grad(x):
                _, gr = self.eval_static_fidelity_gradient(x)
                return gr
            if self.use_fixed_ham:
                budget=300
            else:
                budget=300

            options = optset(optin= {
                                "maxmp": 150,
                                "maxfail": 100,
                                "verbose": False,
                            })
            result, history = skq.minimize(
                        infidelity,
                        x0,
                        bounds=self.val_bounds,
                        budget=budget,
                        method="snobfit",
                        options=options,
                    )

            fi = 1-result.optval

            x = result.optpar


            if self.use_fixed_ham:    
                ol = None
                true_fid = fi = 1-result.optval
            else:
                ol = self.overlap_ss(x)
                true_fid = self.fidelity_ss(x)
            

            if self.verbose:
                if max_fid_seen < fi:
                    max_fid_seen = fi
                    if self.use_fixed_ham:
                        true = None # self.fidelity_ss_av(x, noisy=self.fid_noisy, ham_noisy=self.ham_noisy, test=True)
                    else:
                        true = self.fidelity_ss(x)

                print(f"max_fid: {max_fid_seen}, true fid: {true}")
                # print(f"wd: {wd}")

            
            if not self.adaptive:
                if self.use_fixed_ham:
                    funccalls += budget*self.train_size
                else:
                    funccalls += budget
                if self.verbose:
                      print("fcalls", funccalls)
            else:
                funccalls += self.adp_func_calls_increment
                self.adp_func_calls_increment  = self.draws
                if self.verbose:
                    print(funccalls)
            
            def save_controller_data_aux(): # auxilliary routine to save space
                self.record["time_to_get_fid"] = tt.time()-start_time
                self.record["func_calls"] = funccalls
                self.record["iterations"] = None
                self.record["repeats"] = rep 
                self.record["controller"] = x.tolist()
                # print("wd final soln: ", self.Monte_env.wass_cost(next_store, 1000))
                if self.landscape_exploration:
                    self.record["controllers"] = list(running_controllers.values())
                    if self.records_update_rate:
                        self.record_collector(funccalls, self.record["controllers"] )
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
                            running_controllers[fi]=x.tolist()
                            # print("running_list: \n", running_controller_list)
                        else:
                            #itopop=self.find_min_fid_index(running_controller_list) # time to pop this ###
                            itopop=min(list(running_controllers.keys()))
                            running_controllers.pop(itopop)
                            running_controllers[fi]=x.tolist() # maintain const size list
                            # print("running_list: \n", running_controllers)
                    save_controller_data_aux()

                if run_until_completion_criterion:
                    return self.record["best_fid"]
                if tt.time()-start_time > self.timeout: # relegated to a fail-safe (extremely unlikely e.g. don't want to wait all day for 1 run)
                    print(f"timed out! {self.filename}")
                    raise AssertionError("timeout")
                
                # run for a fixed number of iterations and then terminate
                run_until_completion_criterion = funccalls+1 >= self.run_until_completion_its
            
            
if __name__ == '__main__':

    trial = LBFGS(5, 0, 2, -10, 10, 70, fid_noisy=False, draws=100, repeats=10000, 
                  fid_threshold=0.0, ham_noisy=True, verbose=True, adaptive=False, 
                  adp_tol=0.05, noise=0.05, run_until_completion_its=20000000, run_until_told_to_stop=True,
                  landscape_exploration=True, save_topc=10, use_fixed_ham=False, opt_train_size=100, 
                  #records_update_rate=10,
                  )
    trial.run()
    # trial.adam(500000)
    # trial.ngd_torch(1000000)
    
    
