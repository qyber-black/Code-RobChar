#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 20:05:32 2020

@author: irtazakhalid
"""

import numpy as np
import scipy as sp
import scipy.linalg


class Environment(object):
    "simple XX spin chain environment with either a ring or linear topology"
    def __init__(self, nspin, in_spin, out_spin, action_vector=None, final_time = 6, 
                 topo="linear", timestep_res=0.01, max_time=30, bmin=-20, bmax=20,
                 fid_noisy=False, ham_noisy=False, draws=20, adaptive=False, adp_tol=0.05, 
                 noise=0.05, transfer_learning=False, heisenberg_int: bool = False, 
                 use_fixed_ham = False, opt_train_size = 100):
        
        self.Nspin = nspin
        self.in_spin = in_spin
        self.out_spin = out_spin
        self.topo = topo
        self.heisenberg_int = heisenberg_int
        self.timestep = 0
        self.tres = timestep_res
        self.action = np.zeros(self.Nspin) if type(action_vector)==type(None) else np.diag(action_vector) # create a diagonal matrix of actions on individual spins
        if transfer_learning:
            self.sys = (self.system_hamiltonian() + self.structured_perturabation(0.1))
            mask = np.ones_like(self.sys)- np.eye(self.Nspin)
            # print(mask)
            self.sys = self.sys*mask
            print(f"old ham {self.sys}")
        else:
            self.sys = self.system_hamiltonian()
        self.in_state = self.state_vector(self.in_spin)
        self.out_state = self.state_vector(self.out_spin)
        self.maxtime = max_time
        self.final_time = self.maxtime # min(abs(final_time), self.maxtime)
        self.min = bmin
        self.max = bmax
        self.noise = noise
        # print(self.in_state)
        # print(self.out_state)
        self.fid_noisy = fid_noisy
        self.ham_noisy = ham_noisy
        self.draws=draws
        self.adaptive = adaptive
        self.adp_func_calls_increment = self.draws
        self.adp_var_tol = adp_tol
        self.tf = 0
        self.use_fixed_ham = use_fixed_ham
        self.train_size = opt_train_size
        self.randH, self.randH_test = self.randHset_constructor(train_size=self.train_size)

    def randHset_constructor(self, train_size=1000, test_size=10000):
        # TODO cache this and make this a universal training set
        np.random.seed(4)
        out_train = np.zeros((train_size, self.Nspin, self.Nspin), dtype="complex128")
        for i in range(train_size):
            H = self.sys.copy()
            H += self.structured_perturabation(self.noise)
            out_train[i] = H
            
        out_test = np.zeros((test_size, self.Nspin, self.Nspin), dtype="complex128")
        for i in range(test_size):
            H = self.sys.copy()
            H += self.structured_perturabation(self.noise)
            out_test[i] = H
            
        return out_train, out_test

    def reinit_sys_hamiltonian(self):
        self.sys = (self.system_hamiltonian() + self.structured_perturabation(.1))
        mask = np.ones_like(self.sys)- np.eye(self.Nspin)
        # print(mask)
        self.sys = self.sys*mask
        print(f"new ham: {self.sys}")
    
    def system_hamiltonian(self):
        J = np.zeros((self.Nspin,self.Nspin));
        for l in range (1,self.Nspin):
            J[l-1,l] = 1
            J[l,l-1] = 1
        if self.topo == "ring":    
            J[self.Nspin-1,0] = 1
            J[0,self.Nspin-1] = 1
        if self.heisenberg_int:
            t = 0.5*np.triu(J).sum().sum()*np.ones(self.Nspin) - np.sum(J, axis=1)
            J += np.diag(t)
        return J
    
    def control_hamiltonians(self):
        "biased bases"
        CC = []
        for k in range(0,self.Nspin):
            CM = np.zeros((self.Nspin,self.Nspin))
            CM[k,k] = 1*self.biases[k]
            CC.append(CM)

        return CC
    
    def state_vector(self, occ):
        psi = np.zeros(self.Nspin)
        psi[occ] = 1
        return psi
    
    # as density matrices, probably not needed atm
    def input_state(self):
        rho0 = np.zeros((self.Nspin,self.Nspin))
        rho0[self.in_spin,self.in_spin] = 1
        return rho0
        
    def output_state(self):
        rho1 = np.zeros((self.Nspin,self.Nspin))
        rho1[self.out_spin,self.out_spin] = 1
        return rho1
    
    
    def structured_perturabation(self, noise):
        z=np.zeros((self.Nspin,self.Nspin))
        for i in range(self.Nspin):
            z[i][i] = np.random.normal(scale=noise)
            nn, nnn = np.random.normal(scale=noise), 0 # np.random.normal(scale=0.05) # nearest neighbour and next nearest neighbour
            if i >= 1:
                z[i][i-1]=nn
                z[i-1][i]=nn
            if i >=2:
                z[i][i-2]=nnn
                z[i-2][i]=nnn
        return z
    
    
    def change_sys_ham(self, default_variation = 0.1):
        """check if there is such a thing as haar random uniform for spin chain hams 
        otherwise just add small gaussian perturbations to the og self.sys"""
        for i in range(self.Nspin):
            nn = np.random.normal(scale=default_variation)  # nearest neighbour 
            if i >= 1:
                self.sys[i][i-1]+=nn
                self.sys[i-1][i]+=nn

        
    
    def state(self, action=None):
        action = self.action if type(action) == type(None) else action
        
        self.timestep = abs(self.timestep)
        self.timestep = self.timestep%self.maxtime if self.timestep > self.maxtime else self.timestep
        # using training dataset
        if self.use_fixed_ham:
            ham_list = self.randH[:self.train_size]
            out = None
            for H in ham_list:
                U = sp.linalg.expm(-1j*(self.timestep)*(H+action))
                if out is None:
                    out = U
                else:
                    out += U
            self.in_state = np.matmul(out/len(ham_list), self.in_state)
            return
        elif not self.ham_noisy:
            H = self.sys + action
        else:
            H = self.sys + action + self.structured_perturabation(self.noise) # add 5% gaussian noise to couplings+nn+nnn
            # print(H)
            
        
        # print(self.timestep, np.diag(action))
        U = sp.linalg.expm(-1j*self.timestep*H)
    
        #print(np.allclose(np.matmul(np.conj(U).T, U), np.eye(self.Nspin))) # sanity check. am i producing unitaries as I step?
        
        self.in_state = np.matmul(U, self.in_state)
        # self.timestep += self.tres
        return U
    


    def reset(self):
        self.timestep = 0
        self.in_state = self.state_vector(self.in_spin)
        self.action = np.zeros((self.Nspin, self.Nspin))
        #print(self.action)
        U = self.state()
        
        return self.action


    def fidelity(self):
        overlap = np.matmul(np.conj(self.in_state).T, self.out_state)
        fid = np.conj(overlap)*overlap
        # print(fid)
        assert np.allclose(np.imag(fid), 0)==True, "fid not real!" # sanity: correct fid is always real
        
        # if self.timestep < self.final_time:  # try out the Bukov et. al. (2018) approach
        #     if fid > 0.5:
        #         return np.real(fid)
        #     return 0
        
        # if fid < 0.9:
        #     return 0
        
        if not self.fid_noisy:
            return np.real(fid)
        else:
            sample = np.random.binomial(self.draws, fid)
            if not self.adaptive:
                return sample / self.draws
            
            else:
                a,b = 0.5, 0.5 # jeffrey's prior for the conjugate dist. beta
                mean = a / (a+b)  # beta mean
                var = mean*(1-mean) / (a+b+1) # beta var
                # print(mean, np.sqrt(var))
                while np.sqrt(var) > self.adp_var_tol:
                    s = np.random.binomial(self.draws, fid)
                    a+=s
                    b+=(self.draws-s)
                    mean = (a+s) / (a+b+self.draws)
                    var = mean*(1-mean) / (a+b+self.draws+1) 
                    self.adp_func_calls_increment += self.draws 
                # print(mean, np.sqrt(var), self.adp_func_calls_increment)
                return mean
              
                
    def _true_fid_single(self, action, base_H=None, timestep_n=None):
        if base_H is None:
            base_H = self.sys.copy()
            timestep_n=self.timestep
        H = self.sys + action
        U = sp.linalg.expm(-1j*timestep_n*H)  # non-noisy ham
        true_in_state = np.matmul(U, self.in_state)
        overlap = np.matmul(np.conj(true_in_state).T, self.out_state)
        fid = np.conj(overlap)*overlap
        return np.real(fid)
        
    def true_fid(self, action, timestep_n=None):        
        if self.use_fixed_ham:
            size=len(self.randH_test)
            fids = np.zeros(size)
            for rep in range(size):
                fids[rep] = self._true_fid_single(action, base_H=self.randH_test[rep], timestep_n=timestep_n)
            out = fids.mean()
            return out
        else:
            return self._true_fid_single(action)
            
        
    
    def normalize(self):
        "remove redundancy in the parameter space defined by the constraints in __init__"
        self.action = self.action%np.diag(np.sign(self.action)*self.max) if (np.abs(self.action) > self.max).any() else self.action
        self.timestep = abs(self.timestep)
        self.timestep = self.timestep%self.maxtime if self.timestep > self.maxtime else self.timestep
        
        
    def step(self, action):
        self.action += action
        self.action = self.action%np.diag(np.sign(self.action)*self.max) if (np.abs(self.action) > self.max).any() else self.action # check if i need to do mod negative, but atm b is symmetric about 0 
        try:
            if not self.use_fixed_ham: 
                self.tf = self.true_fid(self.action)
            self.state(self.action)  # evolve the state
            reward = self.fidelity()
            done_flag = True if self.timestep > self.final_time else False
            
            self.in_state = self.state_vector(self.in_spin) # reset the instate to get controllers that work at the end
            #next_state = unitary
            #print(self.action)
            return self.action, reward, done_flag
        except ValueError as e:
            print(e)
            return np.zeros_like(self.action), 0, False
    
def timeout(time_out):
    "time out after time_out seconds using this decorator"
    def timeout2(func):
        import time as tt
        start=tt.time()
        def method_executioner(*args, **kwargs):
            if tt.time()-start > time_out:
                raise AssertionError("timeout!")
            return func(*args, **kwargs)
        return method_executioner
    return timeout2
        
import unittest

class Envtest(unittest.TestCase):
    "could add more tests from Newton based algorithms for more sanity"
    
    def test_one_step_fid_correctness(self):
        "sanity check 1"
        
        action = np.array([  9.76909983,  10.65815206,  10.65467358  , 9.71995292, -12.,
                                    8.69457352 , 12. , -11.77314325, -11.29782006  , 5.27449319,])
        
        final_time = 25.13468797
        
        env = Environment(10, 0, 3, np.zeros(10), final_time=final_time, timestep_res=final_time)
        env.reset()
        env.timestep = final_time
        _, fid, _ = env.step(np.diag(action))
        
        self.assertAlmostEqual(fid, 0.995, places = 2)
        
        # another one
        action = np.array([-0.20574245,  4.3713235,  -0.30473375])
        final_time = 22.035034
        env = Environment(3, 0, 2, np.zeros(3),)
        env.reset()
        env.timestep = final_time
        _, fid, _ = env.step(np.diag(action))
        # print(env.action, env.timestep)
        
        self.assertAlmostEqual(fid, 0.90, places = 2)
        
        # another one
        action = np.array([2.9160861365962774, 4.385934774763882, 2.9311789427883923, 
                           9.826275581493974, 9.276727781863883, 5.071161912055686,])
        final_time = 3.6651542489416897
        env = Environment(6, 0, 2, np.zeros(6),)
        env.reset()
        env.timestep = final_time
        _, fid, _ = env.step(np.diag(action))
        # print(env.action, env.timestep)
        
        self.assertAlmostEqual(fid, 0.9025, places = 2)
        
        # bad one
        action= np.array([ 3.86111206, -0.8067965 ,  3.86887524,  5.8814842 , -3.03354326,
        7.42084848])
        final_time = 24.83387072
        env = Environment(6, 0, 2, np.zeros(6),)
        env.reset()
        env.timestep = final_time
        _, fid, _ = env.step(np.diag(action))
        self.assertTrue(fid < 0.9025)
        
        
    def test_hermitianity_of_structured_perturbation(self):
        
        env = Environment(20,0,6, np.zeros(20))
        z = env.structured_perturabation(env.noise) + env.sys
        self.assertTrue(np.allclose(z, np.conjugate(z.T)), "Perturbed ham must be real!")
        another_z = env.structured_perturabation(env.noise) + env.sys
        self.assertFalse(np.all(z==another_z), "don't fix the perturbation!")
        self.assertTrue(np.allclose(another_z, np.conjugate(another_z.T)), "Perturbed ham must be real!")
        
        
    def test_plot_fid_vs_true_fid_for_adaptive_protocol(self):
        
        env = Environment(5,0,3, np.zeros(5), fid_noisy=True, 
                          adaptive=True, draws=5, adp_tol=0.05)
        fid = 0.8
        ov=np.sqrt(fid)
        env.in_state = np.array([ov,0,0,0,0])
        env.out_state = np.array([1,0,0,0,0])
        
        # self.assertTrue(abs(env.fidelity()-fid)<0.05)
        env.fidelity()
        self.assertTrue(env.adp_func_calls_increment > 5)
        
        # more intense visualization!
        import matplotlib.pyplot as plt
        plt.figure()
        fid_space = np.linspace(0,1,100)
        for tol in [0.01, 0.02, 0.03, 0.06,0.1,0.2]:
            calls = []
            for fid in fid_space:
                env.adp_func_calls_increment = 5
                ov=np.sqrt(fid)
                env.in_state = np.array([ov,0,0,0,0])
                env.out_state = np.array([1,0,0,0,0])
                env.adp_var_tol = tol
                env.fidelity()
                calls.append(env.adp_func_calls_increment)
                
            plt.plot(calls, fid_space, label=f"tolerated var {tol}")
            
        
        plt.xlabel("draws or function calls")
        plt.ylabel("fidelity")
        plt.legend()
        
        # seems to be symmetric about 0.5 in the amount of apetite for func_calls; makes sense, 
        

    def test_timeout(self):
        @timeout(0)    # just throw an exception
        def timeout_function():
            pass       
        self.assertRaises(AssertionError, timeout_function) # catch it. complain if not caught
        
        
if __name__ == '__main__':
    unittest.main()

# raise AssertionError("custom break point")

