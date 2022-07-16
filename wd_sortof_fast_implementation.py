#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 20:35:16 2021

@author: irtazakhalid
"""
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from math import factorial

def check_fidtype(f):
    def method(fids, *args, **kwargs): 
        # basic sanity checks
        if not isinstance(fids, np.ndarray):
            if isinstance(fids, list):
                fids = np.array(fids)
            else:
                fids = np.array([fids])
                
        
        if (np.abs(fids-1e-8) > 1).any() or (np.abs(fids-1e-8) < 0).any():

            raise AssertionError("illegal fids values - must be in [0,1]")
        if type(fids) is type(None):
            raise Exception("please supply a vector or a scalar")
        else:
            return f(fids,*args, **kwargs)
    return method

def normalize(cdf: np.ndarray) -> np.ndarray:
    cdf /= cdf.sum()
    assert abs(cdf[-1]-1) < 1e-7, "couldn't normalize"
    return cdf


def compute_dkw_error(alpha, nobs ):
    return np.sqrt(np.log(2/alpha)/(2*nobs))

@check_fidtype
def dkw_ecdf_bounds(cdf, conf_level: float, visualize: bool = False)-> Tuple[np.ndarray, np.ndarray]:
    """
    Computes DKW inequality bounds for an estimated cumulative probability distribution, cdf.

    Parameters
    ----------
    cdf : Any list. 
        Cumulative probabilty distribution
    conf_level : float
        confidence level e.g. 0.9 is 90% confidence level.
    visualize : bool, optional
        Plot the confidence interval. The default is False.

    Returns
    -------
    lower : np.ndarray
        lower bound
    upper : np.ndarray
        upper bound

    """
    
    alpha = 1-conf_level
    # dkw confidence interval width
    epsilon = compute_dkw_error(alpha, cdf.shape[-1])
    lower = np.clip(cdf-epsilon,0,1)
    upper = np.clip(cdf+epsilon,0,1)
    
    if visualize:
        plt.figure()
        plt.step(cdf, np.arange(len(cdf))/len(cdf), label="ecdf", c="b")
        plt.step(lower, np.arange(len(cdf))/len(cdf), label="lower", c="r")
        plt.step(upper, np.arange(len(cdf))/len(cdf), label="upper", c="r")
        plt.ylabel(r"$Q_F$")
        plt.xlabel(r"$F$")
        plt.legend()
    
    return (lower, upper)
    

@check_fidtype
def wd_from_ideal(fids, sort_fids: bool = True):
    """
    Computes Wasserstein robustness measure of a fidelities represented by a 1D 
    array of samples, fids, by computing the 1-Wasserstein distance between the 
    fidelity distribution and the ideal distribution delta(x-1).

    Parameters
    ----------
    fids : List of fidelity values. Must be in [0-1]
    sort_fids : bool, optional
        Sort the fidelity values in ascending order. The default is True.

    Returns
    -------
    wd_1 : 1-wasserstein distance from \delta(x-1)
    
    Notes
    ------
    no bins: check convention at `intervals`. Assume that the fids array is unsorted
    """
    
    if sort_fids:
        fids.sort(kind="quicksort")
    
    # construct intervals f[ecdf] - f[ecdf-1], starting from 0 ending before 1, similar to the scipy implementation 
    intervals = np.diff(np.concatenate((fids, [1])))
    
    # no need to create an explicit cdf as we are comparing with a const comparison cdf, edit: this is a simple step cdf
    cdf = np.arange(1, fids.size+1) / fids.size

    # weighted sum in the style W(., \delta(1))
    wd_1 = np.multiply(intervals, cdf).sum()
    
    return wd_1


def wd_from_ideal_zero(fids, sort_fids: bool = True):
    """
    Computes Wasserstein robustness measure of fidelities represented by a 1D 
    array of samples, fids, by computing the 1-Wasserstein distance between the 
    fidelity distribution and the ideal distribution delta(x-0).

    Parameters
    ----------
    fids : List of fidelity values. Must be in [0-1]
    sort_fids : bool, optional
        Sort the fidelity values in ascending order. The default is True.

    Returns
    -------
    wd_1 : 1-wasserstein distance from \delta(x-0)
    
    Notes
    ------
    no bins: check convention at `intervals`. Assume that the fids array is unsorted
    """

    wd_1 = 1-wd_from_ideal(fids, sort_fids)
    
    return wd_1

def binomial(n,k):
    return factorial(n)/(factorial(k)*factorial(n-k))

@check_fidtype
def RIM_p(fids: np.ndarray, p=2)->float:
    """
    Computes the p-Wasserstein robustness infidelity measure (p-RIM) of fidelities. 
    Generalizes previous measures.

    Parameters
    ----------
    fids : List of fidelity values. Must be in [0-1]
    p : int, optional
        Order of the Wasserstein distance. The default is 2.

    Returns
    -------
    wd_p : p-wasserstein distance from \delta(x-1)
    
    Notes
    ------
    The default 0-RIM is 1. Note that the infinity norm version is not yet implemented 
    for p=\infty.
    """
    if p==0:
        return 1
    out = np.power(1-fids, p).mean()
    # older version:
    # for i in range(0,p+1):
    #     out += binomial(p,i)*np.power(fids, i).mean()*pow(-1, i)
    return pow(out, 1/p)



import unittest 
from scipy.stats import wasserstein_distance


class testwdimplementation(unittest.TestCase):

    X = np.array([0.11080853, 0.19674286, 0.2515852 , 0.33965725, 0.39020078,
               0.56853594, 0.57607307, 0.67321294, 0.8323267 , 0.9901584 ])

    def test_custom_wd_from_ideal(self):
        
        Y = np.ones_like(self.X)
        
        mine = wd_from_ideal(self.X)
        print("mine: ", mine)
        wdscipy = wasserstein_distance(self.X,Y)
        
        print("official implementation: ", wdscipy)
            
        self.assertAlmostEqual(wdscipy, mine)
        self.assertAlmostEqual(mine, RIM_p(self.X, p=1))

    def test_RIM_p_2(self):
        mine = wd_from_ideal(self.X)
        self.assertAlmostEqual(np.sqrt(mine*mine+self.X.var()), RIM_p(self.X, p=2))
        X = np.random.normal(0.85, 0.8, size=10000).clip(min=0, max=1) 
        mine=wd_from_ideal(X)
        self.assertAlmostEqual(np.sqrt(mine*mine+X.var()), RIM_p(X, p=2))
    
    def test_normal_wd_from_ideal_1(self):
        X = np.random.normal(0.85, 0.02, size=10000) 
        Y = np.ones_like(X)
        mine=wd_from_ideal(X)
        self.assertAlmostEqual(wasserstein_distance(X,Y), mine)
        self.assertAlmostEqual(mine, RIM_p(X, p=1))
        self.assertAlmostEqual(np.sqrt(mine*mine+X.var()), RIM_p(X, p=2))
        
    def test_normal_wd_from_ideal_2(self):
        X = np.random.normal(0.67, 0.02, size=10)
        Y = np.ones_like(X)
        self.assertAlmostEqual(wasserstein_distance(X,Y), wd_from_ideal(X))
        self.assertAlmostEqual(wd_from_ideal(X), RIM_p(X, p=1))
        
    def test_wd_from_ideal_uniform(self):
        X = np.random.uniform(size=10)
        Y = np.ones_like(X)
        self.assertAlmostEqual(wasserstein_distance(X,Y), wd_from_ideal(X))
        self.assertAlmostEqual(wd_from_ideal(X), RIM_p(X, p=1))

    def test_wd_from_ideal_all_perfect_fids(self):
        X = np.array([1,1,1,1,1])
        Y = np.ones_like(X)
        
        self.assertAlmostEqual(wasserstein_distance(X,Y), wd_from_ideal(X))
        self.assertAlmostEqual(wd_from_ideal(X), RIM_p(X, p=1))
    
    def test_wd_from_ideal_some_perfect_fids(self):
        X = np.array([1,0,1,1,0])
        Y = np.ones_like(X)
        
        self.assertAlmostEqual(wasserstein_distance(X,Y), wd_from_ideal(X))
        self.assertAlmostEqual(wd_from_ideal(X), RIM_p(X, p=1))
        
    def test_wd_from_ideal_all_not_perfect_fids(self):
        X = np.array([0,0,0,0,0])
        Y = np.ones_like(X) 
        self.assertAlmostEqual(wasserstein_distance(X,Y), wd_from_ideal(X))
        self.assertAlmostEqual(wd_from_ideal(X), RIM_p(X, p=1))
        
    def test_scalar(self):
        X = 0.76
        Y = np.ones_like([X]) 
        self.assertAlmostEqual(wasserstein_distance([X],Y), wd_from_ideal(X))
        self.assertAlmostEqual(wd_from_ideal(X), RIM_p(X, p=1))
        
    def test_custom_wd_from_ideal_zero(self):
        X = np.array([0.11080853, 0.19674286, 0.2515852 , 0.33965725, 0.39020078,
               0.56853594, 0.57607307, 0.67321294, 0.8323267 , 0.9901584 ])
        
        Y = np.zeros_like(X)
        
        mine = wd_from_ideal_zero(X)
        
        print("mine: ", mine)

        wdscipy = wasserstein_distance(X,Y)
        
        print("official implementation: ", wdscipy)
            
        self.assertAlmostEqual(wdscipy, mine)
        self.assertAlmostEqual(mine, 1-RIM_p(X, p=1))
    
    def test_normal_wd_from_ideal_1_zero(self):
        X = np.random.normal(0.85, 0.02, size=10000) 
        Y = np.zeros_like(X)
        self.assertAlmostEqual(wasserstein_distance(X,Y), wd_from_ideal_zero(X))
        self.assertAlmostEqual(wd_from_ideal_zero(X), 1-RIM_p(X, p=1))  

    def test_normal_wd_from_ideal_2_zero(self):
        X = np.random.normal(0.67, 0.02, size=10)
        Y = np.zeros_like(X)
        self.assertAlmostEqual(wasserstein_distance(X,Y), wd_from_ideal_zero(X))
        self.assertAlmostEqual(wd_from_ideal_zero(X), 1-RIM_p(X, p=1))   
        
    def test_wd_from_ideal_uniform_zero(self):
        X = np.random.uniform(size=10)
        Y = np.zeros_like(X)
        self.assertAlmostEqual(wasserstein_distance(X,Y), wd_from_ideal_zero(X))
        self.assertAlmostEqual(wd_from_ideal_zero(X), 1-RIM_p(X, p=1))   
        
    def test_wd_from_ideal_all_perfect_fids_zero(self):
        X = np.array([1,1,1,1,1])
        Y = np.zeros_like(X)
        
        self.assertAlmostEqual(wasserstein_distance(X,Y), wd_from_ideal_zero(X))
        self.assertAlmostEqual(wd_from_ideal_zero(X), 1-RIM_p(X, p=1))   
    
    def test_wd_from_ideal_some_perfect_fids_zero(self):
        X = np.array([1,0,1,1,0])
        Y = np.zeros_like(X)
        self.assertAlmostEqual(wasserstein_distance(X,Y), wd_from_ideal_zero(X))
        self.assertAlmostEqual(wd_from_ideal_zero(X), 1-RIM_p(X, p=1)) 
        
    def test_wd_from_ideal_all_not_perfect_fids_zero(self):
        X = np.array([0,0,0,0,0])
        Y = np.zeros_like(X) 
        self.assertAlmostEqual(wasserstein_distance(X,Y), wd_from_ideal_zero(X))
        self.assertAlmostEqual(wd_from_ideal_zero(X), 1-RIM_p(X, p=1)) 
        
    def test_scalar_zero(self):
        X = 0.76
        Y = np.zeros_like([X]) 
        self.assertAlmostEqual(wasserstein_distance([X],Y), wd_from_ideal_zero(X))
        self.assertAlmostEqual(wd_from_ideal_zero(X), 1-RIM_p(X, p=1)) 
        
        


if __name__ == '__main__':

    unittest.main()
    

