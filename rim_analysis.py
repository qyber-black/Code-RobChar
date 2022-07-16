#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 21:34:18 2022

@author: irtazakhalid
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 

def dom(a,b=1,points=100):
    return np.linspace(a,b,points)

def right_tail(dom, power=5):
    f = lambda x: 1/(x**power)
    return f(dom)/f(dom).sum()

def left_tail(dom, power=5):
    f = lambda x: 1/(x**power)
    return (f(dom)/f(dom).sum())[::-1]

def uniform(dom):
    return 1/len(dom)

def gaussian(dom):
    mean = np.mean(dom)
    f = lambda x: np.exp(-0.25*(x-mean)**2)
    return f(dom)/f(dom).sum()

def moments_vs_tails(a, pdfs=[right_tail, left_tail, gaussian, uniform]):
    fig, ax = plt.subplots(ncols=len(pdfs))
    ax = ax.ravel()
    a_grid = np.linspace(a,1,100)
    for j, pdf in enumerate(pdfs):
        sdict = {kk:np.zeros(100) for kk in ["mean", "std", "mom_2", "mom_3"]}
        for i,a in enumerate(a_grid):
            x = dom(a=a,b=1, points=50)
            # x[-1]=0.2
            # reduce a to make  pdf weights grow faster, transition to delta
            # weights are always constant, just shifting the domain..
            pdfw = pdf(dom(a=0.5, b=1, points=50))
            mean = (pdfw*x).sum()
            sdict["mean"][i] = mean
            sdict["std"][i] = np.sqrt((pdfw*(x-mean)**2).sum())
            sdict["mom_2"][i] = np.power((pdfw*(x)**2).sum(),1)#*(1/50)**(1/3-1/1)
            sdict["mom_3"][i] = np.power((pdfw*(x)**3).sum(),1)
            
        for key in sdict:
            ax[j].plot(a_grid, sdict[key], label=key)
            ax[j].set_xlabel("a dom left")
            ax[j].set_title(pdf.__name__)
            ax[j].vlines(0.5, 0, 1, linestyles="--")
        ax[0].set_ylabel("statistic")
        ax[0].legend(fontsize=7)
    plt.tight_layout()        
    
moments_vs_tails(0.001)


def p_order_rim(a=0.2, b=1,pdfs=[right_tail, left_tail, gaussian, uniform]):
    ps=range(1,50)
    x = dom(a=a, b=b, points=100)
    plt.figure()
    for pdf in pdfs:
        out = []
        for power in ps:
            pdfw = pdf(dom(a=0.5, b=1, points=100))
            # x[-1]=0.2
            out.append(np.power((pdfw*(1-x)**power).sum(), 1/power))
        
        plt.plot(ps, out, label=pdf.__name__)
        f = lambda x,a,b: a*np.log(x)+b
        ff,_=curve_fit(f, ps, out)
        plt.plot(ps, f(ps, *ff), linestyle="--",label=f"log fit slope {round(ff[0],3)}")
        
    plt.xlabel("p")
    plt.ylabel("p-order rim")
    plt.legend()
    plt.title(f"dom [{a}, {b}]")

def samples_vs_mean_val():
    plt.figure()
    for low in np.linspace(0.01, 0.99, 10):
        rims = []
        ns = np.arange(10,100,10)
        for n in ns:
            rims.append((1-np.random.uniform(low=low, high=1, size=n)).mean())
        plt.plot(ns, rims*ns**(0.5), label=f"min(rim)={np.round(low,2)}")
    plt.xlabel("samples")
    plt.ylabel("rim upper bound")
    plt.legend()
p_order_rim()
samples_vs_mean_val()
plt.show()
            
        
        