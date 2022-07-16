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
"""

import numpy as np
import scipy as sp


class noise_function:
    def __init__(self, generator, **args):
        self.generator = generator
        self.args = args
        
    def __call__(self, **extraargs):
        """
        Parameters
        ----------
        scale : float, optional
            some measure of strength of the perturbation
        size : int, optional
            number of random numbers to be generatred
        **args : TYPE: sundry
            extra args specific to the `generator` constructor 

        Returns
        -------
        random numbers: array, list or scalar

        """
        # update extraargs dict
        for arg in extraargs:
            self.args[arg] = extraargs[arg]
                
        return self.generator(**self.args)
        

class noise_model_base:
    """
    Parameters
    ----------
    Nspin : int, optional
        Spin chain length. The default is 5.
    inspin : int, optional
        input state. The default is 0.
    outspin : int, optional
        output state. The default is 2.
    noise : float, optional
        noise strength. The default is 0.02.
    topo : str, optional
        topology: can be either "chain" or "ring". The default is "chain".
    rng : noise_function, optional
        random number generator

    Returns
    -------
    None.

    """
    def __init__(self, Nspin: int = 5, inspin: int = 0, outspin: int = 2, noise: float = 0.02,
                 topo: str = "chain", rng: noise_function = None):

        self.Nspin= Nspin
        self.inspin = inspin
        self.outspin = outspin
        self.noise = noise
        self.rng = self.default_gaussian_noise_generator(scale=self.noise) if rng is None else rng
        self.HH = np.zeros((Nspin,Nspin), dtype=np.complex128) 
        for l in range (1,self.Nspin):
            self.HH[l-1,l] = 1
            self.HH[l,l-1] = 1
        if topo =="ring":
            self.HH[self.Nspin-1,0] = 1
            self.HH[0,self.Nspin-1] = 1
            
        self.CC= self.controls()
        
    def controls(self):        
        CC = []
        for k in range(0,self.Nspin):
            CM = np.zeros((self.Nspin,self.Nspin))
            CM[k,k] = 1
            CC.append(CM)
        return CC
  

    def evaluate_noisy_fidelity(self, x, ham_noisy: bool = False):
        T = abs(x[self.Nspin])
        H = self.HH.copy()
        if ham_noisy:
            H += self.perturbation()
        for l in range(self.Nspin):
            H += x[l] * self.CC[l]
        U = sp.linalg.expm(-1j*T*H)
        phi = U[self.outspin,self.inspin]
        
        fid = (phi.real * phi.real + phi.imag * phi.imag)
        return fid 
    
    def perturbation(self) -> np.ndarray:
        raise NotImplementedError
        
    def default_gaussian_noise_generator(self, **genargs):
        return noise_function(np.random.normal, **genargs)
    
class structured_perturbation(noise_model_base):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def perturbation(self) -> np.ndarray:
        """
        Parameters
        ----------
        rng : noise_function, optional
            Random noise generator

        Returns
        -------
        z : np.ndarray
            Structured perturbation of the same matrix form as `HH`

        """
        z=np.zeros((self.Nspin,self.Nspin), dtype=np.complex128)

        for i in range(self.Nspin):
            z[i][i] = self.rng()
            nn, nnn = self.rng(), 0 # np.random.normal(scale=0.05) # nearest neighbour and next nearest neighbour
            nn2, nnn2 = self.rng(), 0 # np.random.normal(scale=0.05) # nearest neighbour and next nearest neighbour
            if i >= 1:
                z[i][i-1]=nn+1j*nn2
                z[i-1][i]=nn-1j*nn2
            if i >=2:
                z[i][i-2]=nnn+1j*nnn2
                z[i-2][i]=nnn-1j*nnn2
        return z
    
 
class directional_perturbation(noise_model_base):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.directions = [(0,0), (self.Nspin-1, self.Nspin-1)]
        for d in range(1,self.Nspin-1):
            for o in [-1,0,1]:
                self.directions.append((d, d+o))
        
        self.directions.append((0,1))
        self.directions.append((1,0))
        self.directions.append((self.Nspin-2, self.Nspin-1))
        self.directions.append((self.Nspin-1, self.Nspin-2))
        
    def perturbation(self) -> np.ndarray:
    
        """
        perturb 2 random points in a hermitian matrix by a deterministic value
        given by the `self.noise` parameter.
        
        e.g. [[_,_,_]   ->   [[_,_+a-0.435j,_]
              [_,_,_]        [_+a+0.435j,_,_]]
              [_,_,_]]       [_,_,_]]   

        Returns
        -------
        z : np.ndarray
            Structured perturbation on a single entry on `HH`

        """
        # biased older method
        
        # diag_dir = np.random.randint(low=0, high=self.Nspin) # diagonal direction
        
        # if diag_dir == 0 or diag_dir == self.Nspin-1: # at the boundary
        #     dir_offset = np.random.randint(low=-1, high=1) 
        # else:
        #     dir_offset = np.random.randint(low=-1, high=2) # diag offset (currently only nearest neighbours)

        # pert_index = (diag_dir, diag_dir+dir_offset) # boundary condition
        # pert_index2 = (diag_dir+dir_offset, diag_dir)
        
        pert_index = self.directions[np.random.randint(low=0, high=len(self.directions))]
        pert_index2 = (pert_index[1], pert_index[0])
        
        z=np.zeros((self.Nspin,self.Nspin), dtype=np.complex128)
        nval = self.rng(size=2)
        z[pert_index] = nval[0]+1j*nval[1]
        z[pert_index2] = nval[0]-1j*nval[1]
        
        return z
        
if __name__ == '__main__':
    x = structured_perturbation(rng=noise_function(np.random.uniform, low=0, high=2))
    y = directional_perturbation(Nspin=2, outspin=1)
    print(x.perturbation(), x.evaluate_noisy_fidelity(np.random.uniform(size=6), True))
    print(y.perturbation(), y.evaluate_noisy_fidelity(np.random.uniform(size=6)))
    x.rng(high=3000)  # call once to change noise strength: TODO: maybe do this in a more palatable fashion?
    print(x.perturbation())
