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

from ppo import PPO_en 
from qnewton import LBFGS, Adam, SNOB
from nmplus import NMPlus
import json
import os
import numpy as np
from dataclasses import dataclass
from parse import get_noise_analysis_args
from typing import Dict


# three levels of difficulty for the spin chain transition: 
#              * to -1, 
#              * to mid 
#              * to a fixed value (in between above)

@dataclass
class ExperimentNamer:
   experiment_name: str = "alpha"
   Nspin: int = 5
   inspin: int = 0
   outspin: int = 2
   numcontrollers: int = 100
   global_dir: str = "experiments" # change if not amenable
        
   def home(self):
       self.home = self.global_dir+"/"+self.experiment_name 
       if not os.path.exists(self.home):
           os.mkdir(self.home)  # all experiments of different individual hyperparameters will be saved here
       return self.home
   
   def __call__(self):
       return f"{self.home()}/ppo_spin_{self.Nspin}_{self.inspin}-{self.outspin}_c_{self.numcontrollers}"
       
        

class ModelDoesNotExistError(Exception):
    def __init__(self):
        self.message = "Model not found in the current database!"
        super().__init__(self.message)

class DirectoryDoesNotExistError(Exception):
    def __init__(self, global_exp_path):
        self.message = "Directory not found in {}!".format(global_exp_path)
        super().__init__(self.message)


class Experiment:
    def __init__(self, experiment_name: str = "pipeline_alpha", ip1=None, ip2=None, 
                 Nspin: int = None, inspin: int = None, outspin: int = None, 
                 draws: int = None, fid_noisy: bool = False, 
                 ham_noisy: bool = False, noises: np.ndarray= np.linspace(0,0.1,11),
                 fid_threshold: float = 0.99, runs: int = 100, chances: int = 10, timeout: int = 1080000,
                 verbose: bool = False, respawn_from_checkpoint: bool = True,
                 run_until_completion_its=600000, 
                 run_until_told_to_stop=False, 
                 use_fixed_ham: bool = False, opt_train_size: int = 100,
                 records_update_rate: float = 1e5, ):
        
                
        # initialize with a bunch of parameters that you want your experiment to run for e.g. models, ind vars etc.
        
        self.experiment_name = experiment_name
        assert isinstance(self.experiment_name, str), "Experiment name needs to be a string. Think of something memorable."
        
        self.ip1 = ip1  # independent parameter 1
        self.ip2 = ip2  # ..
        
        self.run_until_completion_its=run_until_completion_its, 
        self.run_until_told_to_stop=run_until_told_to_stop
        self.spin=Nspin
        self.inspin=inspin
        self.outspin=outspin
        
        self.args = dict(nspin=Nspin, 
                    in_spin=inspin, 
                    out_spin=outspin,
                    timeout=timeout, 
                    draws=draws, 
                    fid_noisy=fid_noisy, 
                    ham_noisy=ham_noisy, 
                    verbose=verbose, 
                    testing=False,
                    run_until_completion_its=run_until_completion_its, 
                    run_until_told_to_stop=run_until_told_to_stop, 
                    use_fixed_ham=use_fixed_ham, 
                    opt_train_size=opt_train_size,
                    records_update_rate=records_update_rate)
        
        self.models = ["ppo", "lbfgs", "nmplus", "snob"]
        

        self.noises: np.ndarray = noises
        self._save_results, self._checkpoint_respawn= True, respawn_from_checkpoint # continue data collection where you left off
        self.fid_threshold = fid_threshold
        self.controllers = runs
        self.filename = self.get_experiment_name()
        self.chances=chances
        
    
    def get_experiment_name(self):
        return ExperimentNamer(experiment_name=self.experiment_name,
                               Nspin = self.spin, inspin = self.inspin, 
                               outspin=self.outspin, 
                               numcontrollers=self.controllers)()
        
    
    def init_chosen_models(self, model_choices):
        
        choices_for_now = {"ppo":PPO_en, 
                           "lbfgs":LBFGS, 
                           "snob": SNOB, 
                           "adam": Adam,
                           "nmplus": NMPlus
                           } 
        inits = {}
        for choice in model_choices:
            if choice not in choices_for_now:
                raise ModelDoesNotExistError
            inits[choice]=choices_for_now[choice]
        
        return inits

    def run_var_noise(self, model_choices=None):
        "1 controller is obtained after 1 independent run of an optimizer model"
        if model_choices is None:
            model_choices= self.models
        
        if not isinstance(model_choices, list):
            assert isinstance(model_choices, str), "model choices need to be str list of elements from {}".format(self.models)
            model_choices = [model_choices]
        
        if self._checkpoint_respawn and os.path.exists(self.filename):
            self.results = json.load(open(self.filename))
            print(self.results["ppo"].keys())
        else:
            self.results = {model_name:{} for model_name in model_choices}
            
        bfgs_pr_flag=True 
        for noise in self.noises:
            model_inits=self.init_chosen_models(self.results)
            for model_name in model_inits:
                # dont run this for variable noise for lbfgs for now
                if model_name == "lbfgs":
                    cond=self.spin not in self.results[model_name]
        
                else:
                    cond = noise not in self.results[model_name]
                # to ensure we can checkpoint, stop the data collection and still 
                # start re-running before recomputing any prior noise level
                # which become strings instead of floats after json reloading
                for key in list(self.results[model_name].keys()):
                    if isinstance(key, str):
                        if key == str(noise): 
                            cond = False
                        elif key == str(self.spin):
                            cond = False
                if cond:
                    i=0;j=0
                    while i < self.controllers:
                        try:
                            x = model_inits[model_name](**self.args)
                            x.fid_threshold = self.fid_threshold
                            if model_name != "lbfgs":
                                x.env.noise = noise 
                            else:
                                x.noise = noise
                            x.run()
                            
                            if model_name == "lbfgs":
                                cond=self.spin not in self.results[model_name]
                            else:
                                cond = noise not in self.results[model_name]
                            
                            if cond:
                                # record single experimental run set
                                if model_name == "lbfgs":
                                    self.results[model_name][self.spin] = {}
                                else:
                                    self.results[model_name][noise] = {}
                                for label in x.record:
                                    if model_name == "lbfgs":
                                        self.results[model_name][self.spin][label]=[x.record[label]] 
                                    else:
                                        self.results[model_name][noise][label]=[x.record[label]] 
                            else:
                                # update single experimental run set
                                for label in x.record:
                                    if model_name == "lbfgs":
                                       self.results[model_name][self.spin][label].append(x.record[label])  
                                    else:
                                        self.results[model_name][noise][label].append(x.record[label]) 
                                
                            i+=1
                            #print(f"results={results} \n i={i}")
                            print(f"i={i}, model_name {model_name} {noise}")
                        except Exception as e:
                            print(e)
                            j+=1
                            if j > self.chances:
                                break
                    
                    if self._save_results:
                        json.dump(self.results, open(self.filename, 'w'))
                        if model_name=="lbfgs" and bfgs_pr_flag is True: # don't want to lie now when I'm skipping some runs
                            print(f"saved {model_name} {noise} {i}")
                            bfgs_pr_flag=False
                        elif model_name!="lbfgs":
                            print(f"saved {model_name} {noise} {i}")
                    
    def run_var_spins(self, model_choices=None, spins: list = None, transitions: list = None):
        
        if model_choices is None:
            model_choices= self.models
        
        if not isinstance(model_choices, list):
            assert isinstance(model_choices, str), "model choices need to be str list of elements from {}".format(self.models)
            model_choices = [model_choices]
        
        if self._checkpoint_respawn and os.path.exists(self.filename):
            results = json.load(open(self.filename))
            print(results)
        else:
            results = {model_name:{} for model_name in model_choices}
            
        if spins is None:
            spins = range(3,11,1)
        if transitions is None:
            transitions = [2]*len(spins)
            
        assert len(spins) == len(transitions), "spins and transitions must have the same len: {} != {}".format(len(spins), len(transitions))
            
        for spin, outspin in zip(spins, transitions):
            model_inits=self.init_chosen_models(results)
            for model_name in model_inits:
                if spin not in results[model_name]:
                    i=0;j=0
                    while i < self.controllers:
                        try:
                            self.args["nspin"] = spin
                            self.args["out_spin"] = outspin
                            x = model_inits[model_name](**self.args)
                            x.fid_threshold = self.fid_threshold

                            x.run()
                            
                            if spin not in self.results[model_name]:
                                # record single experimental run set
                                self.results[model_name][spin] = {}
                                for label in x.record:
                                    self.results[model_name][spin][label]=[x.record[label]] 
                            else:
        
                                # record the entire distribution and do stuff later
                                for label in x.record:
                                    self.results[model_name][spin][label].append(x.record[label])      
                            i+=1
                            #print(f"results={results} \n i={i}")
                            print(f"i={i}, model_name {model_name} sp {spin}")
                        except Exception as e:
                            print(e)
                            j+=1
                            if j > self.chances:
                                break
                    
                if self._save_results:
                    json.dump(results, open(self.filename, 'w'))
                    print(f"saved {model_name} {spin} {i}")
                    

    def singlerun_ccollector(self, model_choices=None, custom_args: Dict =None):
        "all controllers obtained from a single run of an optimizer"
        # change filename
        self.filename += ".le"
        
        if model_choices is None:
            model_choices= self.models
        
        if not isinstance(model_choices, list):
            assert isinstance(model_choices, str), "model choices need to be str list of elements from {}".format(self.models)
            model_choices = [model_choices]
        
        if self._checkpoint_respawn and os.path.exists(self.filename):
            self.results = json.load(open(self.filename))
            print(self.results["ppo"].keys())
        else:
            self.results = {model_name:{} for model_name in model_choices}
            
        self.args["landscape_exploration"] = True
        self.args["save_topc"] = self.controllers
        
        if custom_args:
            if not isinstance(custom_args, Dict):
                raise TypeError
            for key in custom_args:
                self.args[key] = custom_args[key]
                self.filename += "_"+str(key)+"_"+str(custom_args[key])
        bfgs_pr_flag=True 
        for noise in self.noises:
            model_inits=self.init_chosen_models(self.results)
            for model_name in model_inits:
                # dont run this for variable noise for lbfgs for now
                if model_name == "lbfgs":
                    cond=self.spin not in self.results[model_name]
        
                else:
                    cond = noise not in self.results[model_name]
                # to ensure we can checkpoint, stop the data collection and still 
                # start re-running before recomputing any prior noise level
                # which become strings instead of floats after json reloading
                for key in list(self.results[model_name].keys()):
                    if isinstance(key, str):
                        if key == str(noise): 
                            cond = False
                        elif key == str(self.spin):
                            cond = False
                if cond:
                    if model_name == "lbfgs": # if discrepancy necessary
                        argscopy = self.args.copy()
                        argscopy["run_until_completion_its"] = 1*self.args["run_until_completion_its"]
                        x = model_inits[model_name](**argscopy)
                    else:
                        x = model_inits[model_name](**self.args)
                    x.fid_threshold = self.fid_threshold
                    if model_name != "ppo":
                        x.noise = noise
                    else:
                        x.env.noise = noise 
                    x.run()
                    
                    if model_name == "lbfgs":
                        cond=self.spin not in self.results[model_name]
                    else:
                        cond = noise not in self.results[model_name]
                    
                    if cond:
                        # record single experimental run set
                        if model_name == "lbfgs":
                            self.results[model_name][self.spin] = {}
                        else:
                            self.results[model_name][noise] = {}
                        for label in x.record:
                            if label == "controllers":
                                if model_name == "lbfgs":
                                    self.results[model_name][self.spin]["controller"]= x.record[label] 
                                else:
                                    self.results[model_name][noise]["controller"]= x.record[label]
 
                    print(f"done model_name {model_name} {noise}")
                    
                    if self._save_results:
                        json.dump(self.results, open(self.filename, 'w'))
                        if model_name=="lbfgs" and bfgs_pr_flag is True: # don't want to lie now when I'm skipping some runs
                            print(f"saved {model_name} {noise}")
                            bfgs_pr_flag=False
                        elif model_name!="lbfgs":
                            print(f"saved {model_name} {noise}")
               
                    
    def singlerun_ccollector_nstoch_sampling(self, model_choices=None):
        # change filename
    
        if self.args['use_fixed_ham']:
            self.filename += ".le_nsh"
        else:
            self.filename += ".le_sh"
        
        if model_choices is None:
            model_choices= self.models
        
        if not isinstance(model_choices, list):
            assert isinstance(model_choices, str), "model choices need to be str list of elements from {}".format(self.models)
            model_choices = [model_choices]
        
        if self._checkpoint_respawn and os.path.exists(self.filename):
            self.results = json.load(open(self.filename))
        else:
            self.results = {model_name:{} for model_name in model_choices}
            
        self.args["landscape_exploration"] = True
        self.args["save_topc"] = self.controllers
        


        for noise in self.noises:
            model_inits=self.init_chosen_models(self.results)
            for model_name in model_inits:

                cond = noise not in self.results[model_name]

                for key in list(self.results[model_name].keys()):
                    if isinstance(key, str):
                        if key == str(noise): 
                            cond = False

                if cond:
                    print(model_name)
                    x = model_inits[model_name](**self.args)
                    x.fid_threshold = self.fid_threshold
                    if model_name != "ppo":
                        x.noise = noise
                    else:
                        x.env.noise = noise 
                    x.run()
                    
                    cond = noise not in self.results[model_name]
                    
                    if cond:
                        # record single experimental run set
                        self.results[model_name][noise] = {}
                        for label in x.records:
                            self.results[model_name][noise][label]= x.records[label]
 
                        print(f"done model_name {model_name} {noise}")
                    
                    if self._save_results:
                        json.dump(self.results, open(self.filename, 'w'))
                        print(f"saved {model_name} {noise}")
        
    def load(self):
        # load pre-existing experiments if passed an option. could come in handy...
        raise NotImplementedError
        

def run_experiments_single_controller_set_with_le():
    args = get_noise_analysis_args()
    exp = Experiment(args.exp_name, 
                     Nspin=args.nspin, 
                     inspin=args.inspin,
                     outspin=args.outspin,
                     fid_threshold=args.fid_threshold,
                     fid_noisy=args.fid_noisy,
                     ham_noisy=args.ham_noisy,
                     noises=np.linspace(0,args.max_noise, args.noise_res), 
                     respawn_from_checkpoint=args.respawn_from_checkpoint, 
                     verbose=args.verbose,
                     run_until_told_to_stop=True,
                     run_until_completion_its=args.run_until_completion_its,
                     runs=args.num_controllers, 

                     )
    exp.singlerun_ccollector()


def run_controller_getter_without_landscape_exploration():
    args = get_noise_analysis_args()
    exp = Experiment(args.exp_name, 
                     Nspin=args.nspin, 
                     inspin=args.inspin,
                     outspin=args.outspin,
                     fid_threshold=args.fid_threshold,
                     fid_noisy=args.fid_noisy,
                     ham_noisy=args.ham_noisy,
                     noises=np.linspace(0,args.max_noise, args.noise_res)[:], 
                     draws=args.draws,
                     respawn_from_checkpoint=args.respawn_from_checkpoint, 
                     verbose=args.verbose,
                     run_until_told_to_stop=args.run_until_told_to_stop,
                     run_until_completion_its=args.run_until_completion_its,
                     runs=args.num_controllers)
    
    exp.run_var_noise(args.algo_name)

def run_ppo_test():
    trial_exp = Experiment("pipeline_ppo_experiments_2", Nspin=5, inspin=0, outspin=2, fid_threshold=0.0, 
                            ham_noisy=True, run_until_told_to_stop=True, run_until_completion_its=1e6,
                            runs=1000, noises=np.linspace(0,0.1,11)[2:3])
    
    for lam, gamma in zip([0.8,0.2,0.8,0.2],[0.8,0.8,0.2,0.2]):
        
        trial_exp.singlerun_ccollector(model_choices="ppo", custom_args={"lam":lam, "gamma": gamma})

if __name__=='__main__':
    run_experiments_single_controller_set_with_le()
