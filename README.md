# RobChar - Robust Characterisation of Quantum Controls and Control Algorithms

Monte Carlo ROBustness CHARacterization and analysis of static controllers obtained using various gradient and gradient-free methods 
This is code accompanying the work "Statistically Characterising Robustness and Fidelity of Quantum Controls and
Quantum Control Algorithms (2022)" [here](https://arxiv.org/abs/2207.07801). The quantum model is an effective Heisenberg XXZ chain. 
In the paper, we don't enable the Z interaction available in the code. 

To setup, please run 

```bash
git clone https://github.com/qyber-black/robchar_public/
cd robchar_public
pip3 install -r requirements.txt
```

Important/Likely Novel dependencies are sklearn, skquant and pytorch.

## Top view Functionality 

The code is mostly modular with every optimization algorithm being a subclass of `LBFGS` which is also an optimizer. The API to port your own optimizer is by overriding the `run` function. See `qnewton.py` for many examples. The `noise_analysis.py` file then lets you port your optimizer for collecting controllers for the spin chain  

There is a separate class for handling cachable Monte Carlo simulations that only needs to be run once after the controllers have been collected. 

## Reproducing the figures

Data for reproducing the figures is provided with the code. Just running the python files individually or running the following

```bash
bash generate_all_figures.sh
```
will produce all the figures in the `gray_scale_adjusted_paperfigs` directory along with their grayscale representations in a `gray` sub-folder.

## Running experiments to generate new controllers

To reproduce the data from the paper,

run
```bash
bash get_paper_data.sh
```
This will take a long time depending on your compute resources. But this should effectively populate/repopulate the `experiments` directory so that running the figure reproduction incantation above will produce figures based on your version of the experiments. On the other hand, if you look at `get_paper_data.sh`, you will realize that there are a number of distint sub-problems (e.g. a particular spin chain transition) 
delimited by a single line. Each is a separete experiment and can be run on its own.

## Running experiments for other spin problems

Please have a look at `get_paper_data.sh` and `parser.py`. It should be straightforward to run a variant of the following with custom args to generate controllers for other problems not covered by the paper.

```python
python noise_analysis.py --nspin `CHAIN_LENGTH` --outspin `OutputState` \
--inspin `InputState` \
--fid_threshold 0.1 --run_until_told_to_stop True \
--run_until_completion_its `num_objective_function_calls` --num_controllers 100
```

# Locations

The code is developed and maintained on [qyber\black](https://qyber.black)
at https://qyber.black/spinnet/code-robchar

This code is mirrored at
* https://github.com/qyber-black/robchar_public

The mirrors are only for convenience, accessibility and backup.

# Citation

I. Khalid, C. A. Weidner, E. A. Jonckheere, S. G. Shermer, F. C. Langbein. **RobChar: Robust Characterisation of Quantum Controls and Control Algorithms**, V1.0.1. Code, https://github.com/Qyber-black/robchar_public, 22nd July 2022.
[[DOI:10.5281/zenodo.6891381]](https://doi.org/10.5281/zenodo.6891381)
