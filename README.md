# Generating Deontic Obligations From Utility-Maximizing Systems
This repository contains the code used for model checking and search in the paper submitted to AIES 2022.
The code is offered as-is, and should be recognized as a preliminary offering for review and replication purposes only - not as a unitary package for use in future research or production.

## Requirements
To run the relevant code, the following python packages are required.
Most are available from conda or pypi.
boss can be acquired from https://github.com/henrymoss/BOSS.

igraph
numpy
matplotlib
nltk
emukit
tqdm
mdptoolbox
chime
boss
cProfile

PRISM must also be installed, and a path to its binary must be included in model_check.py in the checkPCTL function.

## Instructions
To run one of the experiments reported in the paper, run the examples.py script.
The MDP executed on is defined in examples.py, and the parameters for Bayesian optimization are defined in bayes_opt.py.
GrammarAcquisitionOptimizer.py is the genetic algorithm adapted from BOSS that uses model checking to generate valid maximizing formulas.
model_check.py contains the algorithms used for model checking an MDP.
NOTE: we sometimes had difficulty sampling from the grammar parameter space (e.g. random\_design.get\_samples()). This function was provided as part of BOSS, but can sometimes take a long time to complete. We have found that running in debug mode, calling get\_samples() in the debugging console, sending a keyboard interrupt, and then calling get\_samples() again, then letting the code run from there will make the get\_samples() function run quickly and correctly. We have no explanation for this.
