[![DOI](https://zenodo.org/badge/806409571.svg)](https://zenodo.org/doi/10.5281/zenodo.13769225)

# Bridging Cognitive Maps

This repository contains the code for the experiments and figures used in [A Hierarchical Active Inference Model of Spatial Alternation Tasks and the Hippocampal-Prefrontal Circuit](https://www.nature.com/articles/s41467-024-54257-3) by Toon Van de Maele, Bart Dhoedt, Tim Verbelen, and Giovanni Pezzulo. 

## Installation

The experiments require *Python 3.11* (we ran using Python 3.11.6). The easiest way to set up the requirements is by creating a virtualenvironment and running: 
```
pip install -r requirements.txt
```
and install the repo in editable mode by running:
```
python setup.py develop 
``` 

The figures of the papers assume that trained models, and dataset are in the `data` folder, located in the project root. They can be downloaded using this [link](https://drive.google.com/drive/folders/1zwax2Q12B2WlhYp9xXvYrqzcnvgh0Oy8?usp=sharing).  

## Running the experiments 

The cognitive maps can be trained using the scripts in `experiments/model-learning`:  

- **The sequence dataset** can be generated using the `generate_full_explore_dataset.py` script. 
- **The navigation model** can be trained using the `train_navigation_*.py` script, where the * can be either a cscg or a hmm, depending on which model you need. This requires the dataset to be in the `data/` folder. 
- The navigation models can be evaluated using the resepective `evaluate_planning_and_inference_*.py`. Which will measure the success rate, when the agent starts in each possible pose in the maze, and is tasked to go to each of the corridors. NOTE: these scripts point to a location of a model to evaluate. 
- **The task model** can be trained using the `train_task_cscg_loc.py` script. 

The paper experiments and figures are located in `experiments/figures`. Where dedicated notebooks exist for each of the experiments. 

## Acknowledgments
The code for training the clone structured cognitive graphs comes from [CSCG](https://github.com/vicariousinc/naturecomm_cscg). The active inference implementation relies on [PyMDP](https://github.com/infer-actively/pymdp). 


## Citation

If you find the code useful, please refer to our work using:

```
@article{VandeMaele2024,
  title = {A hierarchical active inference model of spatial alternation tasks and the hippocampal-prefrontal circuit},
  volume = {15},
  ISSN = {2041-1723},
  url = {http://dx.doi.org/10.1038/s41467-024-54257-3},
  DOI = {10.1038/s41467-024-54257-3},
  number = {1},
  journal = {Nature Communications},
  publisher = {Springer Science and Business Media LLC},
  author = {Van de Maele,  Toon and Dhoedt,  Bart and Verbelen,  Tim and Pezzulo,  Giovanni},
  year = {2024},
  month = nov 
}
```