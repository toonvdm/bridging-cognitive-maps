# Bridging Cognitive Maps

This repository contains the code for the experiments and figures used in [Bridging Cognitive Maps: a Hierarchical Active Inference Model of Spatial Alternation Tasks and the Hippocampal-Prefrontal Circuit](https://arxiv.org/abs/2308.11463) by Toon Van de Maele, Bart Dhoedt, Tim Verbelen, and Giovanni Pezzulo. 

If you find the code useful, please refer to our work using:

```
@misc{vandemaele2023bridging,
      title={Bridging Cognitive Maps: a Hierarchical Active Inference Model of Spatial Alternation Tasks and the Hippocampal-Prefrontal Circuit}, 
      author={Toon Van de Maele and Bart Dhoedt and Tim Verbelen and Giovanni Pezzulo},
      year={2023},
      eprint={2308.11463},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC}
}
```

## Installation

The experiments were ran using python 3.11.6. The easiest way to set up the requirements is by creating a virtualenvironment and running: 
```
pip install -r requirements.txt
```
and install the repo dependencies by running
```
python setup.py install 
``` 

The figures of the papers assume that trained models, and dataset are in the `data` folder, located in the project root. They can be downloaded using this [link](todo_fill_in_link).  

## Running the experiments 

The cognitive maps can be trained using the scripts in `experiments/model-learning`:  

- **The sequence dataset** can be generated using the `generate_full_explore_dataset.py` script. 
- **The navigation model** can be trained using the `train_navigation_*.py` script, where the * can be either a cscg or a hmm, depending on which model you need. This requires a sequence dataset of the environment. These models can be evaluated using the resepective `evaluate_planning_and_inference_*.py`. Which will measure the success rate, when the agent starts in each possible pose in the maze, and is tasked to go to each of the corridors.
- **The task model** can be trained using the `train_task_cscg_loc.py` script. 

The paper experiments and figures are located in `experiments/figures`. Where dedicated notebooks exist for each of the experiments. 


## Acknowledgments
The code for training the clone structured cognitive graphs comes from [CSCG](https://github.com/vicariousinc/naturecomm_cscg). The active inference implementation relies on [PyMDP](https://github.com/infer-actively/pymdp). 


