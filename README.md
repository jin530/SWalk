# SWalk
This is the official code for the WSDM 2022 paper: <br>[`S-Walk: Accurate and Scalable Session-based Recommendation with Random Walks.`] (https://arxiv.org/abs/2201.01091).</br>


We implemented our model based on the session-recommedndation framework [**session-rec**](https://github.com/rn5l/session-rec), and you can find the other session-based models and detailed usage on there.</br> 
Thanks for sharing the code.

**`README.md` and the comments in source code will be updated, again.**

The slides can be found [here](https://drive.google.com/file/d/1qMBHALxZqH7b6g7kmlKdN6V21GeSrgDU/view?usp=sharing).

## Dataset
Datasets can be downloaded from: </br>
https://drive.google.com/drive/folders/1ritDnO_Zc6DFEU6UND9C8VCisT0ETVp5

- Unzip any dataset file to the data folder, i.e., rsc15-clicks.dat will then be in the folder data/rsc15/raw 
- Run a configuration with the following command:
For example: ```python run_preprocesing.py conf/preprocess/window/rsc15.yml```

## Basic Usage
- Change the expeimental settings and the model hyperparameters using a configuration file `*.yml`. </br>
- When a configuration file in conf/in has been executed, it will be moved to the folder conf/out.
- Run `run_config.py` with configuaration folder arguments to train and test models. </br>
For example: ```python run_confg.py conf/in conf/out```

## Running SLIST
- The yml files for slist used in paper can be found in `conf/save_swalk`

## Requirements
- Python 3
- NumPy
- Pyyaml
- SciPy
- Sklearn
- Pandas
- Psutil

<!--
## Citation
Please cite our papaer:
```
@inproceedings{}
```
-->
