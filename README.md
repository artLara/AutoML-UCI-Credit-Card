# AutoML-UCI-Credit-Card

## Description
AutoML project based on CI Credit Card Dataset for preprocessing, training and testing pipelines.

## Instalation
``` bash
pip install -r requirements.txt
```

## Run scripts
The main.py file manage all the process and recive different commands, for run all the process use the next command
``` bash
python3 main.py all src/cfg.yaml
```

## Configuration file
The configuration file specify variables and paths useful to create and manage all pipelines from a single YAML file. 
run: specify  the project name, path and logging lavel.
algorithms: describes hiperparameters and architectures of machine learning algorithms.
data: useful to set paths for read and write files.
details: detailed description of the current experiment.

## Preprocessing
Delete unknow categorical data rows and ID column.

## Feature Engineering
Normalization and split data using 80% for training and 20% for testing.

## Model Generation
Use Grid Search CV to optimize hiper-parameters of SVM, Random Forest and Decision Tree algorithms. Metrics like R2 and RMSE are using to meassure performance.
