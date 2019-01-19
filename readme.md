# DROW-pytorch
* This repository is an unofficial implementation of 
[DROW: Real-Time Deep Learning-Based Wheelchair Detection in 2-D Range Data](
https://ieeexplore.ieee.org/document/7797258) with PyTorch.
* The official implementation is [here](
https://github.com/VisualComputingInstitute/DROW). Thank authors for opening 
their codes.

## Installation
1. Clone thie repository.
2. Install Python dependencies.
```
pip install -r requirements.txt
```

## Dataset
1. The dataset is provided in [official repository](
https://github.com/VisualComputingInstitute/DROW/releases). Download `DROW v1 
Dataset (Wheelchairs + Walkers)`.
2. Move training data to `DROW-data/train`. Move testing data to 
`DROW-data/test`.
```
├── DROW-data
│   ├── LICENSE
│   ├── README.md
│   ├── test
│   │   ├── run_t_2015-11-26-11-22-03.bag.csv
│   │   ├── run_t_2015-11-26-11-22-03.bag.wa
│   │   ├── run_t_2015-11-26-11-55-45.bag.wc
│   │   ├── …
│   ├── train
│   │   ├── lunch_2015-11-26-12-04-23.bag.csv
│   │   ├── lunch_2015-11-26-12-04-23.bag.wa
│   │   ├── …
```

## Training
```
python main.py TRAIN --use_cuda --save_model
```
The model trained is saved in `checkpoints`.<br>
If you have not put dataset in `DROW-data` as above, you need show 
the directory of dataset clearly.
```
python main.py TRAIN --train_data_dir="the/directory/of/training/data" --use_cuda --save_model 
```
## Testing
```
python main.py TEST
```
Again, you need show the directory of dataset clearly, if you have not put 
dataset in `DROW-data`.
```
python main.py TEST --test_data_dir="the/directory/of /testing/data"
```
All results will be saved in `results`. The results of data in the same file 
will be in one directory. One scan labeled generates one picture.
