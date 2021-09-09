## About
This repository contains all necessary the code to replicate results of paper: **Adversarial examples by perturbing high-level features in intermediate decoder layers**

## Dependencies
We use libraries with versions: PyTorch 1.9.0 and  Foolbox 3.3.1. Geomloss library and pretrained ImageNet models can be installed using pip:
```
pip install geomloss
pip install pytorch_pretrained_gans
pip install efficientnet-pytorch
```
## Pretrained MNIST models
This repository contains weights of pretrained models used in the MNIST experiments. Model are in PyTorch.


Weights are stored in:
```
models/runs
```

Model definitions are in 

```
models/models.py
```

Easiest way to access and use the pretrained models is to use classes prepared in the ``` models/pretrained.py ``` file.

## MNIST experiments
### Generate MNIST data

To generate additional MNIST samples, use ```sample_data.py```. To replicate our results, run with following parameters:
```shell
python sample_data.py -g ali-mnist -c mnist-cls -o data --seed 1 -m grid
python sample_data.py -g ali-mnist -c mnist-cls -o data --seed 1 -m row
```

### Experiments

To run custom experiments, modify the .json file and run ```main.py``` file. To replicate our results, use provided ```experiment.json``` file and run:
```shell
python main.py -p "experiment.json" -i "data/sampled_data_grid.pickle" -o "results"
```

To generate data used in our figures, run follwoing:
```shell
python compare_difference.py -i "data/sampled_data_row.pickle" -o "results"
python compare_cw_attacks.py -i "data/sampled_data_grid.pickle" -o "results"
```

## ImageNet experiments

To encode images to their latent representation to run following:
```shell
python imagenet_encode.py -p <folder with images to encode>
```

"/images_animals/encoded/encoded_z_0.npy"
"/images_broccoli/encoded/encoded_z_0.npy"

To replicate our imagenet results, run:
```shell
python imagenet_attack.py -p "experiment.json" -o "results" -z z_animals.npy -i z_broccoli.npy
```

## How to replicate figures
Figures for both MNIST and ImageNet experimets can be create using their respective ```create_figures.py``` file.

```shell
python create_figures.py -i <path to folder with results> -o <path to output folder>
```

