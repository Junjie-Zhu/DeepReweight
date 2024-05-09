# DeepReweight

We developed a novel force field optimization strategy based on an explainable deep learning framework, DeepReweighting, for rapid and precise force field re-parameterization and optimization. DeepReweighting demonstrates a significant increase in re-parameterization efficiency compared to traditional Monte Carlo method and exhibits greater robustness. Please follow the instruction below to use DeepReweighting.

### Installation

1. Clone the repository
```sh
git clone https://github.com/Junjie-Zhu/DeepReweight
```

2. Install the required packages
```sh
pip install tqdm==4.65.0 numpy==1.24.1
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```
Note that any version of the required packages that support the functions used in the code should work.

### Usage

1. Prepare the data for correlating force field

2. Train the DeepReweight model, take CMAP as an example

```sh
python get_CMAP_4_protein_ram.py -i /path/to/input/data -o /path/to/output/cmap -t 0.0
```

Required parameters:
- `-i` or `--input`: the path to the input data
- `-o` or `--output`: the path to the output data
- `-t` or `--target`: the target physical quantity value to fit

Optional parameters:
- `-w` or `--weight`: the weight for eRMSD, default is 1e-2
- `-s` or `--steps`: the number of steps for training the model, default is 1000
- `-opt` or `--optimizer`: the optimizer for training the model, default is 'Adam'
- `-lr` or `--learning_rate`: the learning rate for training the model, default is 0.001
- `--device`: the device for training the model, default is 'cuda:0'