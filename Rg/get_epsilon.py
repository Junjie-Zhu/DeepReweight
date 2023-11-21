"""
Running this script for reweighting requires two input files:
Phi-Psi information: Nframes * Nangles * 2
E-Prop information: Nframes * 2 (E in the first column and Prop in the second)
"""

import sys

import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, default='./LJ_rg_p53.dat')

# initial epsilon and target rg (required in float type)
parser.add_argument('--epsilon', '-e', type=float, required=True)
parser.add_argument('--target', '-t', type=float, required=True)

parser.add_argument('--output', '-o', type=str, default='./epsilon.dat')
parser.add_argument('--steps', '-s', type=int, default=1000)
parser.add_argument('--learning_rate', '-l', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cpu')  # Currently not support CUDA
args, _ = parser.parse_known_args()

# Constants that may be required
invkT = 1.0 / (0.0019872041 * 300)
RT = 8.314 * 300


def main():
    # Extract phi-psi values from trajectory
    lj, rg = get_data(args.input)
    lj = torch.Tensor(lj).to(args.device)
    rg = torch.Tensor(rg).to(args.device)
	
    # Extract target property from simulation
    epsilon_old = args.epsilon * 1000  # Prevent loss of float accuracy
    Exp_prop = args.target

    # Start training
    print('Start training...')
    model = DeepReweighting(initial_parameter=epsilon_old).to(args.device)

    epsilon_record = []
    prediction_record = []
    step_record = np.arange(0, args.steps, 1)

    for steps in tqdm.tqdm(range(args.steps)):
        loss_local, epsilon_update, Prop_pred = one_iter(model, lj_sim=lj, Prop_sim=rg, Prop_exp=Exp_prop, epsilon_old=epsilon_old)

        # Getting training record
        #print(f'##### STEP {steps} #####')
        #for name, param in model.named_parameters():
        #    if param.grad is not None:
        #        print(f"Gradient Norm: {param.grad.norm().item()}")

        #print(f'Step {steps}: loss {loss_local}\n')

        epsilon_record.append(epsilon_update.detach().numpy() / 1000)
        prediction_record.append(Prop_pred.detach().numpy())

    ploting(step_record, epsilon=epsilon_record, rg=prediction_record)
    print('final epsilon: %.8f' % epsilon_record[-1])


def loss(delta_E, Prop_sim, Prop_exp):
    """
    IMPORTANT: Here the loss should be pre-defined before reweighting

    :param delta_E: Delta energy of each conformation from simulation
    :param Prop_sim: Target property of conformations in simulation
    :param Prop_exp: Expected property of the system
    :return: loss value
    """
    # Reweighting
    weights = torch.exp(- delta_E * invkT)

    # Predict Reweighted property vector
    weighted_Prop_sum = torch.mul(Prop_sim, weights)
    Prop_pred = torch.sum(weighted_Prop_sum, dim=0) / torch.sum(weights)

    # Calculate loss
    _loss = torch.abs(Prop_exp - Prop_pred) / Prop_exp

    return _loss, Prop_pred


class DeepReweighting(nn.Module):

    def __init__(self, initial_parameter):
        super(DeepReweighting, self).__init__()

        self.update = nn.Parameter(torch.Tensor([initial_parameter]), requires_grad=True)
        self.optimizer = optim.Adam([self.update], args.learning_rate)

    def forward(self, lj, epsilon_old, **kwargs):
        """
        :param lj: lj potential for each conformation
        :param epsilon_old: initial epsilon value
        :return: delta energy for each conformation
        """
        
        delta_E = lj * torch.sqrt(self.update / epsilon_old) - lj
        return delta_E

    def fit(self, delta_E, Prop_sim, Prop_exp):
        loss_calculate = loss(delta_E, Prop_sim, Prop_exp)
        self.loss = loss_calculate[0]

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        return loss_calculate[1]


def one_iter(model, lj_sim, Prop_sim, Prop_exp, epsilon_old):
    model.train()

    delta_E = model(lj_sim, epsilon_old)
    Prop_pred = model.fit(delta_E, Prop_sim, Prop_exp)

    return model.loss, model.update, Prop_pred


def get_data(file_name):
    E_LJ = []
    rg = []
    file = open(file_name, "r")
    file.readline()
    for line in file.readlines():
        data = line.split()
        E_LJ.append(float(data[1]))
        rg.append(float(data[2]))
    file.close()
    E_LJ_np = np.array(E_LJ)
    rg_np = np.array(rg)
    return E_LJ_np, rg_np


def ploting(steps, epsilon, rg):
    font_title = {"family": 'Times New Roman',
                  "style": 'italic',
                  "weight": "bold",
                  "color": "black",
                  "size": 24}
    font_label = {"family": 'Times New Roman',
                  # "style":'italic',
                  "weight": "bold",
                  "color": "black",
                  "size": 18}
    font_legend = {"family": 'Times New Roman',
                   # "style":'italic',
                   "weight": "bold",
                   # "color":"black",
                   "size": 16}
    fig = plt.figure(figsize=[20, 5.5])
    ax = fig.add_subplot(111)
    ax.plot(steps, epsilon, "blue", linewidth=3, label='Epsilon')
    ax2 = ax.twinx()
    ax2.plot(steps, rg, 'green', linewidth=3, label="Target")
    ax.legend(loc="center right", prop=font_legend, bbox_to_anchor=(1.003, 0.61))
    ax2.legend(loc="center right", prop=font_legend, bbox_to_anchor=(0.995, 0.5))
    ax.grid(False)
    ax.set_xlabel("Steps", font_label)
    ax.set_ylabel("Epsilon (Kcal/mol)", font_label)
    ax2.set_ylabel("Target Function", font_label)
    # plt.tick_params(labelsize=16)
    labels = ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]
    [label.set_fontweight("bold") for label in labels]
    [label.set_fontsize(16) for label in labels]
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["top"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)

    import os
    dir_name = os.getcwd()
    dir_name = dir_name.split("\\")[-1]
    plt.title(dir_name, fontdict=font_title)
    # plt.plot(x,z)
    # plt.show()
    plt.savefig("reweighting_result.png", dpi=600)


def check_rationality(rg_list, target):
    if target < torch.max(rg_list) and target > torch.min(rg_list):
        print(torch.max(rg_list), torch.min(rg_list))
    else:
        raise ValueError('target not rational')

if __name__ == '__main__':
    main()
