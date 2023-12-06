"""
Running this script for reweighting requires two input files:
Phi-Psi information: Nframes * Nangles * 2
E-Prop information: Nframes * 2 (E in the first column and Prop in the second)
"""

import time
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, default='./LJ_rg.dat')

# initial epsilon and target rg (required in float type)
parser.add_argument('--epsilon', '-e', type=float, required=True)
parser.add_argument('--target', '-t', type=float, required=True)

parser.add_argument('--output', '-o', type=str, default='./result.csv')
parser.add_argument('--steps', '-s', type=int, default=1000)
parser.add_argument('--learning_rate', '-l', type=float, default=5e-3)
parser.add_argument('--verbose', '-v', action='store_true', default=False)
parser.add_argument('--schedule', action='store_true', default=False)
parser.add_argument('--early_stop', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cpu')  # Currently not support CUDA

parser.add_argument('--acc', type=float, default=1000)
parser.add_argument('--test', type=float, default=1.0)
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
    epsilon_old = args.epsilon * args.acc  # Prevent loss of float accuracy
    Exp_prop = args.target

    # Start training
    print('Start training...')
    model = DeepReweighting(initial_parameter=epsilon_old).to(args.device)
    optimizer = torch.optim.Adam([model.update], args.learning_rate)
    # optimizer = torch.optim.SGD([model.update], lr=args.learning_rate, momentum=0.8)
    # optimizer = torch.optim.AdamW([model.update], args.learning_rate)
    # optimizer = torch.optim.ASGD([model.update], args.learning_rate)
    # optimizer = torch.optim.Rprop([model.update], args.learning_rate, etas=(0.6, args.test), step_sizes=(1e-06, 50))
    # optimizer = torch.optim.Adadelta([model.update], args.learning_rate)
    # optimizer = torch.optim.RMSprop([model.update], args.learning_rate)

    schedule = False
    early_stop = False
    min_lr = 1e-8
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=20,
                                                           factor=0.6, verbose=True,
                                                           threshold=1e-2, min_lr=min_lr, cooldown=5)
    early_stopping = EarlyStopping(patience=5)

    epsilon_record = []
    prediction_record = []
    step_record = []

    time0 = time.time()
    for steps in tqdm.tqdm(range(args.steps)):

        model.train()

        delta_E = model(lj, epsilon_old)

        loss_calculate = loss(delta_E, rg, Exp_prop)
        losses, Prop_pred = loss_calculate

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Getting training record
        if args.verbose:
            print(f'##### STEP {steps} #####')
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient Norm: {param.grad.norm().item()}")

            print(f'Step {steps}: loss {losses}\n')

        # Check training state
        if args.schedule:
            scheduler.step(losses)
            if optimizer.param_groups[0]['lr'] <= min_lr * 1.5:
                print('converged')
                break

        if args.early_stop:
            early_stopping(losses)
            if early_stopping.early_stop:
                break

        # check NaN
        if torch.isnan(model.update) or torch.isnan(Prop_pred):
            print("NaN encoutered, exiting...")
            break

        step_record.append(steps + 1)
        epsilon_record.append(model.update.detach().numpy() / args.acc)
        prediction_record.append(Prop_pred.detach().numpy())

    print(time.time() - time0)
    # Save re-weighting results
    if args.output.endswith('.csv'):
        output_file = args.output
    else:
        suffix = args.output.split('.')[-1]
        output_file = args.output.replace(suffix, 'csv')

    with open(output_file, 'w') as f:
        f.write('steps,epsilon,rg\n')
        for i in range(len(step_record)):
            f.write('%d,%.6f,%.2f\n' % (step_record[i], epsilon_record[i], prediction_record[i]))

    print('final epsilon: %.6f, final rg: %.2f' % (epsilon_record[-1], prediction_record[-1]))


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


def check_rationality(rg_list, target):
    if torch.max(rg_list) > target > torch.min(rg_list):
        print(torch.max(rg_list), torch.min(rg_list))
    else:
        raise ValueError('target not rational')


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

    def forward(self, lj, epsilon_old, **kwargs):
        """
        :param lj: lj potential for each conformation
        :param epsilon_old: initial epsilon value
        :return: delta energy for each conformation
        """

        delta_E = lj * torch.sqrt(self.update / epsilon_old) - lj
        return delta_E


class EarlyStopping:
    """from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/"""

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):

        if self.best_loss is None:
            self.best_loss = val_loss

        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                # print('INFO: Early stopping')
                self.early_stop = True


if __name__ == '__main__':
    main()
