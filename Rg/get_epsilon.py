"""
Running this script for reweighting requires two input files:
Phi-Psi information: Nframes * Nangles * 2
E-Prop information: Nframes * 2 (E in the first column and Prop in the second)
"""

import numpy as np
import tqdm
import torch
import matplotlib.pyplot as plt

from model import DeepReweighting, EarlyStopping, loss

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, default='./LJ_rg_p53.dat')

# initial epsilon and target rg (required in float type)
parser.add_argument('--epsilon', '-e', type=float, required=True)
parser.add_argument('--target', '-t', type=float, required=True)

parser.add_argument('--output', '-o', type=str, default='./epsilon.dat')
parser.add_argument('--steps', '-s', type=int, default=1000)
parser.add_argument('--learning_rate', '-l', type=float, default=5e-3)
parser.add_argument('--verbose', '-v', action='store_true', default=False)
parser.add_argument('--schedule', action='store_true', default=False)
parser.add_argument('--early_stop', action='store_true', default=False)
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
    optimizer = torch.optim.Adam([model.update], args.learning_rate)

    schedule = False
    early_stop = False
    min_lr = 1e-8
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=20,
                                                           factor=0.6, verbose=True,
                                                           threshold=1e-2, min_lr=min_lr, cooldown=5)
    early_stopping = EarlyStopping(patience=200)

    epsilon_record = []
    prediction_record = []
    step_record = []

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
        if torch.isnan(model.update):
            print("NaN encoutered, exiting...")
            break

        step_record.append(steps + 1)
        epsilon_record.append(model.update.detach().numpy() / 1000)
        prediction_record.append(Prop_pred.detach().numpy())

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

    print('final epsilon: %.6f' % epsilon_record[-1])


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


if __name__ == '__main__':
    main()
