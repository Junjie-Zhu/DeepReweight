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
parser.add_argument('--input', '-i', type=str, default='./dih_AAAA')

# target (required in float type)
parser.add_argument('--target', '-t', type=float, default=0.0)

parser.add_argument('--output', '-o', type=str, default='./result.csv')
parser.add_argument('--steps', '-s', type=int, default=1000)
parser.add_argument('--optimizer', '-opt', type=str, default='adam')
parser.add_argument('--learning_rate', '-l', type=float, default=5e-3)
parser.add_argument('--verbose', '-v', action='store_true', default=False)
parser.add_argument('--schedule', action='store_true', default=False)
parser.add_argument('--early_stop', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cpu')  # Currently not support CUDA

parser.add_argument('--acc', type=float, default=1000)
args, _ = parser.parse_known_args()

# Constants that may be required
invkT = 1.0 / (0.0019872041 * 300)
RT = 8.314 * 300


def main():
    # Extract phi-psi values from trajectory
    alpha, zeta, cross = get_data(args.input)
    cross = torch.Tensor(cross).to(args.device)

    ramachandran = []
    for i in range(len(alpha)):
        ramachandran.append(np.histogram2d(zeta[i], alpha[i],
                                           bins=(24, 24),
                                           range=[[-180, 180], [-180, 180]])[0])
    ramachandran = torch.Tensor(np.array(ramachandran)).to(args.device)

    # Start training
    print('Start training...')
    model = DeepReweighting(initial_parameter=torch.zeros((1, 1, 24, 24))).to(args.device)
    optimizer = get_optimizer(args.optimizer, model, args.learning_rate)

    min_lr = 1e-8
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=20,
                                                           factor=0.6, verbose=True,
                                                           threshold=1e-2, min_lr=min_lr, cooldown=5)
    early_stopping = EarlyStopping(patience=5)

    cmap_record = []
    prediction_record = []
    step_record = []

    time0 = time.time()
    for steps in tqdm.tqdm(range(args.steps)):

        model.train()

        delta_E = model(ramachandran)

        loss_calculate = loss(delta_E, cross, args.target)
        losses, Prop_pred = loss_calculate

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Getting training record
        if args.verbose and (steps % 10 == 0):
            print(f'##### STEP {steps} #####')
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradient Norm: {param.grad.norm().item()}")

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
        if torch.isnan(model.update).any().item() or torch.isnan(Prop_pred):
            print("NaN encoutered, exiting...")
            break

        step_record.append(steps + 1)
        cmap_record.append(model.update.detach().numpy())
        prediction_record.append(Prop_pred.detach().numpy())

    print(time.time() - time0)
    # Save re-weighting results
    if args.output.endswith('.csv'):
        output_file = args.output
    else:
        suffix = args.output.split('.')[-1]
        output_file = args.output.replace(suffix, 'csv')

    '''
    with open(output_file, 'w') as f:
        f.write('steps,cmap,rg\n')
        for i in range(0, len(step_record), 100):
            f.write('%d,%.6f,%.2f\n' % (step_record[i],
                                        cmap_record[i],
                                        prediction_record[i]))
    '''

    print('final cross account: %.2f' % (prediction_record[-1]))


def get_data(file_name):
    """
    :param file_name:
    :return:
    """
    filenames = ['%s_right_all.dat' % file_name, '%s_wrong_all.dat' % file_name]

    alpha = []
    zeta = []
    cross = []

    for filename in filenames:
        with open(filename, 'r') as f:
            for lines in f.readlines():
                if lines[0] != '#':
                    data = lines.split()

                    zeta.append([float(data[4]), float(data[10]), float(data[16])])
                    alpha.append([float(data[5]), float(data[11]), float(data[17])])
                    cross.append(filenames.index(filename))

    return alpha, zeta, cross


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
    _loss = torch.abs(Prop_exp - Prop_pred) / weights.shape[0]
    
    return _loss, Prop_pred


class DeepReweighting(nn.Module):

    def __init__(self, initial_parameter):
        super(DeepReweighting, self).__init__()

        self.update = nn.Parameter(initial_parameter, requires_grad=True)

    def forward(self, ramachandran):
        """
        :param ramachandran: The phi-psi distribution calculated on (24 x 24) grid
        :return: Energy added from CMAP for each conformation
        """

        # The input should be of shape [Batch * 24 * 24]
        output = ramachandran * self.update
        return torch.sum(output.squeeze(0), dim=(-2, -1))


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


def get_optimizer(opt_name, model, lr):
    if opt_name.lower() == 'adam':
        optimizer = torch.optim.Adam([model.update], lr)
    elif opt_name.lower() == 'sgd':
        optimizer = torch.optim.SGD([model.update], lr=lr, momentum=0.8)
    elif opt_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW([model.update], lr)
    elif opt_name.lower() == 'asgd':
        optimizer = torch.optim.ASGD([model.update], lr)
    elif opt_name.lower() == 'rprop':
        optimizer = torch.optim.Rprop([model.update], lr)
    elif opt_name.lower() == 'adadelta':
        optimizer = torch.optim.Adadelta([model.update], lr)
    elif opt_name.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop([model.update], lr)
    else:
        raise ValueError('Optimizer <%s> currently not supported' % opt_name)

    return optimizer


if __name__ == '__main__':
    main()
