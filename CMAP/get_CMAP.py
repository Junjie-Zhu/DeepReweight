"""
Running this script for reweighting requires two input files:
Phi-Psi information: Nframes * Nangles * 2
E-Prop information: Nframes * 2 (E in the first column and Prop in the second)
"""

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dihedral', '-d', type=str, default='./dihedral.dat')
parser.add_argument('--target', '-t', type=str, default='./target.dat')
parser.add_argument('--cmap', '-o', type=str, default='./CMAP.dat')
parser.add_argument('--steps', '-s', type=int, default=50)
parser.add_argument('--learning_rate', '-l', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cpu')  # Currently not support CUDA
args, _ = parser.parse_known_args()


# Constants that may be required
invkT = 1.0 / (0.0019872041 * 300)
RT = 8.314 * 300


def main():
    # Extract phi-psi values from trajectory
    phi, psi = [], []
    with open(args.dihedral, 'r') as f:
        for lines in f.readlines():
            if lines[0] != '#':
                phi.append(float(lines.split()[1]))
                psi.append(float(lines.split()[2]))

    ramachandran = []
    for i in range(len(phi)):
        ramachandran.append(np.histogram2d([phi[i]], [psi[i]],
                                           bins=(24, 24),
                                           range=[[-180, 180], [-180, 180]])[0])
    ramachandran = torch.Tensor(np.array(ramachandran)).to(args.device)

    # Extract target property from simulation
    Exp_prop = [phi[::50], psi[::50]]
    Prop = np.histogram2d(Exp_prop[0], Exp_prop[1],
                          bins=(24, 24),
                          range=[[-180, 180], [-180, 180]])[0]

    Prop = torch.Tensor(Prop) / len(Exp_prop[0])
    Prop = Prop.to(args.device)

    # Start training
    print('Start training...')
    model = CMAP_model().to(args.device)
    CMAP_old = torch.zeros((24, 24)).to(args.device)
    # CMAP_old = Prop - ramachandran

    for steps in range(args.steps):
        loss_local, CMAP_update, Prop_pred = one_iter(model, ramachandran, Prop)

        # Getting training record
        print(f'##### STEP {steps} #####')
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Gradient Norm: {param.grad.norm().item()}")

        print(f'Step {steps}: loss {loss_local}\n')
        CMAP_old = CMAP_update

    np.savetxt(args.cmap, CMAP_old.detach().numpy())


def loss(E_CMAP, Prop_sim, Prop_exp):
    """
    IMPORTANT: Here the loss should be pre-defined before reweighting

    :param E_CMAP: CMAP energy of each conformation from simulation
    :param Prop_sim: Target property of conformations in simulation
    :param Prop_exp: Expected property of the system
    :return: loss value
    """
    # Reweighting
    weights = torch.exp(- E_CMAP * invkT)

    # Predict Reweighted phi-psi vector
    weighted_Prop_sum = Prop_sim * weights[:, None, None]
    Prop_pred = torch.sum(weighted_Prop_sum, dim=0) / torch.sum(weights)

    # Calculate loss
    _loss = nn.MSELoss()(Prop_exp.view(1, 576), Prop_pred.view(1, 576))
    # _loss = SparseMSELoss()(Prop_exp.view(1, 576), Prop_pred.view(1, 576))
    # _loss = nn.KLDivLoss(reduction='batchmean')(Prop_exp.view(1, 576), Prop_pred.view(1, 576))
    return _loss, Prop_pred


class CMAP_model(nn.Module):
    """
    Here we treat CMAP as a (24 x 24) parameter tensor to optimize it using preset optimizers.
    """

    def __init__(self):
        super(CMAP_model, self).__init__()

        self.update = nn.Parameter(torch.zeros((1, 1, 24, 24)), requires_grad=True)
        self.optimizer = optim.Adam([self.update], args.learning_rate)

    def forward(self, ramachandran):
        """
        :param ramachandran: The phi-psi distribution calculated on (24 x 24) grid
        :return: Energy added from CMAP for each conformation
        """
        # The input should be of shape [Batch * 24 * 24]
        output = ramachandran * self.update
        return torch.sum(output.squeeze(0), dim=(-2, -1))

    def fit(self, pred, sim, exp):
        loss_calculate = loss(pred, sim, exp)
        self.loss = loss_calculate[0]

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        return loss_calculate[1]


class SparseMSELoss(nn.Module):
    def __init__(self):
        super(SparseMSELoss, self).__init__()

    def forward(self, y_true, y_pred):
        # Create a mask for non-zero elements in both y_true and y_pred
        mask = (y_true != 0) & (y_pred != 0)

        # Apply the mask to extract non-zero elements
        y_true_non_zero = y_true[mask]
        y_pred_non_zero = y_pred[mask]

        # Compute the mean squared error for non-zero elements
        mse = torch.mean((y_true_non_zero - y_pred_non_zero) ** 2)

        return mse


def one_iter(model, Prop_sim, Prop_exp):
    model.train()

    E_CMAP = model(Prop_sim)
    Prop_pred = model.fit(E_CMAP, Prop_sim, Prop_exp)

    return model.loss, model.update.squeeze().squeeze(), Prop_pred


if __name__ == '__main__':
    main()
