import torch
import torch.nn as nn
import torch.optim as optim

# Constants that may be required
invkT = 1.0 / (0.0019872041 * 300)
RT = 8.314 * 300


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
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
