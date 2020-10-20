import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from emnist_train.utils.definitions import ROOT_DIR

class ARGS(object):
    """docstring for ARGS"""
    def __init__(self):
        super(ARGS, self).__init__()
        self.lr = 0.001
        self.momentum = 0.9
        

def build_model():

	 # Define the model, while tuning the size of our hidden layer
    model = nn.Sequential(nn.Linear(784, 64),
                          nn.ReLU(),
                          nn.Linear(64, 1))
    return model

def get_dataset():

    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(ROOT_DIR)), 'data/')

    train_dir = os.path.join(DATA_DIR, 'train.csv')
    train_df = pd.read_csv(train_dir, header=None)

    test_dir = os.path.join(DATA_DIR, 'test.csv')
    test_df = pd.read_csv(test_dir, header=None)

    label = 0
    features = list(range(1, 785))

    train_x = train_df[features].values
    train_y = train_df[label].values

    test_x = test_df[features].values
    test_y = test_df[label].values

    return train_x, train_y, test_x, test_y

def get_dataloader(train_x, train_y, test_x, test_y):

    # Define our data loaders
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)
    train_dataset = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_x = torch.Tensor(test_x)
    test_y = torch.Tensor(test_y)
    test_dataset = TensorDataset(test_x, test_y)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_dataloader, test_dataloader

    


def train_model(args):

    # Load dataset
    train_x, train_y, test_x, test_y = get_dataset()

    # Get Dataloader
    train_dataloader, test_dataloader = get_dataloader(train_x, train_y, test_x, test_y)


    # Define model
    model = build_model()
    criterion = nn.MSELoss()

    # Tune hyperparameters in our optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    epochs = 20
    for e in range(epochs):
        for batch_id, (data, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pred = model(data)
            label = label.view(-1,1)
            loss = criterion(y_pred, label)
            
            loss.backward()
            optimizer.step()
            print(f'{e}/{batch_id} : {loss.item()}')


    val_mse = 0
    num_batches = 0
    # Evaluate accuracy on our test set
    with torch.no_grad():
        for i, (data, label) in enumerate(test_dataloader):
            num_batches += 1
            y_pred = model(data)
            mse = criterion(y_pred, label.view(-1,1))
            val_mse += mse.item()


    avg_val_mse = (val_mse / num_batches)

if __name__ == '__main__':
    args = ARGS()
    train_model(args)