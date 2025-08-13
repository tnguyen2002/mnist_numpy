
import data
import torch 
from torch.utils.data import DataLoader
import numpy as np
from utils import softmax, relu

torch.manual_seed(1)
def main():
    train_dataset, test_dataset = data.get_dataset()
    train_loader, test_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)





def forward(X, W1, W2, b1, b2):
    Z1 = X @ W1 + b1 
    Z1 = relu(0, Z1)
    Z2 = Z1 @ W2 + b2
    Z2 = relu(0, Z2)
    Z = softmax(Z2)
    



# def backward():




if __name__ == "__main__":
    main()