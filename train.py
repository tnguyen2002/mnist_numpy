
import data
import torch 
from torch.utils.data import DataLoader
import numpy as np
from utils import softmax, relu

lr = 1e-4
torch.manual_seed(1)
def main():
    train_dataset, test_dataset = data.get_dataset()
    train_loader, test_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)

        p = forward(X, W1, W2, b1, b2)
        loss = loss_funct(y, p)



def loss_funct(y, p):
    return -np.sum(y * np.log(p))

def forward(X, W1, W2, b1, b2):
    Z1 = X @ W1 + b1 
    Z1 = relu(0, Z1)
    Z2 = Z1 @ W2 + b2
    Z2 = relu(0, Z2)
    Z = softmax(Z2)
    return Z
    



def backward(p, y, X, z1):
    dz1_dW1 = X
    dz2_dW2 = z1
    dl_dz2 = p - y
    dz2_dz1 = W2
    dz2_db2 = 1
    dz1_db1 = 1

    dl_dW1 = dl_dz2 * dz2_dz1 * dz1_dW1
    dl_dW2 = dl_dz2 * dz2_dW2
    dl_db2 = dl_dz2 * dz2_db2
    dl_db1 = dl_dz2 * dz2_dz1 * dz1_db1

    W1 = W1 - lr * dl_dW1
    W2 = W2 - lr * dl_dW2
    b1 = b1 - lr * dl_db1
    b2 = b2 - lr * dl_db2




if __name__ == "__main__":
    main()