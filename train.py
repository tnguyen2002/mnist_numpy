
import data
import torch 
from torch.utils.data import DataLoader
import numpy as np
from utils import softmax, relu, normalize
import matplotlib.pyplot as plt

lr = 1e-3
epochs = 30
batch_size = 32
torch.manual_seed(1)
def main():
    train_dataset, test_dataset = data.get_dataset()
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle =True)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    W1 = np.random.rand(784, 256) * 0.01
    W2 = np.random.rand(256, 10) * 0.01
    b1 = np.zeros((1, 256))
    b2 = np.zeros((1, 10))
    num_correct = 0
    # print("Training Acc before Training", num_correct / len(train_dataset))
    for epoch in range(epochs):
        loss = 0
        num_correct = 0
        for step, (X, y) in enumerate(train_loader):
            X = X.reshape(-1 , 784).numpy()
            y = y.numpy()
            y_one_hot = np.zeros((y.size, 10))
            y_one_hot[np.arange(y.size), y] = 1
            p, Z1, A1 = forward(X, W1, W2, b1, b2)
            loss += loss_funct(y_one_hot, p).sum()
            dl_dW1, dl_dW2 , dl_db1, dl_db2 = backward(p, y_one_hot, X, Z1, W1, W2, b1, b2, A1)
            #update
            W1 = W1 - lr * dl_dW1
            W2 = W2 - lr * dl_dW2
            b1 = b1 - lr * dl_db1
            b2 = b2 - lr * dl_db2
        for step, (X, y)  in enumerate(train_loader):
            X = X.reshape(-1 , 784).numpy()
            y = y.numpy()
            p, _, _= forward(X, W1, W2, b1, b2)
            y_hat = np.argmax(p, axis=1)
            num_correct += np.sum(y == y_hat)
        print("Epoch:", epoch, "loss:", loss / len(train_dataset),  "Training_Acc: ", num_correct / len(train_dataset))



    print("Testing")
    num_test_examples = len(test_dataset)
    num_correct = 0

    for step, (X, y)  in enumerate(test_loader):
        X = X.reshape(-1 , 784).numpy()
        y = y.numpy()
        p, _, _= forward(X, W1, W2, b1, b2)
        y_hat = np.argmax(p, axis=1)
        num_correct += np.sum(y == y_hat)
        if (step % 1000 == 0):
            img = X.reshape(28, 28, 1)
            plt.imshow(img)
            plt.show()
            print(y_hat)
    print("Test Accuracy", num_correct / num_test_examples)






def loss_funct(y, p):
    eps = 1e-12
    loss = -np.mean(np.sum(y * np.log(p + eps), axis = 1))
    return loss

def forward(X, W1, W2, b1, b2):
    Z1 = X @ W1 + b1 
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    p = softmax(Z2)
    return p, Z1, A1
    



def backward(p, y, X, Z1, W1, W2, b1, b2, A1):
    dz1_dW1 = X
    dl_dz2 = p - y
    dz2_dA1 = W2
    mask = Z1 > 0
    dl_dA1 = dl_dz2 @ dz2_dA1.T


    dl_dW1 =  dz1_dW1.T  @ (dl_dA1 * mask)
    dl_db1 = (dl_dA1 * mask).sum(axis=0)



    dl_dW2 = A1.T @ dl_dz2  
    dl_db2 = dl_dz2.sum(axis = 0)



    return dl_dW1, dl_dW2, dl_db1, dl_db2




if __name__ == "__main__":
    main()