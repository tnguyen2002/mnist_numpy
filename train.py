
import data
import torch 
from torch.utils.data import DataLoader
import numpy as np
from utils import softmax, relu, normalize
# import matplotlib.pyplot as plt

lr = 1e-4
epochs = 2
batch_size = 32
torch.manual_seed(1)
def main():
    train_dataset, test_dataset = data.get_dataset()
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle =True)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    W1 = np.random.rand(784, 256)
    W2 = np.random.rand(256, 10)
    b1 = np.random.rand(1, 256)
    b2 = np.random.rand(1, 10)
    for epoch in range(epochs):
        print("epoch: ", epoch)
        loss = 0
        for step, (X, y) in enumerate(train_loader):
            X = X.reshape(-1 , 784).numpy()
            # print("X.shape", X.shape)
            # print("W1.shape", W1.shape)
            # print("W2.shape", W2.shape)
            # print("b1.shape", b1.shape)
            # print("b2.shape", b2.shape)
            y = y.reshape(1, -1).numpy()
            y_one_hot = np.zeros((y.size, 10))
            y_one_hot[np.arange(y.size), y] = 1
            p, Z1 = forward(X, W1, W2, b1, b2)
            # print("p.shape", p.shape)
            # print("p", p)
            # print("y.shape", y.shape)
            # print("loss.shape", loss.shape)
            # print("loss", loss)
            # print("y_one_hot shape", y_one_hot.shape)
            # print("p shape", p.shape)
            loss += loss_funct(y_one_hot, p).sum()
            # return
            # W1, W2, b1, b2 = backward(p, y, X, Z1, W1, W2, b1, b2)
            # print("step", step)
            dl_dW1, dl_dW2 , dl_db1, dl_db2 = backward(p, y_one_hot, X, Z1, W1, W2, b1, b2)
            #update
            W1 = W1 - lr * dl_dW1
            W2 = W2 - lr * dl_dW2
            b1 = b1 - lr * dl_db1
            b2 = b2 - lr * dl_db2
        avg_loss = loss / len(train_dataset)
        print("avg_loss", avg_loss)

    print("Testing")
    num_test_examples = len(test_dataset)
    num_correct = 0
    for step, (X, y)  in enumerate(test_loader):
        # plt.imshow(X.permute(1, 2, 0))
        X = X.reshape(-1 , 784).numpy()
        y = y.numpy()
        p, _ = forward(X, W1, W2, b1, b2)
        y_hat = np.argmax(p, axis=1)
        num_correct += np.sum(y == y_hat)
    print("Test Accuracy", num_correct / num_test_examples)






def loss_funct(y, p):
    # loss = -np.log(p[(np.arange(p.shape[0]), y.T)])
    loss = -1/len(y) * np.sum(np.sum(y * np.log(p)))
    return loss

def forward(X, W1, W2, b1, b2):
    Z1 = X @ W1 + b1 
    Z1 = normalize(Z1)
    Z1 = relu(Z1)
    Z2 = Z1 @ W2 + b2
    # print("before softmax", Z2)
    p = softmax(Z2)
    return p, Z1
    



def backward(p, y, X, Z1, W1, W2, b1, b2):
    dz1_dW1 = X
    dz2_dW2 = Z1
    dl_dz2 = p - y
    dz2_dz1 = W2
    dz2_db2 = np.ones((1, X.shape[0]))
    dz1_db1 = np.ones((1, X.shape[0]))
    # print("dz1_dW1 shape", dz1_dW1.shape)
    # print("dz2_dW2 shape", dz2_dW2.shape)
    # print("dl_dz2 shape", dl_dz2.shape)
    # print("dz2_dz1 shape", dz2_dz1.shape)
    # print("dz2_db2 shape", dz2_db2.shape)
    # print("dz1_db1 shape", dz1_db1.shape)

    dl_dW1 = dz1_dW1.T @ dl_dz2 @ dz2_dz1.T 
    dl_dW2 = dz2_dW2.T @ dl_dz2  
    dl_db2 = dz2_db2 @ dl_dz2 
    dl_db1 = dz1_db1 @ dl_dz2 @ dz2_dz1.T


    return dl_dW1, dl_dW2, dl_db1, dl_db2

    # return W1 - lr * dl_dW1,  W2 - lr * dl_dW2 , b1 - lr * dl_db1, b2 - lr * dl_db2





if __name__ == "__main__":
    main()