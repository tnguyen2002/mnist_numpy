import numpy as np

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis = 0)

# correct solution:
# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0) # only difference



def relu(X):
    return X * (X > 0)

def normalize(X):
    row_sums = X.sum(axis=1)
    new_matrix = X / row_sums[:, np.newaxis]
    return new_matrix


if __name__ == "__main__":
    p = np.array([[.4, .3, .3],
                  [.1, .6, .3]])
    y = np.array([0, 0])
    y_hat = np.argmax(p, axis = 1)
    print("y", y.shape)
    print("y_hat", y_hat.shape)
    print("p", p.shape)
    num_correct = np.sum(y == y_hat)
    print(num_correct.sum())
    