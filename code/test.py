import sklearn
import sklearn.datasets
import numpy as np
from custom import nn

# Display plots inline and change default figure size
# import matplotlib.pyplot as plt
# %matplotlib inline
# matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

if __name__ == '__main__':
    #random generate dataset
    np.random.seed(123)
    X, y = sklearn.datasets.make_moons(200, noise=0.2)
    # plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    nodes_per_layer = [3, 3, 2] # binary classification, so 2 at end
    aft = ['sigmoid', 'tanh']
    lr = 0.01
    reg = 0.1
    num_passes = 10
    nn = nn.NeuralNetwork(
        X=X, Y=y, num_nodes_per_layers=nodes_per_layer,
        activation_func_type=aft, learning_rate=lr,
        regularization=reg, data_format='numpy',
        num_passes=num_passes, loss_function='cross_entropy_loss',
        verbose=True)
    nn.gradient_descent()