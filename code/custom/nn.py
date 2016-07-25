import numpy as np
import pandas as pd
#import theano

ACTIVATION_FUNC_TYPES = ['sigmoid', 'tanh', 'relu']

class NeuralNetwork(object):
    def __init__(self, num_nodes_per_layers, X, Y,
                 activation_func_type=None,
                 learning_rate=0.01, regularization=0.1,
                 data_format='numpy', num_passes=10**3,
                 loss_function='cross_entropy_loss', verbose=True):
        if not isinstance(num_nodes_per_layers, list):
            raise ValueError('Must provide a list of Number of Nodes per '
                             'each Hidden layer')

        self._num_nodes_hidden_layers = num_nodes_per_layers[:-1]
        if not activation_func_type:
            self.activation_func_types = [ACTIVATION_FUNC_TYPES[0] for x in len(self._num_nodes_hidden_layers)]
        elif isinstance(activation_func_type, str) \
            and activation_func_type in ACTIVATION_FUNC_TYPES:
            self.activation_func_types = [activation_func_type for x in len(self._num_nodes_hidden_layers)]
        elif isinstance(activation_func_type, list):
            self.activation_func_types = [x if x in ACTIVATION_FUNC_TYPES
                                          else ACTIVATION_FUNC_TYPES[0]
                                          for x in activation_func_type]

        self._num_classes = len(self._num_nodes_per_layers[-1])
        self._epsilon = learning_rate
        self._reg_lambda = regularization
        self.hidden_layers = []
        self.y_target = Y
        self.data_format = data_format
        self.num_passes = num_passes
        self.verbose = verbose
        if isinstance(self.X, pd.DataFrame):
            self.num_input = len(X.columns)
        if isinstance(self.X, np.ndarray):
            self.num_input = self.X.shape[1]
        if verbose:
            print('NN has: \n\t-> {0} output classes; '
                  '\n\t-> {1} hidden layers;'
                  '\n\t-> This amount of Neurons per hidden layer {2}'
                  .format(self._num_classes, len(self._num_nodes_hidden_layers),
                          self._num_nodes_hidden_layers))

    def feed_forward(self, epoch, init_nn=False):
        """
            Constructs NeuralNetwork
        :return:
        """
        if init_nn and self.verbose:
            print('Starting to construct network from beginning...')
        else:
            if self.verbose:
                print('Applying feedforward for epoch: {} ...'.format(epoch))
        for layer in xrange(len(self._num_nodes_hidden_layers)):
            output_size = self._num_nodes_hidden_layers[layer]
            # randomly initialize for each layer
            if init_nn:
                weights = np.random.randn(input_size, output_size)
                b = np.ones(1, output_size)
            else:
                # TODO
                pass
            # specifics for layer 1 and others:
            if layer == 0:
                input_size = self.num_input
                layer_input = self.x
            else:
                input_size = self._num_nodes_hidden_layers[layer-1]
                layer_input = self.hidden_layers[layer-1].activation_func
            hidden_layer = HiddenLayer(nn_input_dim=input_size, X=layer_input,
                                       W=weights, b=b, reg_lambda=self._reg_lambda,
                                       epsilon=self._epsilon,
                                       activation_func_type=self.activation_func_types[layer])
            self.hidden_layers.append(hidden_layer)

    def back_propagation(self, epoch):
        if self.verbose:
            print('Starting backprop algorithm at epoch: {}...'.format(epoch))
        num_hidden_layers=len(self.hidden_layers)
        for i in xrange(num_hidden_layers):
            cur_layer_idx = num_hidden_layers - i
            prediction = self.hidden_layers[cur_layer_idx].prediction
            self.hidden_layers[cur_layer_idx].compute_gradient(prediction)

    def gradient_descent(self):
        if self.verbose:
            print('Initializing gradient descent')
        for i in self.num_passes:

            params = dict(epoch=i)
            if i == 0:
                params['init_nn'] = True
            print('######\nFeedforward pass for Epoch {}'.format(i))
            self.feed_forward(**params)
            print('------\nBackward prop. pass for Epoch {}'.format(i))
            self.back_propagation(epoch=i)


    def cross_entroypy_loss(self):
        pass
        #probs = np.
        #corect_logprobs = -np.log(probs[range(num_examples), y])


class HiddenLayer(object):
    def __init__(self, nn_input_dim, X, W, b, activation_func_type='tanh',
                 reg_lambda=1, epsilon=0.1):
        self.nn_input_dim = nn_input_dim
        self.reg_lambda = reg_lambda
        self.epsilon = epsilon
        self._x = self.x(X)
        self._w = self.w(W)
        self._z = self.z(X)
        self._b = b
        self._prediction = self._prediction(self.z)
        self._activation_func = self.activation_func(activation_func_type)
        self.gradient = None

    @property
    def x(self):
        return self._x

    @property.setter
    def x(self, xi):
        self._x = xi

    @property
    def z(self):
        return self._z

    @property
    def w(self):
        return self._w

    @property.setter
    def w(self, W):
        self._w = W

    @property.setter
    def z(self, W):
        """ TODO: expand for for storage types
        :return:
        """
        if isinstance(self.x, (np.ndarray, pd.DataFrame)):
            self.z = self.X.dot(W) + self.b

    @property
    def b(self):
        return self._b

    @property.setter
    def b(self, b_val):
        if isinstance(self.x, (np.ndarray, pd.DataFrame)):
            self._b = b_val

    @property
    def prediction(self):
        return self._prediction

    @property.setter
    def prediction(self, val='softmax'):
        if val == 'softmax':
            self._prediction = self.softmax()


    @property
    def activation_func(self):
        return self._activation_func

    @property.setter
    def activation_func(self, func_type):
        if func_type == 'sigmoid':
            self._activation_func = self.sigmoid()
        elif func_type == 'tanh':
            self._activation_func = self.tanh()
        elif func_type == 'relu':
            self._activation_func = self.relu()

    def sigmoid(self):
        if isinstance(self.x, (np.ndarray, pd.DataFrame)):
            return 1/(1 + np.exp(-self.z))

    def diff_sigmoid(self):
        return self.sigmoid()*(1-self.sigmoid())

    def tanh(self):
        if isinstance(self.x, (np.ndarray, pd.DataFrame)):
            return np.tanh(self.z)

    def diff_tan(self):
        return 1-np.pow(self.tanh())

    def relu(self):
        """ ReLu - rectified linear function
        :return:
        """
        if isinstance(self.x, (np.ndarray, pd.DataFrame)):
            return self.x * (self.x > 0)

    def diff_relu(self):
        if isinstance(self.x, (np.ndarray, pd.DataFrame)):
            return 1. * (self.x > 0)

    def softmax(self):
        if isinstance(self.X, (np.ndarray, pd.DataFrame)):
            e = np.exp(self.x - np.max(self.x))  # prevent overflow
            if e.ndim == 1:
                return e / np.sum(e, axis=0)
            else:
                return e / np.ndarray([np.sum(e, axis=1)]).T  # ndim = 2

    def compute_gradient(self, delta):
        if isinstance(self.X, np.ndarray):
            self.gradient = (self.activation_func.T).dot(delta) + self.w * self.reg_lambda

    def update_weights_and_bias(self):
        self.w += -self.epsilon * self.gradient
        self.b += -self.epsilon * self.gradient

