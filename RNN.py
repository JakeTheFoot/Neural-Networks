'''
TODO:
- Add a way to save the recurrent and convolutional layers
- Add a way to load the recurrent and convolutional layers
- Add support for recurrent and convolutional layers in optimizers other than Adam
- Add README in github
'''

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import Matern
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from scipy.signal import convolve2d
from scipy.optimize import minimize
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from collections import deque
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import numpy as np
import statistics
import zipfile
import pickle
import random
import joblib
import copy
import time
import json
import time
import cv2
import gym
import os
import io

# Dense layer


class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):

        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    # Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases

    # Set weights and biases in a layer instance
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

# Dropout layer


class Layer_Dropout:

    # Init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs

        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return

        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,
                                              size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask

# Convolutional layer class


class Layer_Convolutional:

    # Initialization
    def __init__(self, Filters, Padding=0, Biases=0, IsMultipleFilters=True,
                 IsMultipleInputs=True, weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):

        # Set layer variables
        self.Padding = Padding
        self.Biases = Biases
        self.Filters = np.array(Filters, dtype=object)
        self.IsMultipleFilters = IsMultipleFilters
        self.IsMultipleInputs = IsMultipleInputs
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

        # Define blank list
        # to append to
        self.FilterSizes = []

        # Itterate through every filter and append
        # it's size to the FilterSizes list
        for i, kernel in enumerate(self.Filters):

            # Append the size
            self.FilterSizes.append(np.array(self.Filters[i]).shape)

    # ConvolutionalSlicer method
    def ConvolutionalSlicer(self, kernel, SlicerInput, ConvolutionType='Basic_Convolution', Pass='forward', index=0):

        # Set the current sizes
        # (length x hight)
        self.KernelSize = [len(kernel[0]), len(kernel)]
        self.InputSize = [len(SlicerInput[0]), len(SlicerInput)]

        # Calculate output size
        # length x hight
        self.OutputSize = [self.InputSize[0] - self.KernelSize[0] +
                           1, self.InputSize[1] - self.KernelSize[1] + 1]

        # Define blank list
        self.ConvolutionalSlice = []

        # Add value to self object
        self.SlicerInput = SlicerInput

        self.ConvolutionResult = convolve2d(SlicerInput, kernel, mode='valid')

        # If its the foward pass
        # than add a bias
        if Pass == 'forward':

            if np.ndim(self.Biases) > 0:

                # Add bias
                self.ConvolutionResult = np.array(
                    self.ConvolutionResult) + self.Biases[index]

        # Return the reshaped output of
        # the convolution slice after
        # undergoing it's given equation
        return np.reshape(self.ConvolutionResult, self.OutputSize)

# Additive convolution


class Basic_Convolution(Layer_Convolutional):

    # Forward method
    def forward(self, inputs, training):

        self.batch_size = len(model.batch_X)  # ! Change to more general !

        self.XPadded = inputs

        # Padding check
        if self.Padding > 0:

            # Multiple inputs check
            if self.IsMultipleInputs == True:

                self.XPadded = []

                # If true, iterate through
                # inputs and pad
                for matrix in inputs:

                    # Add padding
                    self.XPadded.append(np.pad(matrix, self.Padding))

        # For singular input
        else:

            # Add padding
            self.XPadded = np.pad(self.XPadded, self.Padding)

        # Define blank array
        # for input sizes
        self.InputSize = []

        # Get hight x length
        self.InputSize = [([len(matrix[0]), len(matrix)])
                          for matrix in self.XPadded] if self.IsMultipleInputs == True else [len(self.XPadded[0]), len(self.XPadded)]

        self.output = []

        self.outputPreBatch = []

        # Define a function to perform the convolution for a single input and kernel
        def single_convolution(matrix, kernel, j):
            return self.ConvolutionalSlicer(kernel, matrix, 'Basic_Convolution', 'forward', j)

        # Use ThreadPoolExecutor to parallelize the convolution computation
        with ThreadPoolExecutor() as executor:
            # Itterate through each input
            for i, matrix in enumerate(self.XPadded):
                # And for every kernel
                for j, kernel in enumerate(self.Filters):
                    # Append the output of the convolution
                    self.outputPreBatch.append(
                        executor.submit(single_convolution, matrix, kernel, j).result())

                self.output.append(self.outputPreBatch)

        self.output = np.array(self.outputPreBatch, dtype=object)

    def backward(self, dvalues):

        # Define blank lists to append to
        self.dweights = []
        self.dbiases = []
        self.dinputs = []

        # Iterate through every output (input on
        # the forward pass, since self.output's
        # first dimention is the inputs)
        for i in range(0, self.batch_size):

            # Iterate through every filter index
            for j in range(0, len(self.Filters)):

                # Get the rotated filter (180 degrees)
                self.rotated_filter = np.rot90(self.Filters[j], 2)

                # Convolve the gradient with the rotated filter
                self.dinputs.append(self.ConvolutionalSlicer(
                    self.rotated_filter, np.pad(dvalues[j], 1), 'Basic_Convolution', 'backward'))

                # Append the derivative of the weights at index j
                self.dweights.append(self.ConvolutionalSlicer(
                    dvalues[j], self.XPadded[i], 'Basic_Convolution', 'backward'))
                self.dbiases.append(np.sum(np.sum(dvalues[j])))

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = [np.ones_like(K) for K in self.Filters]
            dL1 = [[[-1 if value < 0 else value for value in row]
                    for row in Filter] for Filter in self.Filters]
            self.dweights = [np.add(dw_matrix, self.weight_regularizer_l1 * np.array(dL1_matrix))
                             for dw_matrix, dL1_matrix in zip(self.dweights, dL1)]

       # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights = [self.dweights[i] + 2 * self.weight_regularizer_l2 *
                             K for i, K in enumerate(self.Filters)]

        ''''
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                self.biases'''

# Flatten layer


class Layer_Flatten:

    # Forward
    def forward(self, inputs, training):

        self.batch_size = len(model.batch_X)  # ! Change to more general !

        # Define output list
        self.output = []

        # Define the shapes of the
        # inputs for backward pass
        self.InputShape = []

        # For every input, append the
        # flattened version of it
        for i, matrix in enumerate(inputs):

            # Append to output
            self.output.append(matrix.ravel())

            # Get the shape of
            # the current input
            self.InputShape.append(matrix.shape)

        self.output = np.concatenate(self.output)

        self.output = np.reshape(self.output, [self.batch_size, -1])

    # Backward
    def backward(self, dvalues):

        self.dvalues = np.ravel(dvalues)

        # Set dinputs as a
        # blank array to be
        # appended to
        self.dinputs = []

        # Set the starting index
        self.start = 0
        self.end = 0

        # For every input in
        # the forward pass
        for i, shape in enumerate(self.InputShape):

            # Multiply the length by
            # hight to find the amount
            # of numbers in the input shape
            self.size = np.prod(shape)

            self.end += self.size
            self.end = int(self.end)

            # For the amount of numbers in
            # the input shape, starting at
            # the end of all the previous
            # amounts of numbers in all of
            # the shapes combined, append
            # those number reshaped to be
            # the size of the inputs into the output
            self.dinputsPreReshape = self.dvalues[self.start: self.end]

            self.dinputs.append(
                self.dinputsPreReshape.reshape(shape[0], shape[1]))

            # Add the amount of numbers
            # used to self.start to find
            # the next starting point
            self.start = self.end
            self.start = int(self.start)

        # Initialize a dictionary to store the sums
        sums = {}

        # Define a function to sum the input matrices with the same shape
        def sum_matrices(input_matrix):
            shape = input_matrix.shape
            if shape not in sums:
                sums[shape] = input_matrix
            else:
                sums[shape] += input_matrix

        # Use ThreadPoolExecutor to parallelize the summing of input matrices
        with ThreadPoolExecutor() as executor:
            # Iterate over the inputs in self.dinputs
            executor.map(sum_matrices, self.dinputs)

        # Create a new array to store the sums
        self.summed_inputs = []

        # Iterate over the keys of the sums dictionary to add the sums to the new array
        for shape, sum_input in sums.items():
            self.summed_inputs.append(sum_input)

        # Convert the summed_inputs array to a NumPy array
        self.dinputs = np.array(self.summed_inputs, dtype=object)

    # forward
    def forward(self, inputs, training):

        self.batch_size = len(model.batch_X)

        # Define output list
        self.output = []

        # Define the shapes of the
        # inputs for backward pass
        self.InputShape = []

        # For every input, apend the
        # flattened version of it
        for i, matrix in enumerate(inputs):

            # Append to output
            self.output.append(matrix.ravel())

            # Get the shape of
            # the current input
            self.InputShape.append(matrix.shape)

        self.output = np.concatenate(self.output)

        self.output = np.reshape(self.output, [self.batch_size, -1])

    # Backward
    def backward(self, dvalues):

        self.dvalues = np.ravel(dvalues)

        # Set dinputs as a
        # blank array to be
        # appended to
        self.dinputs = []

        # Set the starting index
        self.start = 0
        self.end = 0

        # For every input in
        # the forward pass
        for i, shape in enumerate(self.InputShape):

            # Multiply the length by
            # hight to find the amount
            # of numbers in the input shape
            self.size = np.prod(shape)

            self.end += self.size
            self.end = int(self.end)

            # For the amount of numbers in
            # the input shape, starting at
            # the end of all the previous
            # amounts of numbers in all of
            # the shapes combined, append
            # those number reshaped to be
            # the size of the inputs into the output
            self.dinputsPreReshape = self.dvalues[self.start:self.end]

            self.dinputs.append(
                self.dinputsPreReshape.reshape(shape[0], shape[1]))

            # Add the amount of numbers
            # used to self.start to find
            # the next starting point
            self.start = self.end
            self.start = int(self.start)

        # initialize a dictionary to store the sums
        sums = {}

        # iterate over the inputs in self.dinputs
        for input_matrix in self.dinputs:
            shape = input_matrix.shape

            # sum the input matrix with the same shape using advanced indexing and broadcasting
            if shape not in sums:
                sums[shape] = input_matrix
            else:
                sums[shape] += input_matrix

        # create a new array to store the sums
        self.summed_inputs = []

        # iterate over the keys of the sums dictionary to add the sums to the new array
        for shape, sum_input in sums.items():
            self.summed_inputs.append(sum_input)

        # convert the summed_inputs array to a NumPy array
        self.dinputs = np.array(self.summed_inputs, dtype=object)

# Layer_Recurrent


class Layer_Recurrent:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0, initializationMethod="Zeros"):
        self.n_neurons = n_neurons
        # Initialize hiddenVector
        if initializationMethod == "Zeros":
            self.hiddenVector = np.zeros((model.initial_batch_size, n_neurons))
            self.hiddenVectorList = [self.hiddenVector]
        else:
            self.hiddenVector = 0.01 * \
                np.random.randn(model.initial_batch_size, n_neurons)
            self.hiddenVectorList = [self.hiddenVector]
        self.initmethod = initializationMethod
        # Hidden Vector weights initialization
        self.HiddenVectorWeights = 0.01 * np.random.randn(n_neurons, n_neurons)
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        self.inputList = []

        self.timesteps = model.timesteps  # ! TODO make sure this is the correct name

    # reset hidden vector
    def reset(self):

        if self.initmethod == "Zeros":
            self.hiddenVector = np.zeros((len(model.batch_X), self.n_neurons))
        else:
            self.hiddenVector = 0.01 * \
                np.random.randn(len(model.batch_X), self.n_neurons)
        self.hiddenVectorList = [self.hiddenVector]
        self.inputList = []

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs

        self.inputList.append(self.inputs)

        weighted_hidden = np.dot(
            self.hiddenVector, self.HiddenVectorWeights) + self.biases

        # Calculate output values from inputs, weights and biases
        # print('input shape:', inputs.shape)
        # print('weighted hidden shape:', weighted_hidden.shape)
        self.output = np.tanh(self.inputs + weighted_hidden)

        self.hiddenVector = self.output
        self.hiddenVectorList.append(self.hiddenVector)
        self.dinputsT = 1

    # Backward pass
    def backward(self, dvalues):  # !

        self.dinputs = np.zeros_like(self.inputs)
        self.dhiddenVectorWeights = np.zeros_like(self.HiddenVectorWeights)
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)
        dhidden_prev = np.zeros_like(self.hiddenVector)

        for i in reversed(range(self.timesteps)):
            self.dTan = 1 - (self.hiddenVectorList[i] ** 2)
            self.dinputsT = (dvalues + dhidden_prev) * self.dTan
            self.dhiddenVectorWeightsT = np.dot(
                self.hiddenVectorList[i - 1].T, self.dinputsT)
            self.dweightsT = np.dot(self.inputList[i].T, self.dinputsT)
            self.dbiasesT = np.sum(self.dinputsT, axis=0, keepdims=True)

            dhidden_prev = np.dot(self.dinputsT, self.HiddenVectorWeights.T)

            self.dinputs += self.dinputsT
            self.dhiddenVectorWeights += self.dhiddenVectorWeightsT
            self.dweights += self.dweightsT
            self.dbiases += self.dbiasesT

# TODO add hidden weights to regularization
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                self.biases

# Input "layer"


class Layer_Input:

    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs

# ReLU activation


class Activation_ReLU:

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

# Softmax activation


class Activation_Softmax:

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

# Sigmoid activation


class Activation_Sigmoid:

    # Forward pass
    def forward(self, inputs, training):
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

# Linear activation


class Activation_Linear:

    # Forward pass
    def forward(self, inputs, training):
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

# SGD optimizer


class Optimizer_SGD:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If we use momentum
        if self.momentum:

            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.Filters)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.Biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates1
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * \
                layer.dweights
            bias_updates = -self.current_learning_rate * \
                layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.Filters += weight_updates
        layer.Biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# Adagrad optimizer


class Optimizer_Adagrad:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# RMSprop optimizer


class Optimizer_RMSprop:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# Adam optimizer


class Optimizer_Adam:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=1e-3, epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update fully connected layer parameters
    def update_params(self, layer):

        # If it's a convolutional layer
        if hasattr(layer, 'Filters'):

            # If layer does not contain cache arrays, create them filled with zeros
            if not hasattr(layer, 'weight_cache'):
                layer.weight_momentums = []
                layer.weight_cache = []
                for Filter in layer.Filters:
                    layer.weight_momentums.append(np.zeros_like(Filter))
                    layer.weight_cache.append(np.zeros_like(Filter))
                if np.ndim(layer.Biases) > 0:
                    layer.bias_momentums = np.zeros_like(layer.Biases)
                    layer.bias_cache = np.zeros_like(layer.Biases)

            # Update momentum with current gradients
            for i in range(len(layer.Filters)):
                for j in range(len(layer.Filters[i])):
                    for k in range(len(layer.Filters[i][j])):
                        layer.weight_momentums[i][j][k] = self.beta_1 * layer.weight_momentums[i][j][k] + \
                            (1 - self.beta_1) * layer.dweights[i][j][k]
                if np.ndim(layer.Biases) > 0:
                    layer.bias_momentums[i] = self.beta_1 * layer.bias_momentums[i] + \
                        (1 - self.beta_1) * layer.dbiases[i]

            # Get corrected momentum
            # self.iteration is 0 at first pass and we need to start with 1 here
            weight_momentums_correctedK = [momentumK / (1 - self.beta_1 ** (self.iterations + 1))
                                           for momentumK in layer.weight_momentums]
            if np.ndim(layer.Biases) > 0:
                bias_momentums_correctedK = [momentumKB / (1 - self.beta_1 ** (self.iterations + 1))
                                             for momentumKB in layer.bias_momentums]

            # Update cache with squared current gradients
            for i in range(len(layer.Filters)):
                layer.weight_cache[i] = self.beta_2 * layer.weight_cache[i] + \
                    (1 - self.beta_2) * layer.dweights[i]**2
                if np.ndim(layer.Biases) > 0:
                    layer.bias_cache[i] = self.beta_2 * layer.bias_cache[i] + \
                        (1 - self.beta_2) * layer.dbiases[i]**2

            # Get corrected cache
            weight_cache_correctedK = [cache / (1 - self.beta_2 ** (self.iterations + 1))
                                       for cache in layer.weight_cache]
            if np.ndim(layer.Biases) > 0:
                bias_cache_correctedK = [cache / (1 - self.beta_2 ** (self.iterations + 1))
                                         for cache in layer.bias_cache]

            # Vanilla SGD parameter update + normalization with square rooted cache
            for i in range(len(layer.Filters)):
                layer.Filters[i] -= self.current_learning_rate * weight_momentums_correctedK[i] / \
                    (np.sqrt(weight_cache_correctedK[i]) + self.epsilon)
                if np.ndim(layer.Biases) > 0:
                    layer.Biases[i] -= self.current_learning_rate * bias_momentums_correctedK[i] / \
                        (np.sqrt(bias_cache_correctedK[i]) + self.epsilon)

        # If it's a recurrent layer
        elif hasattr(layer, 'HiddenVectorWeights'):
            # If layer does not contain cache arrays,
            # create them filled with zeros
            if not hasattr(layer, 'weight_cache'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.hidden_vector_weight_momentums = np.zeros_like(
                    layer.HiddenVectorWeights)
                layer.hidden_vector_weight_cache = np.zeros_like(
                    layer.HiddenVectorWeights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                layer.bias_cache = np.zeros_like(layer.biases)

            # Update momentum with current gradients
            layer.weight_momentums = self.beta_1 * \
                layer.weight_momentums + \
                (1 - self.beta_1) * layer.dweights
            layer.hidden_vector_weight_momentums = self.beta_1 * \
                layer.hidden_vector_weight_momentums + \
                (1 - self.beta_1) * layer.dhiddenVectorWeights
            layer.bias_momentums = self.beta_1 * \
                layer.bias_momentums + \
                (1 - self.beta_1) * layer.dbiases

            # Get corrected momentum
            # self.iteration is 0 at first pass
            # and we need to start with 1 here
            weight_momentums_corrected = layer.weight_momentums / \
                (1 - self.beta_1 ** (self.iterations + 1))
            hidden_vector_weight_momentums_corrected = layer.hidden_vector_weight_momentums / \
                (1 - self.beta_1 ** (self.iterations + 1))
            bias_momentums_corrected = layer.bias_momentums / \
                (1 - self.beta_1 ** (self.iterations + 1))

            # Update cache with squared current gradients
            layer.weight_cache = self.beta_2 * layer.weight_cache + \
                (1 - self.beta_2) * layer.dweights**2
            layer.hidden_vector_weight_cache = self.beta_2 * layer.hidden_vector_weight_cache + \
                (1 - self.beta_2) * layer.dhiddenVectorWeights**2
            layer.bias_cache = self.beta_2 * layer.bias_cache + \
                (1 - self.beta_2) * layer.dbiases**2

            # Get corrected cache
            weight_cache_corrected = layer.weight_cache / \
                (1 - self.beta_2 ** (self.iterations + 1))
            hidden_vector_weight_cache_corrected = layer.hidden_vector_weight_cache / \
                (1 - self.beta_2 ** (self.iterations + 1))
            bias_cache_corrected = layer.bias_cache / \
                (1 - self.beta_2 ** (self.iterations + 1))

            # Vanilla SGD parameter update + normalization
            # with square rooted cache
            layer.weights -= self.current_learning_rate * \
                weight_momentums_corrected / \
                (np.sqrt(weight_cache_corrected) +
                 self.epsilon)
            layer.HiddenVectorWeights -= self.current_learning_rate * \
                hidden_vector_weight_momentums_corrected / \
                (np.sqrt(hidden_vector_weight_cache_corrected) +
                 self.epsilon)
            layer.biases -= self.current_learning_rate * \
                bias_momentums_corrected / \
                (np.sqrt(bias_cache_corrected) +
                 self.epsilon)

        # If it's not a
        # convolutional layer
        # or recurrent layer
        else:

            # If layer does not contain cache arrays,
            # create them filled with zeros
            if not hasattr(layer, 'weight_cache'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                layer.bias_cache = np.zeros_like(layer.biases)

            # Update momentum with current gradients
            layer.weight_momentums = self.beta_1 * \
                layer.weight_momentums + \
                (1 - self.beta_1) * layer.dweights
            layer.bias_momentums = self.beta_1 * \
                layer.bias_momentums + \
                (1 - self.beta_1) * layer.dbiases

            # Get corrected momentum
            # self.iteration is 0 at first pass
            # and we need to start with 1 here
            weight_momentums_corrected = layer.weight_momentums / \
                (1 - self.beta_1 ** (self.iterations + 1))
            bias_momentums_corrected = layer.bias_momentums / \
                (1 - self.beta_1 ** (self.iterations + 1))

            # Update cache with squared current gradients
            layer.weight_cache = self.beta_2 * layer.weight_cache + \
                (1 - self.beta_2) * layer.dweights**2
            layer.bias_cache = self.beta_2 * layer.bias_cache + \
                (1 - self.beta_2) * layer.dbiases**2

            # Get corrected cache
            weight_cache_corrected = layer.weight_cache / \
                (1 - self.beta_2 ** (self.iterations + 1))
            bias_cache_corrected = layer.bias_cache / \
                (1 - self.beta_2 ** (self.iterations + 1))

            # Vanilla SGD parameter update + normalization
            # with square rooted cache
            layer.weights -= self.current_learning_rate * \
                weight_momentums_corrected / \
                (np.sqrt(weight_cache_corrected) +
                 self.epsilon)
            layer.biases -= self.current_learning_rate * \
                bias_momentums_corrected / \
                (np.sqrt(bias_cache_corrected) +
                 self.epsilon)

    # Update convolutional layer parameters

    def post_update_params(self):
        self.iterations += 1

# Common loss class


class Loss:

    # Regularization loss calculation
    def regularization_loss(self):

        # 0 by default
        regularization_loss = 0

        # If there aren't any trainable convolutional layers
        # in the network, run normal L1 and L2 regularization
        if not any(isinstance(layer, Basic_Convolution) for layer in self.trainable_layers):
            # Calculate regularization loss
            # iterate all trainable layers
            for layer in self.trainable_layers:

                # L1 regularization - weights
                # calculate only when factor greater than 0
                if layer.weight_regularizer_l1 > 0:
                    regularization_loss += layer.weight_regularizer_l1 * \
                        np.sum(np.abs(layer.weights))

                # L2 regularization - weights
                if layer.weight_regularizer_l2 > 0:
                    regularization_loss += layer.weight_regularizer_l2 * \
                        np.sum(layer.weights *
                               layer.weights)

                # L1 regularization - biases
                # calculate only when factor greater than 0
                if layer.bias_regularizer_l1 > 0:
                    regularization_loss += layer.bias_regularizer_l1 * \
                        np.sum(np.abs(layer.biases))

                # L2 regularization - biases
                if layer.bias_regularizer_l2 > 0:
                    regularization_loss += layer.bias_regularizer_l2 * \
                        np.sum(layer.biases *
                               layer.biases)

            return regularization_loss

        # If there are trainable convolutional layers
        # in the network, run L1 and L2 regularization
        # by first checking the layer type
        else:

            # Calculate regularization loss
            # iterate all trainable layers
            for layer in self.trainable_layers:

                if not hasattr(layer, 'Filters') and layer.weight_regularizer_l1 > 0:
                    # L1 regularization - weights
                    # calculate only when factor greater than 0
                    regularization_loss += layer.weight_regularizer_l1 * \
                        np.sum(np.abs(layer.weights))

                elif layer.weight_regularizer_l1 > 0:
                    # L1 regularization - weights
                    regularization_loss += layer.weight_regularizer_l1 * \
                        np.sum([np.sum(np.abs(K))
                               for K in layer.Filters])

                if not hasattr(layer, 'Filters') and layer.weight_regularizer_l2 > 0:
                    # L2 regularization - weights
                    regularization_loss += layer.weight_regularizer_l2 * \
                        np.sum(layer.weights *
                               layer.weights)

                elif layer.weight_regularizer_l2 > 0:
                    # L2 regularization - weights
                    regularization_loss += layer.weight_regularizer_l2 * \
                        np.sum([np.sum(K.ravel() *
                               K.ravel()) for K in layer.Filters])

                if not hasattr(layer, 'Filters') and layer.bias_regularizer_l1 > 0:
                    # L1 regularization - biases
                    # calculate only when factor greater than 0
                    regularization_loss += layer.bias_regularizer_l1 * \
                        np.sum(np.abs(layer.biases))

                elif layer.bias_regularizer_l1 > 0:
                    # L1 regularization - biases
                    # calculate only when factor greater than 0
                    regularization_loss += layer.bias_regularizer_l1 * \
                        np.sum(np.abs(layer.Biases))

                if not hasattr(layer, 'Filters') and layer.bias_regularizer_l2 > 0:
                    # L2 regularization - biases
                    regularization_loss += layer.bias_regularizer_l2 * \
                        np.sum(layer.biases *
                               layer.biases)

                elif layer.bias_regularizer_l2 > 0:
                    # L2 regularization - biases
                    regularization_loss += layer.bias_regularizer_l2 * \
                        np.sum(layer.Biases *
                               layer.Biases)

        return regularization_loss

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y, *, include_regularization=False):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Calculates accumulated loss
    def calculate_accumulated(self, *, include_regularization=False):

        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

# Cross-entropy loss


class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, y_pred, y_true):

        # Number of samples
        samples = len(y_pred)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(y_pred[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / y_pred
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step


class Activation_Softmax_Loss_CategoricalCrossentropy:

    # Backward pass
    def backward(self, y_pred, y_true):
        # Number of samples
        samples = len(y_pred)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = y_pred.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        # print(self.dinputs.shape)

# Binary cross-entropy loss


class Loss_BinaryCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, y_pred, y_true):

        # Number of samples
        samples = len(y_pred)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(y_pred[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_y_pred -
                         (1 - y_true) / (1 - clipped_y_pred)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Mean Squared Error loss


class Loss_MeanSquaredError(Loss):  # L2 loss

    # Forward pass
    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, y_pred, y_true):

        # Number of samples
        samples = len(y_pred)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(y_pred[0])

        # Gradient on values
        self.dinputs = -2 * (y_true - y_pred) / outputs
        # print(self.dinputs.shape)
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Mean Absolute Error loss


class Loss_MeanAbsoluteError(Loss):  # L1 loss

    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        # Return losses
        return sample_losses

    # Backward pass

    def backward(self, y_pred, y_true):

        # Number of samples
        samples = len(y_pred)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(y_pred[0])

        # Calculate gradient
        self.dinputs = -np.sign(y_true - y_pred) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Common accuracy class


class Accuracy:

    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):

        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        # Return accuracy
        return accuracy

    # Calculates accumulated accuracy
    def calculate_accumulated(self):

        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        # Return accuracy
        return accuracy

    # Reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

# Accuracy calculation for classification model


class Accuracy_Categorical(Accuracy):

    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary

    # No initialization is needed
    def init(self, y):
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

# Accuracy calculation for regression model


class Accuracy_Regression(Accuracy):

    def __init__(self):
        # Create precision property
        self.precision = None

    # Calculates precision value
    # based on passed-in ground truth values
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

# Model class


class Model:

    # Innitialization
    def __init__(self, initial_batch_size=32, timesteps=1):
        self.initial_batch_size = initial_batch_size
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None
        self.timesteps = timesteps

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None, learning_rate=None, learning_rate_decay=None):
        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy

        if learning_rate is not None:
            self.optimizer.learning_rate = learning_rate

        if learning_rate_decay is not None:
            self.optimizer.decay = learning_rate_decay

    # Finalize the model
    def finalize(self):

        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers:
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):

            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # The last layer - the next object is the loss
            # Also let's save aside the reference to the last object
            # whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr(self.layers[i], 'weights') or hasattr(self.layers[i], 'Filters'):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(
                self.trainable_layers
            )

        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and \
           isinstance(self.loss, Loss_CategoricalCrossentropy):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = \
                Activation_Softmax_Loss_CategoricalCrossentropy()

    # Train the model
    def train(self, X, y, *, epochs=1, batch_size=None,
              print_every=1, validation_data=None):
        self.batch_size = batch_size

        # Initialize accuracy object
        self.accuracy.init(y)

        # Default value if batch size is not being set
        train_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1

        # Main training loop
        for epoch in range(1, epochs+1):

            # Print epoch number
            print(f'\nepoch: {epoch}')

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):
                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    self.batch_X = X
                    batch_y = y

                # Otherwise slice a batch
                else:
                    self.batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                # TODO add convolutional support AKA 4D input
                # Perform the forward pass
                if self.timesteps > 1 and self.batch_X.ndim == 3:
                    for layer in self.trainable_layers:
                        if hasattr(layer, 'hiddenVector'):
                            layer.reset()
                    for t in range(self.timesteps):
                        # Get the row at the current timestep
                        # Add reshape here
                        row_at_timestep = self.batch_X[:, t,
                                                       :].reshape(-1, self.batch_X.shape[2])

                        # Pass the row to the forward function
                        output = self.forward(row_at_timestep, training=True)
                elif self.timesteps > 1 and self.batch_X.ndim == 2:
                    for t in range(self.timesteps):
                        # Get the row at the current timestep
                        row_at_timestep = self.batch_X[t, :]

                        # Perform the forward pass
                        output = self.forward(row_at_timestep, training=True)

                else:
                    # Perform the forward pass
                    output = self.forward(self.batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y,
                                        include_regularization=True)
                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(
                    output)
                accuracy = self.accuracy.calculate(predictions,
                                                   batch_y)
                # print('output shape:', output.shape)
                # print('true shape:', batch_y.shape)
                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Print a summary
                if not step % print_every or step == train_steps - 1:
                    print(f'    step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(
                    include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}\n')

            # If there is the validation data
            if validation_data is not None:

                # Evaluate the model:
                self.evaluate(*validation_data,
                              batch_size=batch_size)

    # Evaluates the model using passed-in dataset
    def evaluate(self, X_val, y_val, *, batch_size=None):

        # Default value if batch size is not being set
        validation_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Reset accumulated values in loss
        # and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()

        # Iterate over steps
        for step in range(validation_steps):
            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                self.batch_X = X
                batch_y = y

            # Otherwise slice a batch
            else:
                self.batch_X = X[step*batch_size:(step+1)*batch_size]
                batch_y = y[step*batch_size:(step+1)*batch_size]

            # TODO add convolutional support AKA 4D input

            # Perform the forward pass with timesteps
            if self.timesteps > 1 and self.batch_X.ndim == 3:
                for layer in self.trainable_layers:
                    if hasattr(layer, 'hiddenVector'):
                        layer.reset()
                for t in range(self.timesteps):
                    # Get the row at the current timestep
                    # Add reshape here
                    row_at_timestep = self.batch_X[:, t,
                                                   :].reshape(-1, self.batch_X.shape[2])

                    # Pass the row to the forward function
                    output = self.forward(row_at_timestep, training=True)

            elif self.timesteps > 1 and self.batch_X.ndim == 2:
                for t in range(self.timesteps):
                    # Get the row at the current timestep
                    row_at_timestep = self.batch_X[t, :]

                    # Perform the forward pass
                    output = self.forward(row_at_timestep, training=True)

            else:
                # Perform the forward pass
                output = self.forward(self.batch_X, training=True)

            self.loss.calculate(output, batch_y)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print a summary
        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    # Predicts on the samples
    def predict(self, X, *, batch_size=None):

        # Default value if batch size is not being set
        prediction_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size

            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        # Model outputs
        output = []

        # Iterate over steps
        for step in range(prediction_steps):

            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                self.batch_X = X

            # Otherwise slice a batch
            else:
                self.batch_X = X[step*batch_size:(step+1)*batch_size]

            # Perform the forward pass
            batch_output = self.forward(self.batch_X, training=False)

            # Append batch prediction to the list of predictions
            output.append(batch_output)

        # Stack and return results
        return np.vstack(output)

    # Performs forward pass
    def forward(self, X, training):

        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:

            layer.forward(layer.prev.output, training)

        # "layer" is now the last object from the list,
        # return its output
        return layer.output

    # Performs backward pass
    def backward(self, output, y):

        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        y = y.reshape(-1, 1)

        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    # Retrieves and returns parameters of trainable layers
    def get_parameters(self):

        # Create a list for parameters
        parameters = []

        # Iterable trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        # Return a list
        return parameters

    # Updates the model with new parameters
    def set_parameters(self, parameters):

        # Iterate over the parameters and layers
        # and update each layers with each set of the parameters
        for parameter_set, layer in zip(parameters,
                                        self.trainable_layers):
            layer.set_parameters(*parameter_set)

    # Saves the parameters to a file
    def save_parameters(self, path):

        # Open a file in the binary-write mode
        # and save parameters into it
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    # Loads the weights and updates a model instance with them
    def load_parameters(self, path):

        # Open file in the binary-read mode,
        # load weights and update trainable layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    # Saves the model
    def save(self, path):

        # Make a deep copy of current model instance
        model = copy.deepcopy(self)

        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from the input layer
        # and gradients from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # For each layer remove inputs, output and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs',
                             'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Open a file in the binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    # Loads and returns a model

    @staticmethod
    def load(path):

        # Open file in the binary-read mode, load a model
        with open(path, 'rb') as f:
            model = pickle.load(f)

        # Return a model
        return model

# Beyesian Optimizer


class BayesianOptimizer(ABC):
    def __init__(self, model=False):
        kernel = Matern(length_scale_bounds=(1e-06, 1.0))
        self.model = GaussianProcessRegressor(kernel=kernel)
        if model:
            joblib.dump(self.model, 'my_model.joblib')

    def set(self, model, names=None, xi=0.01, kappa=2.576):
        self.evaluation_function = model.train
        self.Model = model
        self.xi = xi
        self.kappa = kappa
        self.names = names

    def predict(self, x):
        return self.model.predict(x)

    def check_bounds(self, bounds):
        for i, (lower, upper) in enumerate(bounds):
            if lower > upper:
                print(
                    f"Bounds error: Lower bound ({lower}) is greater than upper bound ({upper}) at index {i}")
                return False
        return True

    def set_deep_q_network_params(self, param_names, param_values):
        if len(self.names) != len(param_values):
            raise ValueError(
                "Length of param_names and param_values must be equal.")

        param_dict = dict(zip(param_names, param_values))

        if 'MaxMemoryLength' in param_dict:
            param_dict['MaxMemoryLength'] = int(
                round(param_dict['MaxMemoryLength']))

        self.Model.set(**param_dict)

    def generate_initial_data(self, n_samples, bounds, names, seed=None, save_every=None, X=True, y=True, TotalSamples=0, training_iterations=0):
        if TotalSamples == 0:
            TotalSamples == n_samples
        self.names = names
        if seed is not None:
            np.random.seed(seed)

        self.n_hyperparameters = len(bounds)
        if X:
            X_original = np.random.rand(n_samples, self.n_hyperparameters)
            X_modified = X_original.copy()
            self.X = X_modified
        if y:
            self.y = []
        if X:
            for i, (lower, upper) in enumerate(bounds):
                self.X[:, i] = self.X[:, i] * (upper - lower) + lower

        for i in range(n_samples):
            print(f"Sample: {i + 1}")
            self.set_deep_q_network_params(
                self.names, self.X[i + (TotalSamples - n_samples)])
            self.y.append(self.evaluation_function(
                Plot=False, Start=True, OF=True))

            if save_every is not None and (i+1) % save_every == 0:
                self.save_progress_to_file(
                    self.y, n_samples - (i+1), training_iterations, False)

        return self.X, self.y

    def maximize(self, n_iter, X, y, bounds, save_every=None, model=False):
        if not model:
            self.model.fit(X, y)
        for i in range(n_iter):
            print(f"Point: {i + 1}")
            self.next_point = self._acquisition_function(X, y, bounds)
            self.set_deep_q_network_params(self.names, self.next_point)
            self.y_new = self.evaluation_function(
                Plot=False, Start=True, OF=True)
            self.X = np.vstack([self.X, self.next_point])
            self.y = np.append(self.y, self.y_new)
            self.model.fit(self.X, self.y)

            if save_every is not None and (i+1) % save_every == 0:
                self.save_progress_to_file(self.y, 0, n_iter - (i+1), True)

        max_index = np.argmax(y)
        return X[max_index], y[max_index]

    @abstractmethod
    def _acquisition_function(self, X, y, bounds):
        pass

    def save_progress_to_file(self, y, remaining_init_data_iters, remaining_max_iters, model=False):
        np.savetxt('hyperparameters.txt', self.X)
        np.savetxt('scores.txt', y)
        if model:
            joblib.dump(self.model, 'my_model.joblib')

        remaining_iters = {
            'remaining_init_data_iters': remaining_init_data_iters,
            'remaining_max_iters': remaining_max_iters
        }
        with open('remaining_iters.json', 'w') as f:
            json.dump(remaining_iters, f)

    def save_parameters_to_file(self, params):
        with open('train_parameters.json', 'w') as f:
            json.dump(params, f)

    def load_parameters_from_file(self):
        with open('train_parameters.json', 'r') as f:
            return json.load(f)

    def pickup_training(self, model=True):
        train_params = self.load_parameters_from_file()
        self.X = np.loadtxt('hyperparameters.txt')
        self.y = np.loadtxt('scores.txt')
        if model:
            self.model = GaussianProcessRegressor(kernel=Matern())
            self.model.fit(self.X, self.y)

        with open('remaining_iters.json', 'r') as f:
            remaining_iters = json.load(f)

        remaining_init_data_iters = remaining_iters['remaining_init_data_iters']
        remaining_max_iters = remaining_iters['remaining_max_iters']

        initial_samples = train_params['initial_samples']
        training_iterations = train_params['training_iterations']
        selected_hyperparameters = train_params['selected_hyperparameters']
        RegisterTime = train_params['RegisterTime']
        ReturnTime = train_params['ReturnTime']
        ReturnHP = train_params['ReturnHP']
        save_every = train_params['save_every']

        all_hyperparameter_ranges = All_Hyperparameter_Ranges_Create()
        filtered_hyperparameter_ranges = {
            key: value for key, value in all_hyperparameter_ranges.items()
            if key in selected_hyperparameters
        }

        HP_Ranges = [value for key,
                     value in filtered_hyperparameter_ranges.items()]
        HP_Names = [key for key in filtered_hyperparameter_ranges.keys()]
        HP_Dict = filtered_hyperparameter_ranges

        HP_Range_Flipped = flip_bounds(HP_Ranges)

        self.names = HP_Names

        start = time.time()

        if remaining_init_data_iters is not None and remaining_init_data_iters > 0:
            self.X, self.y = self.generate_initial_data(remaining_init_data_iters, HP_Ranges, self.names, save_every=save_every, X=False,
                                                        y=False, TotalSamples=initial_samples, training_iterations=training_iterations)

        mid = time.time()

        if remaining_max_iters is not None and remaining_max_iters > 0:
            best_hyperparameters, best_objective = self.maximize(
                remaining_max_iters, self.X, self.y, HP_Range_Flipped, save_every=save_every, model=True)
        else:
            max_index = np.argmax(self.y)
            best_hyperparameters, best_objective = self.X[max_index], self.y[max_index]

        end = time.time()

        if ReturnHP == 3:
            if ReturnTime:
                return start, mid, end, HP_Names, HP_Ranges, HP_Dict, best_hyperparameters, best_objective
            return HP_Names, HP_Ranges, HP_Dict, best_hyperparameters, best_objective
        elif ReturnHP == 2:
            if ReturnTime:
                return start, mid, end, HP_Names, HP_Ranges, best_hyperparameters, best_objective
            return HP_Names, HP_Ranges, best_hyperparameters, best_objective
        elif ReturnHP == 1:
            if ReturnTime:
                return start, mid, end, HP_Names, best_hyperparameters, best_objective
            return HP_Names, best_hyperparameters, best_objective
        elif ReturnHP == 0:
            if ReturnTime:
                return start, mid, end, best_hyperparameters, best_objective
            return best_hyperparameters, best_objective

    def train(self, initial_samples, training_iterations, selected_hyperparameters, RegisterTime=False, ReturnTime=False, ReturnHP=1, save_every=None, TotalSamples=0):

        if not RegisterTime:
            ReturnTime = False

        all_hyperparameter_ranges = All_Hyperparameter_Ranges_Create()

        filtered_hyperparameter_ranges = {
            key: value for key, value in all_hyperparameter_ranges.items()
            if key in selected_hyperparameters
        }

        HP_Ranges = [value for key,
                     value in filtered_hyperparameter_ranges.items()]
        HP_Names = [key for key in filtered_hyperparameter_ranges.keys()]
        HP_Dict = filtered_hyperparameter_ranges

        self.names = HP_Names

        HP_Range_Flipped = flip_bounds(HP_Ranges)

        # Save the train method parameters
        train_params = {
            'initial_samples': initial_samples,
            'training_iterations': training_iterations,
            'selected_hyperparameters': selected_hyperparameters,
            'RegisterTime': RegisterTime,
            'ReturnTime': ReturnTime,
            'ReturnHP': ReturnHP,
            'save_every': save_every
        }
        self.save_parameters_to_file(train_params)

        # Generate initial data and maximize
        if RegisterTime:
            start = time.time()
        self.X, self.y = self.generate_initial_data(initial_samples, HP_Ranges, self.names, save_every=save_every, X=True,
                                                    y=True, TotalSamples=TotalSamples, training_iterations=training_iterations)
        if RegisterTime:
            mid = time.time()
        best_hyperparameters, best_objective = self.maximize(
            training_iterations, self.X, self.y, HP_Range_Flipped, save_every=save_every, model=False)
        if RegisterTime:
            end = time.time()

        if RegisterTime:
            print(
                f"Start to Mid: {mid - start}, Mid to End: {end - mid}, Full Program: {end - start}")

        if ReturnHP == 3:
            if ReturnTime:
                return start, mid, end, HP_Names, HP_Ranges, HP_Dict, best_hyperparameters, best_objective
            return HP_Names, HP_Ranges, HP_Dict, best_hyperparameters, best_objective
        elif ReturnHP == 2:
            if ReturnTime:
                return start, mid, end, HP_Names, HP_Ranges, best_hyperparameters, best_objective
            return HP_Names, HP_Ranges, best_hyperparameters, best_objective
        elif ReturnHP == 1:
            if ReturnTime:
                return start, mid, end, HP_Names, best_hyperparameters, best_objective
            return HP_Names, best_hyperparameters, best_objective
        elif ReturnHP == 0:
            if ReturnTime:
                return start, mid, end, best_hyperparameters, best_objective
            return best_hyperparameters, best_objective

# Probability improvement (PI) aquisition function


class ProbabilityImprovement(BayesianOptimizer):
    # Child class for Probability of Improvement (PI) acquisition function.
    def _acquisition_function(self, X, y, bounds):
        def neg_pi(x):
            mu, sigma = self.model.predict(x.reshape(1, -1))
            best_y = np.max(y)
            Z = (mu - best_y - self.xi) / sigma
            PI = norm.cdf(Z)
            return -PI

        res = minimize(neg_pi, X[np.argmax(y)], bounds=bounds)
        return res.x

# Expected improvement (PI) aquisition function


class ExpectedImprovement(BayesianOptimizer):
    # Child class for Expected Improvement (EI) acquisition function.
    def _acquisition_function(self, X, y, bounds):
        if not self.check_bounds(bounds):
            raise ValueError("Invalid bounds provided.")

        def neg_ei(x):
            mu, sigma = self.model.predict(x.reshape(1, -1), return_std=True)
            best_y = np.max(y)
            Z = (mu - best_y - self.xi) / sigma
            EI = (mu - best_y - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -EI

        res = minimize(neg_ei, X[np.argmax(y)], bounds=bounds)
        return res.x

# Upper confidence bound (UCB) aquisition function


class UpperConfidenceBound(BayesianOptimizer):
    # Child class for Upper Confidence Bound (UCB) acquisition function.
    def _acquisition_function(self, X, y, bounds):
        def neg_ucb(x):
            mu, sigma = self.model.predict(x.reshape(1, -1))
            UCB = mu + self.kappa * sigma
            return -UCB

        res = minimize(neg_ucb, X[np.argmax(y)], bounds=bounds)
        return res.x

# Deep Q-learning Agent


class DQNAgent:

    def set(self, state_size=None, action_size=None, episodes=None, batch_size=None, MaxMemoryLength=None,
            gamma=None, Agent_Learning_Rate=None, Agent_Learning_Rate_Min=None, Agent_Learning_Rate_Decay=None,
            epsilon=None, epsilon_min=None, epsilon_decay=None, model=None, learning_rate=None, learning_rate_decay=None):
        if state_size:
            self.state_size = state_size
        if action_size:
            self.action_size = action_size
        if MaxMemoryLength:
            self.memory = deque(maxlen=MaxMemoryLength)
        if gamma:
            self.gamma = gamma    # discount rate
        if epsilon:
            self.epsilon = epsilon  # exploration rate
        if epsilon_min:
            self.epsilon_min = epsilon_min
        if epsilon_decay:
            self.epsilon_decay = epsilon_decay
        if Agent_Learning_Rate:
            self.learning_rate = Agent_Learning_Rate
        if Agent_Learning_Rate_Min:
            self.learning_rate_min = Agent_Learning_Rate_Min
        if Agent_Learning_Rate_Decay:
            self.learning_rate_decay = Agent_Learning_Rate_Decay
        if episodes:
            self.episodes = episodes
        if batch_size:
            self.batch_size = batch_size
        if model:
            self.model = model

        if learning_rate and learning_rate_decay:

            model.set(
                optimizer=Optimizer_Adam(
                    learning_rate=learning_rate, decay=learning_rate_decay)
            )

        elif learning_rate:

            model.set(
                optimizer=Optimizer_Adam(learning_rate=learning_rate)
            )

        elif learning_rate_decay:

            model.set(
                optimizer=Optimizer_Adam(decay=learning_rate_decay)
            )

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size, episode):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.train(np.reshape(
                state, [1, state_size]), target_f, epochs=1, print_every=100)
        if self.epsilon > self.epsilon_min and self.epsilon_decay:
            self.epsilon = self.epsilon * \
                (1 / (1 + self.epsilon_decay * episode))
        if self.learning_rate > self.learning_rate_min and self.learning_rate_decay:
            self.learning_rate = self.learning_rate * \
                (1 / (1 + self.learning_rate_decay * episode))

    def start(self, batch_size):

        # Iterate the game
        for e in range(batch_size):

            # reset state in the beginning of each game
            state = env.reset()[0]
            state = np.reshape(state, state_size)

            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of 500
            # the more time_t the more score
            for time_t in range(500):
                # turn this on if you want to render
                # env.render()

                # Decide action
                action = agent.act(state)

                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, _, _ = env.step(action)
                next_state = np.reshape(next_state, state_size)

                # memorize the previous state, action, reward, and done
                agent.memorize(state, action, reward, next_state, done)

                # make next_state the new current state for the next frame.
                state = next_state

    def train(self, Plot=True, Start=True, OF=False, ListHP=False):

        if ListHP:
            print(
                f"state_size: {self.state_size:.0f}, action_size: {self.action_size:.0f}, episodes: {self.episodes:.0f},")
            print(
                f"batch_size: {self.batch_size:.0f}, MaxMemoryLength: 5000, gamma: {self.gamma:.2f}, Agent_Learning_Rate: {self.learning_rate:.4f},")
            print(
                f"Agent_Learning_Rate_Min: {self.learning_rate_min:.4f}, Agent_Learning_Rate_Decay: {self.learning_rate_decay:.4f},")
            print(
                f"epsilon: {self.epsilon:.4f}, epsilon_min: {self.epsilon_min:.4f}, epsilon_decay: {self.epsilon_decay:.4f},")
            print(
                f"learning_rate: {model.optimizer.learning_rate:.4f}, learning_rate_decay: {model.optimizer.decay:.4f}")

        if Plot or OF:
            PlotDictionary = {'ep': [], 'avg': [],
                              'max': [], 'min': [], 'eps': []}

        if Start:
            self.start(self.batch_size)

        overall_rewards = []

        # Iterate the game
        for e in range(self.episodes):

            if Plot:
                PlotDictionary['ep'].append(e + 1)

            ep_reward = 0

            # reset state in the beginning of each game
            state = env.reset()[0]
            state = np.reshape(state, state_size)

            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of 500
            # the more time_t the more score
            for time_t in range(500):

                if Plot == True:
                    AvgList = []

                # turn this on if you want to render
                # env.render()

                # Decide action
                action = agent.act(state)

                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, _, _ = env.step(action)

                ep_reward += reward

                next_state = np.reshape(next_state, state_size)

                # memorize the previous state, action, reward, and done
                agent.memorize(state, action, reward, next_state, done)

                # make next_state the new current state for the next frame.
                state = next_state

                # done becomes True when the game ends
                # ex) The agent drops the pole
                if done:
                    if (e + 1) % 200 == 0:
                        # print the score and break out of the loop
                        print(
                            f"	episode: {e + 1}/{episodes}, score: {time_t + 1}, epsilon: {round(self.epsilon, 2)}, self LR: {round(self.learning_rate, 4)}, NN LR: {model.optimizer.learning_rate}")
                    overall_rewards.append(ep_reward + 1)
                    env.reset()
                    break

            if Plot or OF:
                if not len(overall_rewards) > 0:
                    overall_rewards.append(1)
                PlotDictionary['avg'].append(statistics.mean(overall_rewards))
                PlotDictionary['min'].append(min(overall_rewards))
                PlotDictionary['max'].append(max(overall_rewards))
                PlotDictionary['eps'].append(self.epsilon * 250)

            # train the agent with the experience of the episode
            agent.replay(self.batch_size, e)
        if Plot:
            plt.plot(PlotDictionary['ep'], PlotDictionary['avg'], label="avg")
            plt.plot(PlotDictionary['ep'], PlotDictionary['min'], label="min")
            plt.plot(PlotDictionary['ep'], PlotDictionary['max'], label="max")
            plt.plot(PlotDictionary['ep'],
                     PlotDictionary['eps'], label="epsilon")
            plt.legend(loc=4)
            plt.show()

        if OF:
            return PlotDictionary['avg'][-1]

    def predict(state):
        # ! placeholder for self.act(state) since vs code doesn't like it
        return agent.act(state)

# Loads a MNIST dataset


def load_mnist_dataset(dataset, path):

    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(
                os.path.join(path, dataset, label, file),
                cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

# MNIST dataset (train + test)


def create_data_mnist(path):

    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test

# Create filters


def Create_Filters(Shapes, Low=0, High=1, Biases=False, BiasLow=-1, BiasHigh=1):

    RandomFilters = []

    if Biases == True:

        RandomBiases = np.random.uniform(BiasLow, BiasHigh, [len(Shapes)])

    for shape in Shapes:

        RandomFilters.append(np.random.uniform(Low, High, shape))

    if Biases == False:

        return RandomFilters

    else:

        return RandomFilters, RandomBiases

# define public hyperparamater ranges


def All_Hyperparameter_Ranges_Create():
    return {
        'Agent_Learning_Rate': (0.1, 0.0001),
        'Agent_Learning_Rate_Min': (0.001, 0.00001),
        'Agent_Learning_Rate_Decay': (0.01, 0.00001),
        'gamma': (0.99, 0.5),
        'epsilon': (1.0, 0.1),
        'epsilon_min': (0.1, 0.01),
        'epsilon_decay': (0.01, 0.0001),
        'learning_rate': (0.1, 0.0001),
        'learning_rate_decay': (0.01, 0.0001),
        'batch_size': (2048, 32),
        'MaxMemoryLength': (100000, 1000),
        'Episodes': (10000, 10),
        'Epochs': (100, 1),
    }

# Create, save, and load hyperparameter ranges


def Handle_Hyperparameter_Ranges(path='filtered_hyperparameter_ranges.json', action=None, dictionary=False, **kwargs):
    all_hyperparameter_ranges = {
        'Agent_Learning_Rate': (0.1, 0.0001),
        'Agent_Learning_Rate_Min': (0.001, 0.00001),
        'Agent_Learning_Rate_Decay': (0.01, 0.00001),
        'gamma': (0.99, 0.5),
        'epsilon': (1.0, 0.1),
        'epsilon_min': (0.1, 0.01),
        'epsilon_decay': (0.01, 0.0001),
        'learning_rate': (0.1, 0.0001),
        'learning_rate_decay': (0.01, 0.0001),
        'batch_size': (2048, 32),
        'MaxMemoryLength': (100000, 1000),
        'Episodes': (10000, 10),
        'Epochs': (100, 1),
    }

    selected_hyperparameters = kwargs.get('selected_hyperparameters', [])

    # Filter the dictionary based on the provided list of strings in kwargs
    filtered_hyperparameter_ranges = {
        key: value for key, value in all_hyperparameter_ranges.items()
        if key in selected_hyperparameters
    }

    if action == 'load':
        # Load and return the saved hyperparameters
        with open(path, "r") as infile:
            loaded_data = json.load(infile)
        loaded_hyperparameters_tuples = [
            tuple(t) for t in loaded_data['tuples']]
        loaded_hyperparameters_names = loaded_data['names']
        if dictionary:
            loaded_hyperparameters_dict = {name: tuple(t) for name, t in zip(
                loaded_hyperparameters_names, loaded_hyperparameters_tuples)}
            return loaded_hyperparameters_tuples, loaded_hyperparameters_names, loaded_hyperparameters_dict
        return loaded_hyperparameters_tuples, loaded_hyperparameters_names

    # If action is None or 'save', create the lists of filtered hyperparameter tuples and names
    filtered_hyperparameters_tuples = [
        tuple(t) for t in filtered_hyperparameter_ranges.values()]
    filtered_hyperparameters_names = list(
        filtered_hyperparameter_ranges.keys())

    if action == 'save':
        # Save the filtered hyperparameter tuples and names as a JSON file at the given 'path'
        data_to_save = {
            'tuples': [list(t) for t in filtered_hyperparameters_tuples],
            'names': filtered_hyperparameters_names
        }
        if dictionary:
            data_to_save['dictionary'] = {name: list(t) for name, t in zip(
                filtered_hyperparameters_names, filtered_hyperparameters_tuples)}
        with open(path, "w") as outfile:
            json.dump(data_to_save, outfile)

    # If action is None, return the filtered hyperparameter tuples, names, and optionally, the dictionary
    if dictionary:
        filtered_hyperparameters_dict = {name: tuple(t) for name, t in zip(
            filtered_hyperparameters_names, filtered_hyperparameters_tuples)}
        return filtered_hyperparameters_tuples, filtered_hyperparameters_names, filtered_hyperparameters_dict
    return filtered_hyperparameters_tuples, filtered_hyperparameters_names

# Order hyperparameters


def Order_Hyperparameters(path=None, **kwargs):
    all_hyperparameter_ranges = {
        'Agent_Learning_Rate': (0.1, 0.0001),
        'Agent_Learning_Rate_Min': (0.001, 0.00001),
        'Agent_Learning_Rate_Decay': (0.01, 0.00001),
        'gamma': (0.99, 0.5),
        'epsilon': (1.0, 0.1),
        'epsilon_min': (0.1, 0.01),
        'epsilon_decay': (0.01, 0.0001),
        'learning_rate': (0.1, 0.0001),
        'learning_rate_decay': (0.01, 0.0001),
        'batch_size': (2048, 32),
        'MaxMemoryLength': (100000, 1000),
        'Episodes': (10000, 10),
        'Epochs': (100, 1),
    }

    if path:
        with open(path, "r") as infile:
            data = json.load(infile)
    else:
        data = kwargs

    ordered_tuples = []
    ordered_names = []
    ordered_dicts = {}

    for key in all_hyperparameter_ranges:
        if key in data.get('names', []):
            index = data['names'].index(key)
            ordered_tuples.append(tuple(data['tuples'][index]))
            ordered_names.append(key)
        if key in data.get('dictionary', {}):
            ordered_dicts[key] = data['dictionary'][key]

    output = [ordered_tuples, ordered_names, ordered_dicts]
    return tuple(output)

# Create Kwargs


def Create_Kwargs(variable_list):
    kwargs = {}
    key_names = ['tuples', 'names', 'dictionary']
    for i, var in enumerate(variable_list):
        key = key_names[i] if i < len(key_names) else f'arg{i}'
        kwargs[key] = var
    return kwargs

# Flip bounds


def flip_bounds(HP_Range):
    flipped_HP_Range = [(upper, lower) for lower, upper in HP_Range]
    return flipped_HP_Range

# Create stock data


def create_stock_data(Num_of_Days, Percent_for_Tests, etf_folder_path_input="stock_dataset/ETFs/"):
    data_matrices = []

    etf_folder_path = etf_folder_path_input

    if os.path.exists(etf_folder_path):
        # Iterate through each file in the folder
        for file in os.listdir(etf_folder_path):
            file_path = os.path.join(etf_folder_path, file)

            # Read the CSV file using pandas
            df = pd.read_csv(file_path)

            # Extract the last Num_of_Days days of data and save it as a 2D matrix
            last_n_days = df[-Num_of_Days:].copy()
            data_matrix = last_n_days[["Open", "High", "Low", "Close"]].values

            # Pad the data_matrix with zeros if the number of rows is less than Num_of_Days
            if data_matrix.shape[0] < Num_of_Days:
                data_matrix = np.pad(data_matrix, ((
                    Num_of_Days - data_matrix.shape[0], 0), (0, 0)), mode='constant', constant_values=0)

            # Append data_matrix to data_matrices
            data_matrices.append(data_matrix)
    else:
        # Access the folder labeled "ETF" from the file "stock_dataset.zip"
        with zipfile.ZipFile("stock_dataset.zip", "r") as zip_ref:
            etf_folder = [info for info in zip_ref.infolist(
            ) if info.filename.startswith("ETF/")]

            # Iterate through each file in the folder
            for file in etf_folder:
                with zip_ref.open(file, "r") as csvfile:
                    # Read the CSV file using pandas
                    df = pd.read_csv(io.TextIOWrapper(csvfile))

                    # Extract the last Num_of_Days days of data and save it as a 2D matrix
                    last_n_days = df[-Num_of_Days:].copy()
                    data_matrix = last_n_days[[
                        "Open", "High", "Low", "Close"]].values

                    # Pad the data_matrix with zeros if the number of rows is less than Num_of_Days
                    if data_matrix.shape[0] < Num_of_Days:
                        data_matrix = np.pad(data_matrix, ((
                            Num_of_Days - data_matrix.shape[0], 0), (0, 0)), mode='constant', constant_values=0)

                    # Append data_matrix to data_matrices
                    data_matrices.append(data_matrix)

    # Convert the list of 2D matrices into a 3D array
    data_3d_array = np.array(data_matrices)

    # Shuffle the 3D array along its first axis
    np.random.shuffle(data_3d_array)

    # Normalize the values in the 3D array to a range of -1 to 1
    half_max = np.max(data_3d_array) / 2
    normalized_3d_array = (data_3d_array - half_max) / half_max

    # Split the matrices into training and testing data
    X_all, X_test_all = train_test_split(
        normalized_3d_array, test_size=Percent_for_Tests, random_state=42)

    # Break off the "Close" column from each matrix and append them to separate variables "y" and "y_test"
    y = X_all[:, :, -1]
    y_test = X_test_all[:, :, -1]

    # Remove the "Close" column from each matrix in X_all and X_test_all
    X = np.delete(X_all, -1, axis=2)
    X_test = np.delete(X_test_all, -1, axis=2)

    return X, X_test, y, y_test


# Initialize Unused variables
if True:
    env = 0
    state_size = 0
    action_size = 0
    agent = 0
    episodes = 0

X, X_test, y, y_test = create_stock_data(
    Num_of_Days=10, Percent_for_Tests=0.1)

y = y[:, -1]
y_test = y_test[:, -1]

# Instantiate the model
model = Model(timesteps=10)

# Add layers
model.add(Layer_Dense(3, 96))
model.add(Activation_ReLU())
model.add(Layer_Recurrent(96, 96))
model.add(Activation_ReLU())
model.add(Layer_Dense(96, 1))
model.add(Activation_Linear())

# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_MeanSquaredError(),
    optimizer=Optimizer_Adam(learning_rate=0.01, decay=1e-3),
    accuracy=Accuracy_Regression()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=100, batch_size=None, print_every=100)
