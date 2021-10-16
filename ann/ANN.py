# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 18:45:28 2020

@author: Aneesah Abdul Kadhar, Dilani Maheswaran
"""

import numpy as np

class ANN:
    INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    HIDDEN_SIZE = []
    train = []
    test = []
    activation_function = "sigmoid"
    
    def set_data_set(train, test):
        ANN.train = train
        ANN.test = test

    def set_hyper_parameters(hidden_size):
        ANN.HIDDEN_SIZE = hidden_size
        
    def set_activation_function(activation_function):
        ANN.activation_function = activation_function
        
    """
    Initialize ANN architecture
    """
    def initialize_network(p):
        n, h, o = ANN.INPUT_SIZE, ANN.HIDDEN_SIZE, ANN.OUTPUT_SIZE
        part = iter(p)
        neural_network = []
        
        inner_layer = n + 1
        
        for h in ANN.HIDDEN_SIZE:
            # connections between input layer and outer layer
            neural_network.append([[next(part) for i in range(inner_layer)] for j in range(h)])
            inner_layer = h + 1

        neural_network.append([[next(part) for i in range(inner_layer)] for j in range(o)])
    
        return neural_network
    
    """
    Forward propogate the ANN using the activation function selected by the user
    """
    def feed_forward(network, example):
        layer_input, layer_output= example[:-1], []

        for layer in network:
            for neuron in layer:
                summ = ANN.sum_dot_product(neuron, layer_input)

                #approximation using user defined activation function
                if ANN.activation_function == "sigmoid":
                   layer_output.append(ANN.sigmoid(summ))
                elif ANN.activation_function == "null":
                    layer_output.append(ANN.null(summ))
                elif ANN.activation_function == "Hyperbolic Tangent":
                     layer_output.append(ANN.hyperbolic_tangent(summ))
                elif ANN.activation_function == "Cosine":
                     layer_output.append(ANN.cosine(summ)) 
                else: 
                     layer_output.append(ANN.gaussian(summ))
                    
            layer_input, layer_output = layer_output, []
        
        return layer_input

    """
    Find Mean Squared Error of the ANN for training dataset
    """
    def mean_square_error(network):
        training = ANN.train
        summ = 0.00
        training_size = len(training)
        
        #For each training data in the dataset, find the mean squared error
        for example in training:
            layer_output= example[-1]
          
            target = []
            target.append(layer_output);
            
            actual = ANN.feed_forward(network, example)
            summ += ANN.sum_square_error(actual, target)
        return summ / training_size

    """
    Find the sum of squared error
    """
    def sum_square_error(actual, target):
        summ = 0.00
        for i in range(len(actual)):
            summ += (actual[i] - target[i])**2
        return summ

    """
    Find the sum of dot products of weights and inputs
    """
    def sum_dot_product(weights, inputs):
        bias = weights[-1]
        summ = 0.00
        for i in range(len(weights)-1):
           summ += (weights[i] * float(inputs[i]))
        return summ + bias
    
    """sigmoid activation function"""
    def sigmoid(x):
        return 1/ (1 + np.exp(-x))
       
    """hyperbolic tangent activation function"""
    def hyperbolic_tangent(x):
        #return(np.sinh(x)/np.cosh(x))
        return np.tanh(x)
    
    """cosine activation function"""
    def cosine(x):
        return(np.cos(x))
       
    """gaussian activation function"""
    def gaussian(x):
        return np.exp(-np.power(x, 2.) / 2)
    
    """null activation function"""
    def null(x):
        return 0