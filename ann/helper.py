# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:16:42 2020

@author: Aneesah Abdul Kadhar, Dilani Maheswaran
"""

import pandas as pd

class Helper:

    """
    Fetch the number of layers and neurons in each layer from user input. 
    """
    def get_layers():
        #fetch number of hidden layers from console
        print("Incorrect inputs will be ignored and default values will be assigned")
        print("Enter the number of hidden layers : ")
        
        layers_input = input()  
        layers = int(layers_input) if layers_input.isnumeric() else 4
        
        #fetch number of neurons per layer from console    
        num_hiddenlayer = []
        for i in range(layers):
            print("Enter the number of neurons in Layer {0} : ". format(i + 1))
            neurons_input = input()
        
            if neurons_input.isnumeric():
               num_hiddenlayer.append(int(neurons_input))
            else:
                num_hiddenlayer = [5,4,3]
                break;
        return num_hiddenlayer
    
    """
    Fetch the activation function from user input. 
    """
    def get_activation_function():
        
        #set default function as sigmoid
        activation_function_index = 1
        #fetch activation function
        activation_function_list = ["Null", "Sigmoid", "Hyperbolic Tangent", "Cosine", "Gaussian"]
        
        print("Input the number to select activation function :")
        print("1. Null")
        print("2. Sigmoid")
        print("3. Hyperbolic Tangent")
        print("4. Cosine")
        print("5. Gaussian")
        
        activation_function_input = input()
        if activation_function_input.isnumeric():
            index = int(activation_function_input) - 1
            activation_function_index = index if 0 <= index <= 4 else 1
            
        activation_function =  activation_function_list[activation_function_index]
            
        return activation_function

    """
    Read dataset
    train_percentage : percentage of training dataset. Float ranging from 0.00 - 1.00
    test_percentage : percentage of testing dataset.  Float ranging from 0.00 - 1.00
    """
    def read_file(filename, train_percentage = 0.70):
        dataset = pd.read_csv(filename, header=None, dtype=float, delim_whitespace=True) 
        train = dataset.sample(frac=train_percentage).fillna(0.00) # get training portion 
        test = dataset.drop(train.index).fillna(0.00) # remainder testing portion 
         
        return train.values.tolist(), test.values.tolist() 
            
            
        