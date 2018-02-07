import numpy as np

def sigmoid(z):
    return_value = 1.0/(1+np.exp(-z))
    return return_value

def sigmoid_derivative(z):
    return_value = sigmoid(z)*(1-sigmoid(z))
    return return_value