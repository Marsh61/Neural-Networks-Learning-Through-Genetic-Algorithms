import numpy as np
import random
import copy
import math
from activation_fuctions import sigmoid_derivative, sigmoid
from datetime import datetime


"""
Basic testing function that runs n images through the network then reports the total accuracy of the network
"""
def test_nn(network, n):
    starttime = str(datetime.now())
    test = load_all_data()
    test = list(test)
    training_counter = 0
    correct_sum_training = 0
    for i in range(len(test)):
        #output will be a 1 if the network guessed right or a 0 if the network guessed wrong
        output = network.train(np.matrix(test[training_counter][0]), test[training_counter][1])
        correct_sum_training += output
        training_counter += 1
        print(training_counter)
        if training_counter == n:
            break

    endtime = str(datetime.now())
    print("TOTAL ACCURACY: " + str(correct_sum_training/training_counter))
    print("Learning rate: " + str(network.learning_rate) + " Decay: " + str(network.decay) + " layers_neurons: " + str(network.layers_neurons))
    print("starttime: " + starttime + "\n endtime: " + endtime)
    return network 


class NeuralNetwork(object):

    def __init__(self, layers_neurons, learning_rate, decay, preset_weights = None, preset_biases = None):
        self.layers_neurons = layers_neurons
        self.learning_rate = learning_rate
        if preset_weights is None:
            self.weights = list()
            self.weights.append(np.array([0]))
            for i in range(1,len(self.layers_neurons)):
                #plus one to account for the bias weight
                rand_values = np.random.uniform(-0.5,0.5,(layers_neurons[i], layers_neurons[i-1]))
                self.weights.append(rand_values)
                
        else:
            self.weights = preset_weights

        if preset_biases is None:
            self.biases = list([np.array([0])])
            for i in range(1,len(self.layers_neurons)):
                #plus one to account for the bias weight
                rand_values = np.random.uniform(-1,1,(layers_neurons[i], 1))
                self.biases.append(rand_values)
            
        else: 
            self.biases = preset_biases

        self.label_matrix = list(np.array([0] * self.layers_neurons[len(self.layers_neurons)-1]))
        self.stored_outputs = list()
        self.stored_correct = list()
        self.weights2 = copy.deepcopy(self.weights[:])
        self.biases2 = copy.deepcopy(self.biases[:])
        self.decay = decay

    def parse_target_value(self,label):
        self.label_matrix[label] = 1


    def single_example_network_output(self, train_input):
        self.stored_outputs = list()
        self.stored_outputs.append(train_input)
        for i in range(1,len(self.layers_neurons)):
            weight = self.weights[i]
            bias = self.biases[i]
            train_input = sigmoid(np.dot(weight,train_input) + bias)
            self.stored_outputs.append(train_input)
        return train_input

    """
    New update weights function that is over 100x faster than the old function
    """
    def update_weights2(self):
        for i in range(1,len(self.weights)):
            weight_matrix = np.asmatrix(self.weights[i])
            tmpCalc = self.learning_rate * np.asmatrix(self.stored_outputs[i-1]) * np.asmatrix(self.stored_correct[i]).T
            decayCalc = self.learning_rate * self.decay * weight_matrix
            self.weights[i] = np.asarray(weight_matrix + tmpCalc.T - decayCalc)
        return self.weights

    """
    New update biases function that is significantly faster than the old function 
    """
    def update_biases2(self):
        for i in range(1,len(self.biases)-1):
            bias_matrix = np.asmatrix(self.biases[i])
            decayCalc = self.learning_rate * self.decay * bias_matrix
            self.biases[i] = np.asarray(bias_matrix + self.learning_rate * 1 * np.asmatrix(self.stored_correct[i]) - decayCalc)
        return self.biases

    def determine_if_prediction_was_correct(self,label):
        if np.argmax(self.stored_outputs[len(self.layers_neurons)-1]) == label:
            return 1
        else: 
            return 0

    def calculate_correctness(self,label):
        stored_counter = 0
        self.parse_target_value(label)
        for i in range(len(self.layers_neurons)-1,0,-1):
            if i == len(self.layers_neurons)-1:
                label_matrix_convert = np.asmatrix(self.label_matrix).T
                subValue = np.subtract(label_matrix_convert, self.stored_outputs[len(self.layers_neurons)-1])
                multiplyValue = np.multiply(subValue, self.stored_outputs[len(self.layers_neurons)-1])
                sub1Value = np.subtract(1,self.stored_outputs[len(self.layers_neurons)-1])
                finalValue = np.multiply(multiplyValue,sub1Value)
                self.stored_correct.append(finalValue)
            else:
                tempCorrectness = list()
                for ii in range(self.layers_neurons[i]):
                    output = self.stored_outputs[i][ii]
                    mutiplyOuts = output * (1-output)
                    sumValue = 0 
                    for iii in range(self.layers_neurons[i+1]):
                        sumValue += self.weights[i+1][iii][ii] * self.stored_correct[stored_counter][iii]
                    correctValue = mutiplyOuts * sumValue
                    tempCorrectness.append(np.asarray(correctValue)[0])
                self.stored_correct.append(np.asmatrix(tempCorrectness))
                stored_counter +=1
        self.stored_correct = list(np.asarray(self.stored_correct))
        self.stored_correct.append(np.matrix([0] * self.layers_neurons[0]))
        self.stored_correct = list(reversed(self.stored_correct))
        return self.stored_correct

    def train (self,train_input,train_label):
        self.label_matrix = list(np.array([0] * self.layers_neurons[len(self.layers_neurons)-1]))
        self.stored_outputs = list()
        self.stored_correct = list()
        self.single_example_network_output(train_input)
        self.calculate_correctness(train_label)
        self.update_weights2()
        self.update_biases2()

        return self.determine_if_prediction_was_correct(train_label)

    def test_no_label(self,test_input):        
        self.stored_outputs = list()
        self.stored_correct = list()
        self.single_example_network_output(test_input)
        return np.argmax(self.stored_outputs[len(self.layers_neurons)-1])

    def get_weights_as_list(self):
        flat_list = list()

        for weight in self.weights:
            for item in weight.flatten():
                flat_list.append(item)
        
        return flat_list

    def get_biases_as_list(self):
        flat_list = list()

        for bias in self.biases:
            for item in bias.flatten():
                flat_list.append(item)
        
        return flat_list

    def list_to_weights(self,input_list):
        new_list = list()

        new_list.append(np.array(input_list.pop(0)))

        for layers in range(len(self.layers_neurons)):
            if (layers + 1 > len(self.layers_neurons)-1):
                break
            tmp = list()
            for i in range((self.layers_neurons[layers+1])):
                tmp2 = list()
                for ii in range(self.layers_neurons[layers]):
                    tmp2.append([input_list.pop(0)])
                
                tmp.append(tmp2)
            tmp = np.array(tmp)
            new_list.append(tmp.reshape(self.layers_neurons[layers+1],self.layers_neurons[layers]))

        self.weights = new_list

    def list_to_biases(self,input_list):
        new_list = list()

        new_list.append(np.array(input_list.pop(0)))



        for layers in range(len(self.layers_neurons)):

            if (layers + 1 > len(self.layers_neurons)-1):
                break
            tmp = list()
            for i in range((self.layers_neurons[layers+1])):
                tmp2 = list()
                tmp2.append(input_list.pop(0))
                
                tmp.append(tmp2)
            tmp = np.array(tmp)
            new_list.append(tmp)
            

        self.biases = new_list

if __name__ == "__main__":
    network = NeuralNetwork([1,4,2],0,0)
    print (network.biases)
    bias = network.get_biases_as_list()
    #print (bias)
    network.list_to_biases(bias)

    print("\n\n")
    print(network.biases)


    
    

    


    



