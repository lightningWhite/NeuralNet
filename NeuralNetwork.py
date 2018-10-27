# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 11:53:14 2018

@author: Daniel
"""

import numpy as np
from numpy.random import seed
from numpy.random import randn
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class NNetNode:
    def __init__(self):
        self.inputs = []
        self.weights = []
        self.output = -1
        self.thresh = 0
        self.doRandWeights = True

    def _set_rnd_weights(self):
        # Generate a random number, positive or negative, for each weight
        self.weights = randn(len(self.inputs))
        self.doRandWeights = False
    
    def set_inputs(self, passed_inputs):

        self.inputs = passed_inputs
        np.append(self.inputs,-1) # Add a bias node
      
        if self.doRandWeights:
            self._set_rnd_weights() # Set the weights for each input to random values

    def _activation_fun(self, raw_output):
        # A simple step function
        if raw_output > self.thresh:
            return 1
        else:
            return 0
    
    def get_output(self):
        # Calculate the summation of inputs * weights
        raw_output = 0
        for i in range(len(self.inputs)):
#            print(type(self.inputs[i]))
#            print(self.inputs)
#            print(type(self.weights[i]))
#            print(self.weights)
            raw_output += self.inputs[i] * self.weights[i]
        self.output = self._activation_fun(raw_output)
        return self.output
   
class NNetClassifier:
    def __init__(self):
        self.net = [] # A list of lists of nodes. Each list of nodes represents a layer
        self.hidden_layers = [] # A list of lists of nodes. Each list of nodes represents a layer.
        self.outputs = []
        self.num_targets = 0
        
    def set_num_targets(self, num):
        self.num_targets = num
    
    # Take a list of layer sizes. Each size in the list will create a layer consisting of 'size' nodes
    def configure_hidden_layers(self, layer_sizes): 
        # Add each layer of nodes to the hidden_layers list
        for size in layer_sizes:
            nodes = []
            for i in size:
                nodes.append(NNetNode())
            self.hidden_layers.append(nodes)
            
    
    def _create_net(self):
        first_layer = []
        # Create the first layer consisting of as many nodes as attributes
        for attribute in range(self.num_targets):#data_targets[0]:
            node = NNetNode()
            first_layer.append(node)
            self.net.append(first_layer)
                    
        # Append all the hidden layers onto the net
        for layer in self.hidden_layers:
            print("Hidden layer")
            self.net.append(layer)
    
    def _propogate_through_hidden_layers(self, layer_num):
        # For some readon the length of net is 3... Figure this out
        # The last layer has been reached
        if layer_num == len(self.net):
            return
        
        prev_layer_outputs = []
        
        # Get the outputs of the previous layer
        for node in self.net[layer_num - 1]:
            prev_layer_outputs.append(node.get_output)
        
        # Use the outputs of the previous layer as the inputs to this layer
        for node in self.net[layer_num]:
            node.set_inputs(prev_layer_outputs)
        
        # Recursively propogate to the next layer
        print("recurring")
        self._propogate_through_hidden_layers(layer_num + 1)
            
    # Train the NNet
    def fit(self, data_train, data_targets):
        self._create_net()
        
        first_layer_num = 0
        
        # Send each row through the NNet
        for row in data_train:
            # Set the inputs of each node in the first layer
            for node in self.net[first_layer_num]:
                # Setting the attributes in the nodes trains the net
                node.set_inputs(row) 
                
#                self._propogate_through_hidden_layers(first_layer_num + 1)
#                _back_propogate() ??
                
    def _get_outputs(self):
        outputs = []
        # Get the outputs from the last layer of nodes in the net
        for node in self.net[len(self.net)-1]:
            outputs.append(node.get_output())
        return outputs
    
    # Run the list of attributes through the network and return the outputs        
    def predict(self, data_test):
        
        predictions = []
        
        for attributes in data_test:
            
            for node in self.net[0]:
                node.set_inputs(attributes)
                
#            self._propogate_through_hidden_layers(1)
            predictions.append(self._get_outputs())    
        
        return predictions

def one_hot_encode_targets(targets):
    one_hot_cols = ["0", "1", "2"]
    one_hot_targets = pd.get_dummies(targets, columns=one_hot_cols)
    return one_hot_targets

def main(): 
    # Load the iris dataset
    print("Loading the data...")
    dataset = datasets.load_iris()

    # Obtain a normalizing scaler for scaling new data if added later
    std_scaler = preprocessing.StandardScaler().fit(dataset.data)
    
    # Normalize the data
    std_data = preprocessing.scale(dataset.data)
    
    # Randomize the dataset and divide the data for testing and training 
    # The following line uses the iris dataset from sklearn
    data_train, data_test, targets_train, targets_test = train_test_split(
            std_data, dataset.target, test_size = 0.30, random_state=42)
    
    one_hot_targets_train = one_hot_encode_targets(targets_train)
    one_hot_targets_test  = one_hot_encode_targets(targets_test)
    
    # Obtain a classifier. 
    classifier = NNetClassifier()
    classifier.set_num_targets(3)

    # Fit the data
    print("Training...")
    classifier.fit(data_train, one_hot_targets_train)
    
    # Get the predicted targets
    print("Testing...")
    targets_predicted = classifier.predict(data_test)
    
    print("Predictions:")
#    print(targets_predicted)
    
    # Calculate and display the accuracy
    print("Calculating the accuracy...")
    num_predictions = len(targets_predicted)    
    correct_count = 0

#    print(targets_predicted)
#    print(one_hot_targets_test)

    # Convert the pandas dataframe to a list of lists for accuracy compare
    targets_test_as_lists = one_hot_targets_test.values.tolist()

    for i in range(num_predictions):
        if targets_predicted[i] == targets_test_as_lists[i]:
            correct_count+=1
            
    accuracy = float(correct_count) / float(num_predictions)
    print("Total Accuracy: {:.2f}%".format(accuracy * 100.0))
    

main()