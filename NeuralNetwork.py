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
import matplotlib.pyplot as plt


accuracy_graph = []

class NNetNode:
    def __init__(self):
        self.inputs = []
        self.weights = []
        self.old_weights = []
        self.batch_weights = [] # List of lists of weights. Each list contains all weight calculations for the batch
        self.output = -1
        self.thresh = 0
        self.error = 0

    def initialize_weights(self, num_weights):
        self.weights = randn(num_weights+1)
        for i in range(num_weights+1):
            self.batch_weights.append([])
    
    def reset_batch_weights(self):
        self.batch_weights = []
        for i in range(len(self.weights)):
            self.batch_weights.append([])
                
    def set_inputs(self, passed_inputs):
        self.inputs = passed_inputs
        self.inputs = np.append(self.inputs,-1) # Add a bias node

    def _activation_fun(self, raw_output):
        # Sigmoid function
        sigmoid_val = 1/(1+np.e**(-1*raw_output))
        return sigmoid_val
    
    def get_output(self):
        # Calculate the summation of inputs * weights
        raw_output = 0
        for i in range(len(self.inputs)):
            raw_output += self.inputs[i] * self.weights[i]
            
        self.output = self._activation_fun(raw_output)
        return self.output
   
class NNetClassifier:
    def __init__(self):
        self.net = [] # A list of lists of nodes. Each list of nodes represents a layer
        self.hidden_layers = [] # A list of lists of nodes. Each list of nodes represents a layer.
        self.outputs = []
        self.learning_rate = 0.1
    
    # Take a list of layer sizes. Each size in the list will create a layer consisting of 'size' nodes
    # The hidden layers must be specified as well as the size of the output layer
    def configure_hidden_layers(self, layer_sizes): 
        
        # Add each layer of nodes to the net
        for layer_index in range(len(layer_sizes)):
            nodes = []
            for i in range(layer_sizes[layer_index]):
                # Create the node to be added to the layer
                node = NNetNode()

                # Skip the first layer since we don't know how many inputs yet
                if layer_index-1 > -1:
                    # Assign weights for each output from the previous layer
                    node.initialize_weights(layer_sizes[layer_index-1])

                # Add the node to the layer
                nodes.append(node)
                
            # Add the layer to the net
            self.net.append(nodes)     
    
    def _propogate_through_hidden_layers(self, layer_num):
        # The last layer has been reached
        if layer_num == len(self.net):
            return
        
        prev_layer_outputs = []
        
        # Get the outputs of the previous layer
        for node in self.net[layer_num - 1]:
            prev_layer_outputs.append(node.get_output())
            
        # Use the outputs of the previous layer as the inputs to this layer
        for node in self.net[layer_num]:
            node.set_inputs(prev_layer_outputs)
        
        # Recursively propogate to the next layer
        self._propogate_through_hidden_layers(layer_num + 1)
        
    def _set_output_error(self, node, target):
        activation = node.output
        node.error = activation*(1-activation)*(activation-target)
    
    # node: the node for which I'm seeking the error
    # node_index: the index of the node in its layer to be used for getting the right weight
    # layer_index: the index of the layer in which this node resides
    def _set_hidden_error(self, node, node_index, layer_index):
        activation = node.output
        error = 0
        
        k_layer = self.net[layer_index+1]
        
        for k_node in k_layer:
            error += activation*(1-activation)*k_node.old_weights[node_index]*k_node.error
        
        node.error = error
        
    def _update_weights(self, layer_index):
        for node in self.net[layer_index]:
            # Preserve the weights for subsequent weight updates to refer to the correct weight value
            node.old_weights = node.weights 
            
            # Update the weights sequentially, uncomment this and comment out the next two lines to use this
            # Also, comment out the call to self._update_all_weights_to_batch_avg() at the end of .fit
            for i in range(len(node.weights)):
                node.weights[i] = node.old_weights[i]-(self.learning_rate*node.error*node.output)
     
            # Uncomment this to use batch updates and uncomment the call to 
            # self._update_all_weights_to_batch_avg() at the end of the .fit method
            # Store the weights so an average of them can be calculated at the 
            # end of the batch and do a batch update
#            for i in range(len(node.weights)):
#                node.batch_weights[i].append(node.weights[i]-(self.learning_rate*node.error*node.output))
                
    def _update_all_weights_to_batch_avg(self):
        for layer in self.net:
            for node in layer:
                for i in range(len(node.batch_weights)):
                    # Calculate the average of all the weight calculations for the batch
                    avg_weight = np.mean(node.batch_weights[i])                  

                    # Update the node's weights to the average weight
                    node.weights[i] = avg_weight
                node.reset_batch_weights()
                
                
        
    def _back_propogate(self, targets):
        
        # Obtain the outputs of the final layer (This was usually done in predict before back prop)
        self._get_final_outputs()
        
        # Calculate the errors of the output layer nodes
        target_index = 0
        for node in self.net[-1]:
            self._set_output_error(node, targets[target_index])
            target_index += 1
            
        # Update the weights to the output nodes
        self._update_weights(-1)
        
        # Loop through each of the hidden layers going backwards through the net, skip output layer
        for layer_index in range(len(self.net)-2, -1, -1):
            # Calculatet the errors of the nodes in the hidden layers
            for node_index in range(len(self.net[layer_index])):
                self._set_hidden_error(self.net[layer_index][node_index], node_index, layer_index)
            
            # Update the weights of the nodes in the hidden layer now that we have the errors
            self._update_weights(layer_index)
            
            
    # Train the NNet
    def fit(self, data_train, data_targets, is_first):
        
        first_layer_num = 0
        
        # Initialize the weights of the first layer since we know how many inputs now
        if is_first:
            for node in self.net[0]:
                node.initialize_weights(len(data_train[0]))
        
        # Send each row through the NNet
        target_index = 0
        for row in data_train:            
            # Set the inputs of each node in the first layer
            for node in self.net[first_layer_num]:
                node.set_inputs(row) 
              
            # Propogate the inputs and outputs through each layer
            self._propogate_through_hidden_layers(first_layer_num + 1)
            
            # Get the target values for this row as a list from the pandas dataframe
            targets_list = data_targets.iloc[target_index].tolist()
            
            self._back_propogate(targets_list)
            
            target_index += 1
        
        # Uncomment this and the portion in the update weights method for batch updates
        # instead of sequential updates
#        self._update_all_weights_to_batch_avg()
                
    def _get_final_outputs(self):
        outputs = []
        
        # Get the outputs from the last layer of nodes in the net
        for node in self.net[-1]:
       
            sigmoid_val = node.get_output()
            prediction = -1
            
            # Return 1 if closer to 1
            if sigmoid_val >= 0.5:
                prediction = 1
            else:
                prediction = 0
                
            outputs.append(prediction)
        return outputs
    
    # Run the list of attributes through the network and return the outputs        
    def predict(self, data_test):
        
        predictions = []
        
        for attributes in data_test:
            
            for node in self.net[0]:
                node.set_inputs(attributes)
            
            # Run the new ata through the network
            self._propogate_through_hidden_layers(1)
            
            # Get the last layer's outputs
            predictions.append(self._get_final_outputs())    
        return predictions

#-----------------------------------------------------------------------------

def one_hot_encode_iris_targets(targets):
    one_hot_cols = ["0", "1", "2"]
    one_hot_targets = pd.get_dummies(targets, columns=one_hot_cols)
    return one_hot_targets

def one_hot_encode_b_cancer_targets(targets):
    one_hot_cols = ["0", "1"]
    one_hot_targets = pd.get_dummies(targets, columns=one_hot_cols)
    return one_hot_targets

def do_epoch(classifier, data_train, data_test, one_hot_targets_train, one_hot_targets_test, epoch_cnt, is_first_iter):
    print("-------------------------------------------------")
    print("Epoch {}".format(epoch_cnt))
    
    # Fit the data
    print("Training...")
    classifier.fit(data_train, one_hot_targets_train, is_first_iter)
    
    # Get the predicted targets
    print("Testing...")
    targets_predicted = classifier.predict(data_test)
    
    # Calculate and display the accuracy
    print("Calculating the accuracy...")
    num_predictions = len(targets_predicted)    
    correct_count = 0

    # Convert the pandas dataframe to a list of lists for accuracy compare
    targets_test_as_lists = one_hot_targets_test.values.tolist()

    for i in range(num_predictions):
        if targets_predicted[i] == targets_test_as_lists[i]:
            correct_count+=1
            
    accuracy = float(correct_count) / float(num_predictions)
    accuracy_graph.append(accuracy)
    print("Total Accuracy: {:.2f}%".format(accuracy * 100.0))

def main(): 
    # Load the iris dataset
    print("Loading the data...")
    dataset = datasets.load_iris()
#    dataset = datasets.load_breast_cancer()

    # Obtain a normalizing scaler for scaling new data if added later
    std_scaler = preprocessing.StandardScaler().fit(dataset.data)
    
    # Normalize the data
    std_data = preprocessing.scale(dataset.data)
    
    # Randomize the dataset and divide the data for testing and training 
    # The following line uses the iris dataset from sklearn
    data_train, data_test, targets_train, targets_test = train_test_split(
            std_data, dataset.target, test_size = 0.30, random_state=42)
    
    # Iris Dataset
    one_hot_targets_train = one_hot_encode_iris_targets(targets_train)
    one_hot_targets_test  = one_hot_encode_iris_targets(targets_test)
    
    # Breast Cancer Dataset
#    one_hot_targets_train = one_hot_encode_b_cancer_targets(targets_train)
#    one_hot_targets_test  = one_hot_encode_b_cancer_targets(targets_test)
    
    # Obtain a classifier. 
    classifier = NNetClassifier()
    
    # Two hidden layers of 20 nodes, output layer of 3 nodes
    classifier.configure_hidden_layers([40, 40, 3]) # Iris Dataset
#    classifier.configure_hidden_layers([40, 40, 2])  # Breast Cancer Dataset 
    

    # The first training instance - This randomizes the weights at first
    do_epoch(classifier, data_train, data_test, one_hot_targets_train, one_hot_targets_test, 1, True)

    # Number of epochs to train the net
    for i in range(0, 100):
        epoch_cnt = i + 2 # The first epoch has already been done
        do_epoch(classifier, 
                 data_train, 
                 data_test, 
                 one_hot_targets_train, 
                 one_hot_targets_test, 
                 epoch_cnt, 
                 False)

    # Graph the accuracy        
    plt.plot(accuracy_graph)
    plt.show()    

main()