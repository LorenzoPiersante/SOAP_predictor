# 25-07-2022

# This file contains the aggregator functions for:
# - outputs of attention heads (node encoder and SOAP decoder stages)
# - hidden vectors outputs of the graph encoder
# - aggregator for graph_encoding of pre-classified nodes and newly classified nodes
# - GRU aggregator for combining node and graph encodings

#IMPORTS
import numpy as np
import torch
import torch.nn as nn

class gated_aggr_attn(nn.Module):
    '''
    This class computes the aggregated tensor out of the outputs of the node encoder and the SOAP decoder.
    It operates on input vectors all of the same size, and retuns an output of the same size.

    Class initialisation:
    - vector_size: size of the input vectors

    Forward arguments:
    - input_vectors: tensor, shape Nxvector_size, these are the outputs of the previous attention layer
    '''
    def __init__(self, vector_size):
        super().__init__()
        #produce the gates for aggregation
        self.gate1 = nn.Linear(vector_size, vector_size, bias = True)
        self.gate2 = nn.Linear(vector_size, vector_size, bias = True)
        #gate activation functions
        self.F_1 = nn.PReLU(num_parameters=1) #this is a LeakyReLU where the negative_slope is a parameter
        self.F_2 = nn.Tanh()
        
    def forward(self, input_vectors):
        #produce the gate for each input vector
        gate = self.gate1(input_vectors)
        gate = self.F_1(gate)
        gate = self.gate2(gate)
        gate = self.F_2(gate)
        
        #compute gated vectors via element wise multiplication
        gated_vectors = gate*input_vectors
        
        #sum the rows of the tensor
        aggregated_output = torch.sum(gated_vectors, dim=0)
        
        return aggregated_output

class aggr_graph(nn.Module):
    '''
    This class computes the aggregated tensor out of the outputs of the graph encoder.

    Class initialisation:
    - input_size: size of the input vectors
    - target_size: size of output vectors

    Forward arguments:
    - hidden_vectors: tensor, shape Nxvector_size, these are the outputs of graph encoding stage
    (done node by node)
    '''
    def __init__(self, input_size, target_size):
        super().__init__()
        #reshape the input hidden vectors
        self.reshape_input1 = nn.Linear(input_size, target_size, bias = True)
        self.reshape_input2 = nn.Linear(target_size, target_size, bias = True)
        
        #produce the gates for aggregation
        self.gate1 = nn.Linear(input_size, target_size, bias = True)
        self.gate2 = nn.Linear(target_size, target_size, bias = True)
        #gate activation functions
        self.F_1 = nn.PReLU(num_parameters=1) #this is a LeakyReLU where the negative_slope is a parameter
        self.F_2 = nn.Sigmoid()
        
    def forward(self, hidden_vectors):
        #reshape the inputs to correct size
        x = self.reshape_input1(hidden_vectors)
        reshaped_input = self.reshape_input2(x)
        
        #create gate
        y = self.gate1(hidden_vectors)
        y = self.F_1(y)
        y = self.gate2(y)
        gate = self.F_2(y)
        
        #compute gated vectors via element wise multiplication
        gated_vectors = gate*reshaped_input
        
        #sum the rows of the tensor
        aggregated_output = torch.sum(gated_vectors, dim=0) #this removes one dimension when dim=0 is 1
        
        return aggregated_output

class aggr_graph_new_node(nn.Module):
    '''
    This class computes the aggregated tensor out of the aggreagted graph encoding and the new node encoding.

    Class initialisation:
    - graph_size : size of the aggregated graph encoding, and also output size for new node encoding
    - input_size : size of input new node encoding
    - intermediate_size: size of intermediate output of processing of the new_node_encoding

    Forward arguments:
    - graph_encoding : aggregated graph encoding (output of aggr_graph() fn)
    - new_node_encoding : output of encoding initialisation for the new node
    '''

    def __init__(self, graph_size, input_size, intermediate_size):
        super().__init__()
        # processing stages of new_input
        self.process_node1 = nn.Linear(input_size, intermediate_size, bias=True)
        self.process_node2 = nn.Linear(intermediate_size, graph_size, bias=True)
        self.process_node3 = nn.Linear(graph_size, graph_size, bias=True)
        # processing activation functions
        self.P_1 = nn.PReLU()
        self.P_2 = nn.PReLU()

        # produce the gates for aggregation
        self.gate1 = nn.Linear(graph_size, graph_size, bias=True)
        self.gate2 = nn.Linear(graph_size, graph_size, bias=True)
        # gate activation functions
        self.F_1 = nn.PReLU(num_parameters=1)  # this is a LeakyReLU where the negative_slope is a parameter
        self.F_2 = nn.Sigmoid()

        # N.B. A lot of porcessing in needed since the initialisation vector is just dim(16) with dim(48) zero
        # padding, so in comparison to the graph encodig is very poor in structure.

    def forward(self, graph_encoding, new_node_encoding):
        # process new_node_encoding
        z = self.process_node1(new_node_encoding)
        z = self.P_1(z)
        z = self.process_node2(z)
        z = self.P_2(z)
        z = self.process_node3(z)
        
        # generate input for gate mechanism
        #stacking is possible since both graph encoding and z have the same size [N], N = 128 in model blueprint
        x = torch.stack([graph_encoding, z])

        # gate mechanism
        y = self.gate1(x)
        y = self.F_1(y)
        y = self.gate2(y)
        gate = self.F_2(y)

        # compute gated vectors via element wise multiplication
        gated_vectors = gate * x

        # sum the rows of the tensor
        aggregated_output = torch.sum(gated_vectors, dim=0)

        return aggregated_output

class GRU_aggr(nn.Module):
    '''
    This function aggregates the final graph encoding with the node encodings, one at a time.

    Class initialisation:
    - input size : size of the two encodings (it must be the same), the function assumes dim = [N], N = 128
    in blueprint

    Forward arguments:
    - graph_encoding : final graph encoding
    - cetre_nodes_encoding : encoding involving nodes and centres
    '''
    def __init__(self, input_size):
        super().__init__()
        #define the GRU aggregator
        self.gru_aggr = nn.GRU(input_size, input_size) #1st input is input seq, 2nd input is h_0
        
    def forward(self, graph_encoding, cetre_nodes_encoding):
        graph_encoding = torch.unsqueeze(graph_encoding, dim=0) #add two dimension at the specified location
        graph_encoding = torch.unsqueeze(graph_encoding, dim=0)
        cetre_nodes_encoding = torch.unsqueeze(cetre_nodes_encoding, dim=0) #add two dimension at the specified location
        cetre_nodes_encoding = torch.unsqueeze(cetre_nodes_encoding, dim=0)
        
        #input dim: [1, 1, input_size]
        #init hidden state dim: [1, 1, hidden_size]
        output = self.gru_aggr(graph_encoding, cetre_nodes_encoding)[1]
        #output dim: [1, 1, hidden_size]
        
        #extract output vector from embedding, i.e. output dim is [hidden_size]
        output = torch.squeeze(output, dim=0)
        output = torch.squeeze(output, dim=0)
        
        return output
