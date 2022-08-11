import torch
import numpy as np
import torch.nn as nn

#This module contains the NN layers used in the SOAP predictor model

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

class attention_mech(nn.Module):
    '''
    This is the implementation of a one-head shallow attention mechanism.
    This attention mechanism takes 4D inputs.
    
    Initialisation arguments:
    - input_r: dim=-1 of the row inputs, i.e. the key inputs.
    - input_c: dim=-1 of the column inputs, i.e. the queries inputs.
    - output_s: dim=-1 of output tensor.
    
    Forward arguments:
    - tensorL: dim(B, -1, -1, input_r)
    - tensorH: dim(B, -1, -1, input_c)
    B: batch dimension.
    L and H refer to the fact that in the SOAP predictor model the first tensor contains info about the light atoms
    position distribution, while the second about the heavier (or of equal weight) atoms.
    
    Usually the dimensions are:
    dim(B, num_centres, num_nodes, info_size)
    where:
    - num_centres: number of centres for which encodings or SOAP are to be computed.
    - num_nodes: relevant nodes involved.
    - info_size: number of infomation entries.
    
    Returns:
    - outputs: dim(B, -1, -1, output_s).
    '''
    
    
    def __init__(self, input_r, input_c, output_s):
        super().__init__()
        self.keys = nn.Linear(input_r, output_s, bias=True)
        self.queries = nn.Linear(input_c, output_s, bias=True)
        self.values = nn.Linear(2*output_s, output_s, bias=True)

        self.F = nn.Softmax(dim=2)
        
    def forward(self, tensorL, tensorH):
        
        #calculate keys and queries vectors
        keys = self.keys(tensorL)
        queries = self.queries(tensorH)
        
        #apply attention
        attention = torch.matmul(keys, torch.transpose(queries, 2, 3))
        attention = self.F(attention)
        #divide the tensors in the last two dimensions along the row direction into "row" vectors
        attention = torch.unsqueeze(attention, dim=3)
        
        #create input for values using keys and queries
        #each new tensor is used to generate the values for a specific key
        value_inputs_list = []
        for n in range(keys.size()[2]):
            #allocate memory
            new_tensor = torch.zeros((queries.size()[0], queries.size()[1], queries.size()[2], 2*queries.size()[3]), requires_grad=False, dtype=torch.float, device=my_device)
            #assign all queries to input
            new_tensor[:, :, :, 0:queries.size()[3]] = queries
            #assign specific key to input
            new_tensor[:, :, :, queries.size()[3]:2*queries.size()[3]] = torch.unsqueeze(keys[:, :, n, :], dim=2)
            #append to input list
            value_inputs_list.append(new_tensor)
        #stack along dim=2, new keys dimension
        value_inputs = torch.stack(value_inputs_list, dim=2)
        #resulting tensor is dim(B, num_centres, num_keys, num_queries, value_size)
        
        #create values - num_values per key = num_queries
        values = self.values(value_inputs)
        
        #compute output
        outputs = torch.matmul(attention, values)
        #this matrix multiplication is equivalent to contraction of values along the queries direction
        #output will have dim(B, num_centres, num_keys, 1, value_size)
        outputs = torch.squeeze(outputs, dim=3) #remove dummy dimension
        
        return outputs
    
class attention_mech_deep(nn.Module):
    '''
    This is the implementation of a one-head shallow attention mechanism.
    This attention mechanism takes 4D inputs.
    It is different from the attention_mech() class since the keys, queries and value encoders
    are deeeper nets.
    
    Initialisation arguments:
    - input_r: dim=-1 of the row inputs, i.e. the key inputs.
    - input_c: dim=-1 of the column inputs, i.e. the queries inputs.
    - output_s: dim=-1 of output tensor.
    
    Forward arguments:
    - tensorL: dim(B, -1, -1, input_r)
    - tensorH: dim(B, -1, -1, input_c)
    B: batch dimension.
    L and H refer to the fact that in the SOAP predictor model the first tensor contains info about the light atoms
    position distribution, while the second about the heavier (or of equal weight) atoms.
    
    Usually the dimensions are:
    dim(B, num_centres, num_nodes, info_size)
    where:
    - num_centres: number of centres for which encodings or SOAP are to be computed.
    - num_nodes: relevant nodes involved.
    - info_size: number of infomation entries.
    
    Returns:
    - outputs: dim(B, num_centres, num_nodes, output_s).
    '''
    
    def __init__(self, input_r, input_c, output_s):
        super().__init__()
        #keys layers
        self.keys = nn.Linear(input_r, output_s, bias=True)
        self.keys2 = nn.Linear(output_s, output_s, bias=True)
        self.keys3 = nn.Linear(output_s, output_s, bias=True)
        #queries layers
        self.queries = nn.Linear(input_c, output_s, bias=True)
        self.queries2 = nn.Linear(output_s, output_s, bias=True)
        #values layers
        self.values = nn.Linear(2*output_s, output_s, bias=True)
        self.values2 = nn.Linear(output_s, output_s, bias=True)
        self.values3 = nn.Linear(output_s, output_s, bias=True)
        
        #activation functions
        self.F = nn.Softmax(dim=2)
        self.T = nn.Tanh()
        self.ActK = nn.PReLU(num_parameters=1) #for keys
        self.ActV = nn.PReLU(num_parameters=1) #for values
        
        #normalisation layers for keys and values
        self.normK = nn.LayerNorm(output_s) #for keys
        self.normV = nn.LayerNorm(output_s) #for values
        
    def forward(self, tensorL, tensorH):
        
        #calculate keys
        keys = self.keys(tensorL)
        keys = self.ActK(keys)
        keys = self.keys2(keys)
        keys = self.T(keys)
        keys = self.normK(keys)
        keys = self.keys3(keys)
        
        #calculate queries
        queries = self.queries(tensorH)
        queries = self.T(queries)
        queries = self.queries2(queries)
        
        #apply attention
        attention = torch.matmul(keys, torch.transpose(queries, 2, 3))
        attention = self.F(attention)
        #divide the tensors in the last two dimensions along the row direction into "row" vectors
        attention = torch.unsqueeze(attention, dim=3)
        
        #create input for values using keys and queries
        #each new tensor is used to generate the values for a specific key
        value_inputs_list = []
        for n in range(keys.size()[2]):
            #allocate memory
            new_tensor = torch.zeros((queries.size()[0], queries.size()[1], queries.size()[2], 2*queries.size()[3]), requires_grad=False, dtype=torch.float, device=my_device)
            #assign all queries to input
            new_tensor[:, :, :, 0:queries.size()[3]] = queries
            #assign specific key to input
            new_tensor[:, :, :, queries.size()[3]:2*queries.size()[3]] = torch.unsqueeze(keys[:, :, n, :], dim=2)
            #append to input list
            value_inputs_list.append(new_tensor)
        #stack along dim=2, new keys dimension
        value_inputs = torch.stack(value_inputs_list, dim=2)
        #resulting tensor is dim(B, num_centres, num_keys, num_queries, value_size)
        
        #create values - num_values per key = num_queries
        values = self.values(value_inputs)
        values = self.ActV(values)
        values = self.values2(values)
        values = self.T(values)
        values = self.normV(values)
        values = self.values3(values)
        
        #compute output
        outputs = torch.matmul(attention, values)
         #this matrix multiplication is equivalent to contraction of values along the queries direction
        #output will have dim(B, num_centres, num_keys, 1, value_size)
        outputs = torch.squeeze(outputs, dim=3) #remove dummy dimension
        
        return outputs
    
class aggregator(nn.Module):
    '''
    This class aggregates a 4D tensor along dim=2.
    
    Initialisation arguments:
    - input_size: dim=-1 of input tensor
    - output_size: dim=-1 of output tensor
    
    Forward arguments:
    - input_tensor: dim(B, -1_1, -1_2, input_size) tensor
    
    Returns:
    - output: dim (B, -1_1, output_size) tensor
    '''
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.filter = nn.Linear(input_size, output_size, bias=True)
        self.new_val = nn.Linear(input_size, output_size, bias=True)
        
        self.F = nn.Tanh()
        
    def forward(self, input_tensor):
        
        #compute filer
        filter_tensor = self.filter(input_tensor)
        filter_tensor = self.F(filter_tensor)
        
        #compute new tensor values
        values = self.new_val(input_tensor)
        
        #gated aggregation
        output = filter_tensor*values
        output = torch.sum(output, dim=2)
        
        return output

class deep_aggregator(nn.Module):
    '''
    This class aggregates a 4D tensor along dim=2.
    This class contains deeper nets for the calculation of the gate and the value tensors.
    
    Initialisation arguments:
    - input_size: dim=-1 of input tensor
    - intermediate_size: dim=-1 of hidden layer of networks
    - output_size: dim=-1 of output tensor
    
    Forward arguments:
    - input_tensor: dim(B, -1_1, -1_2, input_size) tensor
    
    Returns:
    - output: dim (B, -1_1, output_size) tensor
    '''
    
    
    def __init__(self, input_size, intermediate_size, output_size):
        super().__init__()
        #filter layers
        self.filter1 = nn.Linear(input_size, intermediate_size, bias=True)
        self.filter2 = nn.Linear(intermediate_size, output_size, bias=True)
        #value layers
        self.val1 = nn.Linear(input_size, intermediate_size, bias=True)
        self.val2 = nn.Linear(intermediate_size, output_size, bias=True)
        
        self.F1 = nn.PReLU(num_parameters=1) #device=
        self.F2 = nn.Tanh()
        
    def forward(self, input_tensor):
        
        #compute gate
        filter_tensor1 = self.filter1(input_tensor)
        filter_tensor1 = self.F1(filter_tensor1)
        #compute values
        value1 = self.val1(input_tensor)
        
        #first filter application
        intermediate_tensor = filter_tensor1*value1
        
        #compute second gate
        filter_tensor2 = self.filter2(intermediate_tensor)
        filter_tensor2 = self.F2(filter_tensor2)
        #comnpute second values
        value2 = self.val2(intermediate_tensor)
        
        #final gated aggregation
        output = filter_tensor2*value2
        output = torch.sum(output, dim=2)
        
        return output