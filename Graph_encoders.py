#25-07-2022

#This file contains the functions to create the hidden state encoding for each node of the molecule graph.
#The function will act on the MoleculeGraph object defined in the Test_Data_structures file.

#27-07-2022

#Modify GRU_graph_enc to have the state of the net as a forward argument.
#In this way we can use the same weights for the final evaluation.

#SEE PROPOSED MODEL FILES FOR DETAILED MATHEMATICAL DESCRIPTION OF THE MODELS

import numpy as np
import torch
import torch.nn as nn

#custom modules
import Data_structures as DS

class hidden_state_init(nn.Module):
    '''
    This function calculates the initial hidden state for each node in the graph starting from the node location
    and one-hot class.
    The initialisation goes as: dim(input) = 3 + 10, dim(hidden state) = 16, cat(hidden state, 48-0s) so that
    dim(final hidden state) = 64.
    
    The function returns a list of tensors of size 64, each of which is initialised hidden state + 48 0s.
    
    Initialisation arguments:
    - input_size : size of position + one-hot class, stored in the attibute .init_node_enc of MoleculeGraph object
    - output_size : size of the initialised hidden state vector (attention, this is not the final 0 padded size)
    - pad_len : length of the 0 padding to turn the initialisation hidden state from size 16 to size 64
    given the model test blueprint this value should be 48
    
    Forward arguments:
    - graph : input MoleculeGraph object, interested in the attribute .init_node_enc
    '''
    def __init__(self, input_size, output_size, pad_len):
        super().__init__()
        #initialisation layers
        self.init1 = nn.Linear(input_size, output_size, bias = True)
        self.init2 = nn.Linear(output_size, output_size, bias = True)
        #layer activation
        self.F_1 = nn.PReLU(num_parameters=1)
        
        #padded output size
        self.pad = int(output_size + pad_len)
        self.output_size = output_size
        
        
    def forward(self, graph):
        '''
        It will return all the initial hidden states as the rows of tensor h_0 dim(number of nodes x 64)
        '''
        
        def init_MLP(position_class):
            '''
            This is the MLP used in the computation of the initial hidden state vector for a given node.

            Args:
            - position_class : this is the position + one-hot class encoding of the node, it is obtained from the
            entries of the .init_node_enc attribute of the MoleculeGraph object (list of tensors object)

            Returns the calculated initialisation hidden vector.
            '''
            x = self.init1(position_class)
            x = self.F_1(x)
            x = self.init2(x)
            return x
        
        N = len(graph.init_node_enc) #number of nodes that have been classified thus far
        
        #allocate memory to initialised hidden vector
        h_0 = torch.zeros((N, self.pad), requires_grad=False, dtype=torch.float)
        
        for x, y in zip(graph.init_node_enc, range(N)):
            h_node_0 = init_MLP(x) #compute initialisation vector
            h_0[y, 0:self.output_size] = h_node_0 #assign initialisation vector
        
        return h_0
    
class GRU_graph_enc(nn.Module):
    '''
   This function calculates the final hidden state encoding by taking the SOAP vectors from the previous
   iteration as inputs.
   
   Initialisation arguments:
   - input_size : size of the input SOAPs
   - hidden_size : size of the hidden state and of h_0_node (initialisation hidden vector for node)
   
   Forward arguments:
   - graph: MoleculeGraph object, we are intereted in the .SOAP_inputs attribute, list of tensot inputs for each
   node
   - h_0 : the initialisation outputs from hidden_state_init
   - iteration : bool argument which tells if the state is in the iteration mode or the final mode
   In iteration mode the GRU acts only on the N-1 classified nodes (not the last one), in final mode it acts 
   on all nodes (SOAP is available for all nodes at this stage of the process)
    '''
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first = True)
        
    def forward(self, graph, h_0, iteration):
        
        #Account for addition of first node
        if iteration and len(h_0) == 1:
            h_0 = torch.unsqueeze(h_0, dim=0) #just to get the same output shape as the GRU
            #in the model workflow we then don't need to introduce conditional statements
            return h_0
        
        #Iteration process
        elif iteration:
            #initial hidden states - each row corresponds to a different node initial hidden state
            input_h_0 = h_0[0:int(len(h_0)-1), :]
            #prepate it for GRU input
            #final dimension dim = 1 x N x 64
            input_h_0 = torch.unsqueeze(input_h_0, dim=0)
            
            #prepare soap inputs - stack together tensors in list
            #final dimension N x L x 64 - batch_first = True in torch.GRU
            SOAP_inputs = torch.stack(graph.SOAP_inputs, dim=0)
            
            #update the hidden states using the SOAP inputs
            output = self.gru(SOAP_inputs, input_h_0)[1]
            
            return output
            
        
        #Filanise SOAP prediction 
        else:
            #prepate h_0 for GRU input
            #final dimension dim = 1 x N x 64
            input_h_0 = torch.unsqueeze(h_0, dim=0)
            
            #prepare soap inputs - stack together tensors in list
            #final dimension N x L x 64 - batch_first = True in torch.GRU
            SOAP_inputs = torch.stack(graph.SOAP_inputs, dim=0)
            
            #update the hidden states using the SOAP inputs
            output = self.gru(SOAP_inputs, input_h_0)[1]
            
            return output