#22-07-2022

#In this script we will create the attention heads for the SOAP predictor model
#- self-attention head
#- chem env head

#SEE PROPOSED MODEL FILES FOR DETAILED MATHEMATICAL DESCRIPTION OF THE MODELS

#IMPORTS
import numpy as np
import torch
import torch.nn as nn


class self_attn_init(nn.Module):
    '''
    This class defines the initialisation step of the nodes encoder block.

    Class initialisation:
    - input_size: total dimension of the input, dim(concatenate(node, centre))
    - output_size: size of the output vectors, which is also the size of keys, queries and values

    Forward arguments:
    - nodes: ndarray, list of nodes composing the graph, shape Nx3
    - centre: ndarray, centre, shape 1x3
    '''

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.query = nn.Linear(input_size, output_size, bias=True)
        self.key = nn.Linear(input_size, output_size, bias=True)
        self.values = nn.Linear(2 * output_size, output_size, bias=True)

        # N.B. Linear only supports float dtype
        self.F = nn.Softmax(dim=1)  # fast direction

    def forward(self, nodes, centre):
        # generate concatenated input (must speecify no gradient and dtype = float)
        T_nodes = torch.tensor(nodes, requires_grad=False, dtype=torch.float)
        centre = torch.tensor(centre, requires_grad=False, dtype=torch.float)
        T_centre = torch.zeros((T_nodes.size(dim=0), centre.size(dim=1)), requires_grad=False, dtype=torch.float)
        for i in range(T_nodes.size(dim=0)):
            T_centre[i, :] = centre
        x_nodes_centre = torch.cat((T_nodes, T_centre), dim=1)

        # generate keys and queries
        keys = self.key(x_nodes_centre)
        queries = self.query(x_nodes_centre)

        # attention mechanism
        attention = self.F(torch.matmul(keys, torch.transpose(queries, 0, 1)))

        # pre-allocate output
        outputs = torch.zeros((T_nodes.size(dim=0), self.output_size), requires_grad=False, dtype=torch.float)
        # compute output row by row
        for n in range(T_nodes.size(dim=0)):
            # create key-queries input
            ref_key = torch.zeros((T_nodes.size(dim=0), self.output_size), requires_grad=False, dtype=torch.float)
            for m in range(T_nodes.size(dim=0)):
                ref_key[m, :] = keys[n, :]
            x_queries_key = torch.cat((queries, ref_key), dim=1)

            # create values for referece key
            values_ref_key = self.values(x_queries_key)

            # compute output vector
            output_vector = torch.matmul(attention[n], values_ref_key)

            # assign them to output vector
            outputs[n, :] = output_vector

        # apply skip connection
        outputs = keys + outputs

        return outputs

class self_attn_enc(nn.Module):
    '''
    This class defines a layer of the nodes encoder block.
    It is almost identical to the initialisation block, but it takes the output vectors of the previous
    layer as input and uses them in the definition of new keys and queries.

    Class initialisation:
    - input_size: total dimension of the input, dim(concatenate(node, centre, output layer l-1))
    - output_size: size of the output vectors, which is also the size of keys, queries and values

    Forward arguments:
    - nodes: ndarray, list of nodes composing the graph, shape Nx3
    - centre: ndarray, centre, shape 1x3
    - out_l_1 = output vectors from layer l-1 of the decoder block, shape Nx(any dim)
    '''
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.query = nn.Linear(input_size, output_size, bias = True)
        self.key = nn.Linear(input_size, output_size, bias = True)
        self.values = nn.Linear(2*output_size, output_size, bias = True)

        #N.B. Linear only supports float dtype
        self.F = nn.Softmax(dim = 1) #fast direction

    def forward(self, nodes, centre, out_l_1):
        #generate concatenated input (must speecify no gradient and dtype = float)
        T_nodes = torch.tensor(nodes, requires_grad = False, dtype = torch.float)
        centre = torch.tensor(centre, requires_grad = False, dtype = torch.float)
        T_centre = torch.zeros((T_nodes.size(dim=0), centre.size(dim=1)), requires_grad = False, dtype = torch.float)
        for i in range(T_nodes.size(dim=0)):
            T_centre[i, :] = centre
        #concatenate the output of the previous self atn layer to the other inputs
        x_nodes_centre_out = torch.cat((T_nodes, T_centre, out_l_1), dim=1)

        #generate keys and queries
        keys = self.key(x_nodes_centre_out)
        queries = self.query(x_nodes_centre_out)

        #attention mechanism
        attention = self.F(torch.matmul(keys, torch.transpose(queries, 0, 1)))

        #pre-allocate output
        outputs = torch.zeros((T_nodes.size(dim=0), self.output_size), requires_grad = False, dtype = torch.float)
        #compute output row by row
        for n in range(T_nodes.size(dim=0)):
            #create key-queries input
            ref_key =  torch.zeros((T_nodes.size(dim=0), self.output_size), requires_grad = False, dtype = torch.float)
            for m in range(T_nodes.size(dim=0)):
                ref_key[m, :] = keys[n, :]
            x_queries_key = torch.cat((queries, ref_key), dim=1)

            #create values for referece key
            values_ref_key = self.values(x_queries_key)

            #compute output vector
            output_vector = torch.matmul(attention[n], values_ref_key)

            #assign them to output vector
            outputs[n, :] = output_vector

        #apply skip connection
        outputs = keys + outputs

        return outputs


class SOAP_dec(nn.Module):
    '''
    This is a layer of the decoder block that predicts SOAP for a given centre (node).

    Class initialisation:
        - input_size: total dimension of the input, dim(concatenate(node, centre, system encoding vector))
        - output_size: size of the output vectors, which is also the size of keys, queries and values

    Forward arguments:
    - nodesL: ndarray containing the location of the nodes classified as ligher Z, shape Nx3
    - nodesH: ndarray containing the location of the nodes classified as heavier Z, shape Mx3
    - centre: ndarray of the centre with respect to which SOAP is computed, shape 1x3
    - system_vect: tensor of system wide vector that encodes the graph and the nodes, specific to the centre with respect
    to which SOAP is calculated, shape 1x*
    '''

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.query = nn.Linear(input_size, output_size, bias=True)
        self.key = nn.Linear(input_size, output_size, bias=True)
        self.values = nn.Linear(2 * output_size, output_size, bias=True)

        # N.B. Linear only supports float dtype
        self.F = nn.Softmax(dim=1)  # fast direction

    def forward(self, nodesL, nodesH, centre, system_vect):
        # generate concatenated input (must speecify no gradient and dtype = float)
        nodesL = torch.tensor(nodesL, requires_grad=False, dtype=torch.float)
        nodesH = torch.tensor(nodesH, requires_grad=False, dtype=torch.float)
        centre = torch.tensor(centre, requires_grad=False, dtype=torch.float)
        T_centreL = torch.zeros((nodesL.size(dim=0), centre.size(dim=1)), requires_grad=False, dtype=torch.float)
        for i in range(nodesL.size(dim=0)):
            T_centreL[i, :] = centre
        T_centreH = torch.zeros((nodesH.size(dim=0), centre.size(dim=1)), requires_grad=False, dtype=torch.float)
        for i in range(nodesH.size(dim=0)):
            T_centreH[i, :] = centre
        system_vect = torch.tensor(system_vect, requires_grad=False, dtype=torch.float)
        T_systemL = torch.zeros((nodesL.size(dim=0), system_vect.size(dim=1)), requires_grad=False, dtype=torch.float)
        for i in range(nodesL.size(dim=0)):
            T_systemL[i, :] = system_vect
        T_systemH = torch.zeros((nodesH.size(dim=0), system_vect.size(dim=1)), requires_grad=False, dtype=torch.float)
        for i in range(nodesH.size(dim=0)):
            T_systemH[i, :] = system_vect
        # concatenate inputs for key and queries
        x_keys = torch.cat((nodesL, T_centreL, T_systemL), dim=1)
        x_queries = torch.cat((nodesH, T_centreH, T_systemH), dim=1)

        # generate keys and queries
        keys = self.key(x_keys)
        queries = self.query(x_queries)

        # attention mechanism
        attention = self.F(torch.matmul(keys, torch.transpose(queries, 0, 1)))

        # pre-allocate output
        outputs = torch.zeros((nodesL.size(dim=0), self.output_size), requires_grad=False, dtype=torch.float)
        # compute output row by row
        for n in range(nodesL.size(dim=0)):
            # create key-queries input
            ref_key = torch.zeros((nodesH.size(dim=0), self.output_size))
            for m in range(nodesH.size(dim=0)):
                ref_key[m, :] = keys[n, :]
            x_queries_key = torch.cat((queries, ref_key), dim=1)

            # create values for referece key
            values_ref_key = self.values(x_queries_key)

            # compute output vector
            output_vector = torch.matmul(attention[n], values_ref_key)

            # assign them to output vector
            outputs[n, :] = output_vector

        # apply skip connection
        outputs = keys + outputs

        return outputs