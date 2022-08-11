import torch
import torch.nn as nn
import numpy as np

import NN_layers as NN
import Utils as Ul

#08-08-2022
#removed skip connection to SOAP result from previous iteration
#added normalisation layers
#made the attentiom mech deeper

#GRAPH ENCODING LAYER

class graph_encoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.attn1 = NN.attention_mech_deep(67, 67, 64)
        self.attn2 = NN.attention_mech_deep(64, 64, 128)
        self.aggr = NN.deep_aggregator(128, 128, 128)
        
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(128)
        
    def forward(self, input_graph_enc):
        '''
        Input: input_graph_enc, first output of make_network_inputs() fn in model_inputs
        
        Returns:
        - graph_enc: dim(B, env, 128)
        '''
        
        X = self.attn1(input_graph_enc, input_graph_enc)
        X = self.norm1(X)
        X = self.attn2(X, X)
        X = self.norm2(X)
        X = self.aggr(X)
        
        return X
    
#GLOBAL NODE-CENTRE ENCODER

class node_centre_encoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.attn1 = NN.attention_mech_deep(8, 8, 16)
        self.attn2 = NN.attention_mech_deep(16, 16, 32)
        self.attn3 = NN.attention_mech_deep(32, 32, 64)
        self.aggr = NN.deep_aggregator(64, 64, 64)
        
        self.norm1 = nn.LayerNorm(16)
        self.norm2 = nn.LayerNorm(32)
        self.norm3 = nn.LayerNorm(64)
        
    def forward(self, input_node_centre_enc):
        '''
        Input: input_node_centre_enc, second output of make_network_inputs() fn in model_inputs
        
        Returns:
        - nc_enc: dim(B, N, 64)
        '''
        
        X = self.attn1(input_node_centre_enc, input_node_centre_enc)
        X = self.norm1(X)
        X = self.attn2(X, X)
        X = self.norm2(X)
        X = self.attn3(X, X)
        X = self.norm3(X)
        X = self.aggr(X)
        
        return X

#SOAP PREDICTION

#New env

class SOAP_pred_head_nenv(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.attn1 = NN.attention_mech_deep(72, 72, 32)
        self.attn2 = NN.attention_mech_deep(40, 40, 16)
        self.attn3 = NN.attention_mech_deep(24, 24, 16)
        self.aggr = NN.deep_aggregator(16, 16, 16)
        
        self.normL1 = nn.LayerNorm(32)
        self.normH1 = nn.LayerNorm(32)
        self.normL2 = nn.LayerNorm(16)
        self.normH2 = nn.LayerNorm(16)
        self.normL3 = nn.LayerNorm(16)
        self.normH3 = nn.LayerNorm(16)
        
    def forward(self, nodes_L_nce, nodes_H_nce, nodes_L, nodes_H):
        '''
        This class generates the SOAP prediction for component l for newly added chemical environments
        
        Inputs:
        - nodes_L_nce: node-centre for class L with global nce, dim(B, C, C_L, 70) (produced combining 
        input_SOAP_pred[L] and nce using class_nce_combination in Utils)
        - nodes_H_nce: node-centre for class H with global nce, dim(B, C, C_H, 70) (produced combining 
        input_SOAP_pred[H] and nce using class_nce_combination in Utils)
        - nodes_L: node-centre for class L (input_SOAP_pred[H]), dim(B, C, C_L, 6)
        - nodes_H: node-centre for class H (input_SOAP_pred[H]), dim(B, C, C_H, 6)
        
        L: lighter atom, lower Z
        H: heavier atom, greater Z
        input_SOAP_pred: 3rd output of make_network_inputs() fn in model_inputs
        
        Returns:
        - SOAP_component: dim(B, N, 16)
        '''
        
        outL1 = self.attn1(nodes_L_nce, nodes_H_nce)
        outH1 = self.attn1(nodes_H_nce, nodes_L_nce)
        outL1 = self.normL1(outL1)
        outH1 = self.normH1(outH1)
        
        nodes_L1 = torch.cat((nodes_L, outL1), -1)
        nodes_H1 = torch.cat((nodes_H, outH1), -1)
        
        outL2 = self.attn2(nodes_L1, nodes_H1)
        outH2 = self.attn2(nodes_H1, nodes_L1)
        outL2 = self.normL2(outL2)
        outH2 = self.normH2(outH2)
        
        #new layer with skip connection
        nodes_L2 = torch.cat((nodes_L, outL2), -1)
        nodes_H2 = torch.cat((nodes_H, outH2), -1)
        
        outL3 = self.attn3(nodes_L2, nodes_H2)
        outH3 = self.attn3(nodes_H2, nodes_L2)
        outL3 = self.normL3(outL3)
        outH3 = self.normH3(outH3)
        
        outL3 = outL3 + outL2
        outH3 = outH3 + outH2
        
        out = torch.cat((outL3, outH3), 2)
        out = self.aggr(out)
        
        return out

#Previously classified env

class SOAP_pred_head(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.attn1 = NN.attention_mech_deep(72, 136, 64)
        self.attn2 = NN.attention_mech_deep(72, 72, 32)
        self.attn3 = NN.attention_mech_deep(40, 40, 16)
        self.attn4 = NN.attention_mech_deep(24, 24, 16)
        self.aggr = NN.deep_aggregator(16, 16, 16)
        
        self.normL1 = nn.LayerNorm(64)
        self.normH1 = nn.LayerNorm(64)
        self.normL2 = nn.LayerNorm(32)
        self.normH2 = nn.LayerNorm(32)
        self.normL3 = nn.LayerNorm(16)
        self.normH3 = nn.LayerNorm(16)
        self.normL4 = nn.LayerNorm(16)
        self.normH4 = nn.LayerNorm(16)
        
    def forward(self, nodes_L_nce, nodes_H_nce, nodes_L_env, nodes_H_env, nodes_L, nodes_H):
        '''
        This class generates the SOAP prediction for component l for previously classified chemical
        environments.
        
        Inputs:
        - nodes_L_nce: node-centre for class L with global nce, dim(B, C, C_L, 70) (produced combining 
        input_SOAP_pred[L] and nce using class_nce_combination in Utils)
        - nodes_H_nce: node-centre for class H with global nce, dim(B, C, C_H, 70) (produced combining 
        input_SOAP_pred[H] and nce using class_nce_combination in Utils)
        - nodes_L_env: node-centre for class L with env information, dim(B, C, C_L, 134) (produced combining 
        input_SOAP_pred[L] and env aggregated vector using class_env_combination in Utils
        - nodes_H_env: node-centre for class H with env information, dim(B, C, C_H, 134) (produced combining 
        input_SOAP_pred[H] and env aggregated vector using class_env_combination in Utils
        - nodes_L: node-centre for class L (input_SOAP_pred[H]), dim(B, C, C_L, 6)
        - nodes_H: node-centre for class H (input_SOAP_pred[H]), dim(B, C, C_H, 6)
        
        L: lighter atom, lower Z
        H: heavier atom, greater Z
        input_SOAP_pred: 3rd output of make_network_inputs() fn in model_inputs
        
        Returns:
        - SOAP_component: dim(B, N, 16)
        '''
        
        outL1 = self.attn1(nodes_L_nce, nodes_H_env)
        outH1 = self.attn1(nodes_H_nce, nodes_L_env)
        outL1 = self.normL1(outL1)
        outH1 = self.normL1(outH1)
        
        nodesL1 = torch.cat((nodes_L, outL1), -1)
        nodesH1 = torch.cat((nodes_H, outH1), -1)
        
        outL2 = self.attn2(nodesL1, nodesH1)
        outH2 = self.attn2(nodesH1, nodesL1)
        outL2 = self.normL2(outL2)
        outH2 = self.normL2(outH2)
        
        nodesL2 = torch.cat((nodes_L, outL2), -1)
        nodesH2 = torch.cat((nodes_H, outH2), -1)
        
        outL3 = self.attn3(nodesL2, nodesH2)
        outH3 = self.attn3(nodesH2, nodesL2)
        outL3 = self.normL3(outL3)
        outH3 = self.normL3(outH3)
        
        #new layer with skip connection
        nodesL3 = torch.cat((nodes_L, outL3), -1)
        nodesH3 = torch.cat((nodes_H, outH3), -1)
        
        outL4 = self.attn4(nodesL3, nodesH3)
        outH4 = self.attn4(nodesH3, nodesL3)
        outL4 = self.normL4(outL4)
        outH4 = self.normL4(outH4)
        outL4 = outL4 + outL3
        outH4 = outH4 + outH3
       
        out = torch.cat((outL4, outH4), 2)
        out = self.aggr(out)
        
        return out
    
#MODEL DEFINITION

class SOAP_predictor_model(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.graph_encoder = graph_encoder()
        
        self.node_centre_encoder = node_centre_encoder()
        
        #SOAP predictor heads, new env
        self.SOAP_n0 = SOAP_pred_head_nenv()
        self.SOAP_n1 = SOAP_pred_head_nenv()
        self.SOAP_n2 = SOAP_pred_head_nenv()
        self.SOAP_n3 = SOAP_pred_head_nenv()
        
        #SOAP predictor head, previously identified environments
        self.SOAP_0 = SOAP_pred_head()
        self.SOAP_1 = SOAP_pred_head()
        self.SOAP_2 = SOAP_pred_head()
        self.SOAP_3 = SOAP_pred_head()
        
    def forward(self, input_graph_enc, input_node_centre_enc, input_SOAP_pred, chem_env_list_n,
               chem_env_list_n_1, sorted_species_list, SOAP_n_1, iteration):
        '''
        This class is an execution of the model over the inputs at iteration N.
        
        Args:
        - input_graph_enc: dim(B, env, N-1, 70), first output of make_network_inputs() fn in model_inputs, dim=3
        is position + SOAP for that centre for the given env dimension, env index is given by standard SOAP
        ordering of chem env at iteration N-1, (None for iter=1);
        - input_node_centre_enc: dim(B, N, N, 6), second output of make_network_inputs() fn in model_inputs, 
        dim = 3 is position given by dim=2 + centre given by dim=1;
        - input_SOAP_pred: list of at most 3 tensors of dim(B, N, N_i, 6), third output of make_network_inputs() 
        fn in model_inputs, dim = 3 is position given by dim=2 + centre given by dim=1, they are used as
        inputs for the SOAP predictor heads;
        - chem_env_list_n: list of the chem env at iteration N according to standard SOAP ordering;
        - chem_env_list_n_1: list of the chem env at iteration N-1 according to standard SOAP ordering (empty
        for iter=1);
        - sorted_species_list: list of chem symbols of species in molecule at iteration N sorted by Z,
        it is needed to access the input_SOAP_pred;
        - SOAP_n_1: SOAP output at previous iteration dim(B, env_n_1, N-1, 64), (None for iter=1);
        - iteration: current number of the iteration (from 1 to N atoms).
        
        Return:
        -SOAP_n: new SOAP 
        '''
        #compute graph encoding if N>1
        if iteration == 1:
            pass
        else:
            graph_enc = self.graph_encoder(input_graph_enc)
        
        #compute node-centre encoding
        nc_enc = self.node_centre_encoder(input_node_centre_enc)
        
        #SOAP prediction
        #store outputs of SOAP prediction
        env_dict = {}
        
        #create list of new environments
        chem_env_set_n = set(chem_env_list_n)
        chem_env_set_n_1 = set(chem_env_list_n_1)
        new_env_list = list(chem_env_set_n - chem_env_set_n_1)
        
        #Produce SOAP for new chemical environments
        if bool(len(new_env_list)):
            
            for env in new_env_list:
                
                L_index = sorted_species_list.index(env[0])
                H_index = sorted_species_list.index(env[1])
                
                nodes_L = input_SOAP_pred[L_index]
                nodes_H = input_SOAP_pred[H_index]
                
                nodes_L_nce = Ul.class_nce_combination(nodes_L, nc_enc, 72)
                nodes_H_nce = Ul.class_nce_combination(nodes_H, nc_enc, 72)
                
                SOAP_l0 = self.SOAP_n0(nodes_L_nce, nodes_H_nce, nodes_L, nodes_H)
                SOAP_l1 = self.SOAP_n1(nodes_L_nce, nodes_H_nce, nodes_L, nodes_H)
                SOAP_l2 = self.SOAP_n2(nodes_L_nce, nodes_H_nce, nodes_L, nodes_H)
                SOAP_l3 = self.SOAP_n2(nodes_L_nce, nodes_H_nce, nodes_L, nodes_H)
                
                SOAP_env = torch.cat((SOAP_l0, SOAP_l1, SOAP_l2, SOAP_l3), dim=2)
                
                env_dict[env] = SOAP_env
    
        else:
            pass
        
        #Produce SOAP for previously classified environments
        
        #***VALID IF INPUT IS ORDERED ACCORDING TO EUCLIDEAN DISTANCE***
        #Expand SOAP output from previous iteration - append copy of prediction for last
        #centre to the tensor
        #dim(B, env_n_1, N-1, 64) --> dim(B, env_n_1, N, 64)
        
        #this block executes only if list is not empty
        for env, num_env in zip(chem_env_list_n_1, range(len(chem_env_list_n_1))):
            
            L_index = sorted_species_list.index(env[0])
            H_index = sorted_species_list.index(env[1])

            nodes_L = input_SOAP_pred[L_index]
            nodes_H = input_SOAP_pred[H_index]

            nodes_L_nce = Ul.class_nce_combination(nodes_L, nc_enc, 72)
            nodes_H_nce = Ul.class_nce_combination(nodes_H, nc_enc, 72)
            
            nodes_L_env = Ul.class_env_combination(nodes_L, graph_enc, num_env, 136)
            nodes_H_env = Ul.class_env_combination(nodes_H, graph_enc, num_env, 136)
            
            SOAP_l0 = self.SOAP_0(nodes_L_nce, nodes_H_nce, nodes_L_env, nodes_H_env, nodes_L, nodes_H)
            SOAP_l1 = self.SOAP_1(nodes_L_nce, nodes_H_nce, nodes_L_env, nodes_H_env, nodes_L, nodes_H)
            SOAP_l2 = self.SOAP_2(nodes_L_nce, nodes_H_nce, nodes_L_env, nodes_H_env, nodes_L, nodes_H)
            SOAP_l3 = self.SOAP_3(nodes_L_nce, nodes_H_nce, nodes_L_env, nodes_H_env, nodes_L, nodes_H)
            
            SOAP_env = torch.cat((SOAP_l0, SOAP_l1, SOAP_l2, SOAP_l3), dim=2)
                
            env_dict[env] = SOAP_env
            
        final_env_list = []
        for env in chem_env_list_n:
            final_env_list.append(env_dict[env])
            
        final_SOAP = torch.stack(final_env_list, dim=1)
        
        return final_SOAP