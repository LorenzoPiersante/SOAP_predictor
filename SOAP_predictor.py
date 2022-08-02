# 26-07-2022

# Here we combine all the parts into the final test model.

# 29-07-2022

# remove history from model output

#IMPORTS
import os
import shutil
import pathlib as pl

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

#CUSTOM IMPORTS
import Data_structures as DS
import Attention_Mech as AM
import Aggregators as AG
import Graph_encoders as GE

class node_centre_encoding_block(nn.Module):
    '''
    This function creates the nodes-centre encoding for all nodes
    '''
    
    def __init__(self):
        super().__init__()
        self.init_node_centre = AM.self_attn_init(6, 16)
        self.enc1_node_centre = AM.self_attn_enc(22, 64)
        self.aggr_enc_node_centre = AG.gated_aggr_attn(64)
    
    def forward(self, input_graph):
        
        #create nodes-centre encoding list
        node_centre_encodings = []
        #initialisation layer
        for c in input_graph.node_cloud:
            node_centre = self.init_node_centre(input_graph.node_cloud, c)
            node_centre_encodings.append(node_centre)
        #encoding layer 1
        for nce, c, N in zip(node_centre_encodings, input_graph.node_cloud, range(len(node_centre_encodings))):
            node_centre = self.enc1_node_centre(input_graph.node_cloud, c, nce)
            node_centre_encodings[N] = node_centre
        #aggregator layer
        for nce, N in zip(node_centre_encodings, range(len(node_centre_encodings))):
            node_centre = self.aggr_enc_node_centre(nce)
            node_centre_encodings[N] = node_centre
        #result is aggregated-node centre encoding for each node
        #ordering is given by node index in list
        
        return node_centre_encodings

class graph_encoding_block(nn.Module):
    '''
    This function computes the graph encoding for the current iteration step.
    '''
    def __init__(self):
        super().__init__()
        self.hidden_init = GE.hidden_state_init(13, 16, 48)
        self.SOAP_aggr_iter = GE.GRU_graph_enc(64, 64)
        self.hidden_states_aggr = AG.aggr_graph(64, 64)
        self.final_graph_enc = AG.aggr_graph_new_node(64, 64, 64)
        
    def forward(self, input_graph, iteration):
        if iteration:
            #hidden state initialisation
            h_0 = self.hidden_init(input_graph) #each row corresponds toa different atom/node
            #select the hidden state initilisation corresponding to the newly added node
            new_node_h_0 = h_0[-1] #dim = [64]

            #GRU aggregate other hidden state initialisations with previously calculated SOAPs
            #Since iteration = True the h_0 output will present only N-1 nodes
            h_0 = self.SOAP_aggr_iter(input_graph, h_0, iteration) #GRU aggregation is not applied to the first added atom
            h_0 = torch.squeeze(h_0, dim = 0) #remove dummy dimension

            #aggregate SOAPed hidden states into a single graph-wide encoding
            h_G = self.hidden_states_aggr(h_0) #dim = [64]

            #combine graph-wide encoding with new_node encoding to produce final graph wide encoding
            h_G = self.final_graph_enc(h_G, new_node_h_0) #dim = [64]
            #result is graph wide encoding for the current iteration
        else:
            #hidden state initialisation
            h_0 = self.hidden_init(input_graph)
            
            #GRU aggregate hidden state initialisations with previously calculated SOAPs
            #Since iteration = False the h_0 output will present N nodes
            h_0 = self.SOAP_aggr_iter(input_graph, h_0, iteration) #GRU aggregation is not applied to the first added atom
            h_0 = torch.squeeze(h_0, dim = 0) #remove dummy dimension
            
            #aggregate SOAPed hidden states into a single graph-wide encoding
            h_G = self.hidden_states_aggr(h_0) #dim = [64]
        
        return h_G

#This is the encoding that is used as initialisation input of the SOAP predictor head
#It is calculated once for each atom in the SOAP predictor loop and it is not stored
class graph_centre_encoding_block(nn.Module):
    '''
    This function combines a nodes-centre encoding with the graph encoding for the current iteration step.
    It is computed only for the nodes that have been added to the MoleculeGraph.atom_list up until the current
    iteration step.
    '''
    def __init__(self):
        super().__init__()
        self.graph_centre_aggr = AG.GRU_aggr(64)
    
    def forward(self, h_G, nce):
        #combine node-centre encoding with graph-wide encoding
        graph_centre = self.graph_centre_aggr(h_G, nce)
        
        return graph_centre

class SOAP_predictor_head(nn.Module):
    '''
    This function computes a SOAP component from the provided nodes (arranged as lighter, heavier) and the 
    graph-centre encodings.
    '''
    def __init__(self):
        super().__init__()
        self.layer0 = AM.SOAP_dec(70, 64)
        self.aggr0 = AG.gated_aggr_attn(64)
        self.layer1 = AM.SOAP_dec(70, 16)
        self.aggr1 = AG.gated_aggr_attn(16)
        
    def forward(self, L_nodes, H_nodes, centre, graph_centre):
        #layer 0
        x = self.layer0(L_nodes, H_nodes, centre, graph_centre) #output tensor
        x = self.aggr0(x) #aggregate
        #layer 1
        x = self.layer1(L_nodes, H_nodes, centre, x) #output tensor
        SOAP = self.aggr1(x) #aggregate
        
        return SOAP

class SOAP_predictor_model(nn.Module):
    '''
    SOAP predictor model, it is built up from the toolkit available in the Attention_Mech, Graph_encoders and
    Aggregators modules.
    This model primarily acts on the MoleculeGraph object.
    
    Initialisation arguments:
    None
    We just need to intialise the object within the nn.Module calss of torch.
    
    Forward arguments:
    - input_graph : empty Molecule_Graph object, no nodes yet added
    - net_state :
        - 0 : evaluate SOAPs up until N-1
        - 1 : evalluate N
        - 2 : finalise
        - 3 : go thourgh the entire cycle
    
    Returns:
    - MG(final): final MG
    - last_hist: last dictionary of SOAPs computed at the end of the iterative procedure
    '''
    
    def __init__(self):
        super().__init__()
        #here go all the functions that involve learnable parameters
        
        #NODE-CENTRE ENCODING BLOCK
        self.create_node_centre_enc = node_centre_encoding_block()
        
        #GRAPH ENCODING BLOCK - ITERATION
        self.create_graph_enc = graph_encoding_block()
                
        #GRAPH-CENTRE ENCODING
        self.create_graph_centre_enc = graph_centre_encoding_block()
        
        #SOAP COMPONENTS
        self.SOAP_l0 = SOAP_predictor_head()
        self.SOAP_l1 = SOAP_predictor_head()
        self.SOAP_l2 = SOAP_predictor_head()
        self.SOAP_l3 = SOAP_predictor_head()
        
    def forward(self, input_graph, net_state):
        
        #NODE-CENTRE ENCODING
        node_centre_encodings = self.create_node_centre_enc(input_graph)
        
        if net_state == 0:
            
            #ITERATIVE PROCESS IN WHICH ATOMS ARE ADDED ONE AT A TIME
            #STOP WHEN N-1 ATOMS HAVE BEEN ADDED
            for atom in input_graph.atom_labels[0 : len(input_graph.atom_labels)-1]:

                #add a new atom
                input_graph.add_atom(atom)

                #GRAPH-ENCODING
                h_G = self.create_graph_enc(input_graph, True)

                #generate current chem environment tuples
                chem_env = DS.env_manager(input_graph.species).env_tuples()

                for a, n in zip(input_graph.atoms, range(len(input_graph.atoms))):
                    #extract current centre
                    centre = a.position
                    #compute graph-nodes-centres encoding for current centre
                    nce = node_centre_encodings[n]
                    graph_centre = self.create_graph_centre_enc(h_G, nce)

                    for env in chem_env:
                        #extract nodes corresponding to light and heavy species
                        L_symb = env[0]
                        H_symb = env[1]
                        L_nodes = input_graph.nodes_by_class[L_symb]
                        H_nodes = input_graph.nodes_by_class[H_symb]

                        #calculate different SOAP_components
                        SOAP_l0 = self.SOAP_l0(L_nodes, H_nodes, centre, graph_centre)
                        SOAP_l1 = self.SOAP_l1(L_nodes, H_nodes, centre, graph_centre)
                        SOAP_l2 = self.SOAP_l2(L_nodes, H_nodes, centre, graph_centre)
                        SOAP_l3 = self.SOAP_l3(L_nodes, H_nodes, centre, graph_centre)

                        #update the SOAP components
                        input_graph.update_SOAP(a, env, 0, SOAP_l0)
                        input_graph.update_SOAP(a, env, 1, SOAP_l1)
                        input_graph.update_SOAP(a, env, 2, SOAP_l2)
                        input_graph.update_SOAP(a, env, 3, SOAP_l3)
                        
            return input_graph
        
        elif net_state == 1:
            
            #add last atom
            input_graph.add_atom(input_graph.atom_labels[-1])

            #GRAPH-ENCODING
            h_G = self.create_graph_enc(input_graph, True)

            #generate current chem environment tuples
            chem_env = DS.env_manager(input_graph.species).env_tuples()

            for a, n in zip(input_graph.atoms, range(len(input_graph.atoms))):
                #extract current centre
                centre = a.position
                #compute graph-nodes-centres encoding for current centre
                nce = node_centre_encodings[n]
                graph_centre = self.create_graph_centre_enc(h_G, nce)

                for env in chem_env:
                    #extract nodes corresponding to light and heavy species
                    L_symb = env[0]
                    H_symb = env[1]
                    L_nodes = input_graph.nodes_by_class[L_symb]
                    H_nodes = input_graph.nodes_by_class[H_symb]

                    #calculate different SOAP_components
                    SOAP_l0 = self.SOAP_l0(L_nodes, H_nodes, centre, graph_centre)
                    SOAP_l1 = self.SOAP_l1(L_nodes, H_nodes, centre, graph_centre)
                    SOAP_l2 = self.SOAP_l2(L_nodes, H_nodes, centre, graph_centre)
                    SOAP_l3 = self.SOAP_l3(L_nodes, H_nodes, centre, graph_centre)

                    #update the SOAP components
                    input_graph.update_SOAP(a, env, 0, SOAP_l0)
                    input_graph.update_SOAP(a, env, 1, SOAP_l1)
                    input_graph.update_SOAP(a, env, 2, SOAP_l2)
                    input_graph.update_SOAP(a, env, 3, SOAP_l3)
                
            return input_graph
        
        elif net_state == 2:
            
            #FINAL SOAP EVALUATION
            #Finalise the SOAP inputs for graph encoding
            input_graph.finalise_SOAP()        

            #GRAPH-ENCODING
            h_G = self.create_graph_enc(input_graph, False)

            #generate current chem environment tuples
            chem_env = DS.env_manager(input_graph.species).env_tuples()

            for a, n in zip(input_graph.atoms, range(len(input_graph.atoms))):
                #extract current centre
                centre = a.position
                #compute graph-nodes-centres encoding for current centre
                nce = node_centre_encodings[n]
                graph_centre = self.create_graph_centre_enc(h_G, nce)

                for env in chem_env:
                    #extract nodes corresponding to light and heavy species
                    L_symb = env[0]
                    H_symb = env[1]
                    L_nodes = input_graph.nodes_by_class[L_symb]
                    H_nodes = input_graph.nodes_by_class[H_symb]

                    #calculate different SOAP_components
                    SOAP_l0 = self.SOAP_l0(L_nodes, H_nodes, centre, graph_centre)
                    SOAP_l1 = self.SOAP_l1(L_nodes, H_nodes, centre, graph_centre)
                    SOAP_l2 = self.SOAP_l2(L_nodes, H_nodes, centre, graph_centre)
                    SOAP_l3 = self.SOAP_l3(L_nodes, H_nodes, centre, graph_centre)

                    #update the SOAP components
                    input_graph.update_SOAP(a, env, 0, SOAP_l0)
                    input_graph.update_SOAP(a, env, 1, SOAP_l1)
                    input_graph.update_SOAP(a, env, 2, SOAP_l2)
                    input_graph.update_SOAP(a, env, 3, SOAP_l3)

            return input_graph
        
        elif net_state == 3:
            
            #NODE-CENTRE ENCODING
            node_centre_encodings = self.create_node_centre_enc(input_graph)

            #ITERATIVE PROCESS IN WHICH ATOMS ARE ADDED ONE AT A TIME
            for atom in input_graph.atom_labels:

                #add a new atom
                input_graph.add_atom(atom)

                #GRAPH-ENCODING
                h_G = self.create_graph_enc(input_graph, True)

                #generate current chem environment tuples
                chem_env = DS.env_manager(input_graph.species).env_tuples()

                for a, n in zip(input_graph.atoms, range(len(input_graph.atoms))):
                    #extract current centre
                    centre = a.position
                    #compute graph-nodes-centres encoding for current centre
                    nce = node_centre_encodings[n]
                    graph_centre = self.create_graph_centre_enc(h_G, nce)

                    for env in chem_env:
                        #extract nodes corresponding to light and heavy species
                        L_symb = env[0]
                        H_symb = env[1]
                        L_nodes = input_graph.nodes_by_class[L_symb]
                        H_nodes = input_graph.nodes_by_class[H_symb]

                        #calculate different SOAP_components
                        SOAP_l0 = self.SOAP_l0(L_nodes, H_nodes, centre, graph_centre)
                        SOAP_l1 = self.SOAP_l1(L_nodes, H_nodes, centre, graph_centre)
                        SOAP_l2 = self.SOAP_l2(L_nodes, H_nodes, centre, graph_centre)
                        SOAP_l3 = self.SOAP_l3(L_nodes, H_nodes, centre, graph_centre)

                        #update the SOAP components
                        input_graph.update_SOAP(a, env, 0, SOAP_l0)
                        input_graph.update_SOAP(a, env, 1, SOAP_l1)
                        input_graph.update_SOAP(a, env, 2, SOAP_l2)
                        input_graph.update_SOAP(a, env, 3, SOAP_l3)

            last_iteration = input_graph.SOAPs

            #FINAL SOAP EVALUATION

            #Finalise the SOAP inputs for graph encoding
            input_graph.finalise_SOAP()        

            #GRAPH-ENCODING
            h_G = self.create_graph_enc(input_graph, False)

            #generate current chem environment tuples
            chem_env = DS.env_manager(input_graph.species).env_tuples()

            for a, n in zip(input_graph.atoms, range(len(input_graph.atoms))):
                #extract current centre
                centre = a.position
                #compute graph-nodes-centres encoding for current centre
                nce = node_centre_encodings[n]
                graph_centre = self.create_graph_centre_enc(h_G, nce)

                for env in chem_env:
                    #extract nodes corresponding to light and heavy species
                    L_symb = env[0]
                    H_symb = env[1]
                    L_nodes = input_graph.nodes_by_class[L_symb]
                    H_nodes = input_graph.nodes_by_class[H_symb]

                    #calculate different SOAP_components
                    SOAP_l0 = self.SOAP_l0(L_nodes, H_nodes, centre, graph_centre)
                    SOAP_l1 = self.SOAP_l1(L_nodes, H_nodes, centre, graph_centre)
                    SOAP_l2 = self.SOAP_l2(L_nodes, H_nodes, centre, graph_centre)
                    SOAP_l3 = self.SOAP_l3(L_nodes, H_nodes, centre, graph_centre)

                    #update the SOAP components
                    input_graph.update_SOAP(a, env, 0, SOAP_l0)
                    input_graph.update_SOAP(a, env, 1, SOAP_l1)
                    input_graph.update_SOAP(a, env, 2, SOAP_l2)
                    input_graph.update_SOAP(a, env, 3, SOAP_l3)
                    
            return input_graph, last_iteration