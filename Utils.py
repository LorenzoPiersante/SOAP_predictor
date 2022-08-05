import torch
import numpy as np

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))


#SOAP predictor

#combine node-centre by class with chem environment aggregated information (graph encoding output)
def class_env_combination(SOAP_nodes_centre_input_class, graph_enc, env_index, new_size):
    '''
    This function prepares the input for the SOAP predictor head for previously classified environments.
    It adds the environment information to the node-centre-class input.
    
    Args:
    - SOAP_nodes_centre_input_class: dim(B, C, C_i, 6), where: C: total number of centres, C_i: number of nodes
    belonging to class i;
    - graph_enc: dim(B, env_num, ***)
    - env_index: index of chem env of interest, ordering is given by tuple order in env list from previous
    iteration;
    - new_size = 6 + ***.
    
    Returns:
    - dim(B, C, C_i, new_size): append to all node-centre entries the relevant env vector.
    '''
    
    graph_enc = torch.unsqueeze(graph_enc, dim=2)
    graph_enc = graph_enc[:, env_index, :, :]
    
    #allocate memory
    nodes_centre_input_class_with_env = torch.zeros((SOAP_nodes_centre_input_class.size()[0], SOAP_nodes_centre_input_class.size()[1],
                                                    SOAP_nodes_centre_input_class.size()[2], new_size), dtype=torch.float, device=my_device)
    
    #assign values
    nodes_centre_input_class_with_env[:, :, :, 0:6] = SOAP_nodes_centre_input_class
    nodes_centre_input_class_with_env[:, :, :, 6:new_size] = torch.unsqueeze(graph_enc, dim=1)
    
    return nodes_centre_input_class_with_env

#SOAP predictor

#combine node-centre by class with global node-centre encoding for each centre
def class_nce_combination(SOAP_nodes_centre_input_class, node_centre_enc, new_size):
    '''
    This function prepares the input for the SOAP predictor head. It combines the node-centre-class input with
    the global node-centre encoding.
    
    Args:
    - SOAP_nodes_centre_input_class: dim(B, C, C_i, 6), where: C: total number of centres, C_i: number of nodes
    belonging to class i;
    - node_centre_enc: dim(B, C, ***), output of global node centre encoding part of the network;
    - new_size = 6 + ***.
    
    Returns:
    - dim(B, C, C_i, new_size): append to all node-centre entries the relevant global nce.
    '''
    
    node_centre_enc = torch.unsqueeze(node_centre_enc, dim=2)
    
    #allocate memory
    nodes_centre_input_class_with_nce = torch.zeros((SOAP_nodes_centre_input_class.size()[0], SOAP_nodes_centre_input_class.size()[1],
                                                    SOAP_nodes_centre_input_class.size()[2], new_size), dtype=torch.float, device=my_device)
    
    #assign values
    nodes_centre_input_class_with_nce[:, :, :, 0:6] = SOAP_nodes_centre_input_class
    nodes_centre_input_class_with_nce[:, :, :, 6:new_size] = node_centre_enc
    
    return nodes_centre_input_class_with_nce