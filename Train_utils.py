# 27-07-2022

# This file contains support functions needed for training the model.

# 28-07-2022

# Modify the loss function to take in both finalised SOAP and the last SOAP prediction of the iteration procedure
# Create custom dataset of MG objetcs
# Create a "dataloader" - randomised access to the files

#IMPORTS
import os
import shutil
import pathlib as pl
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

#CUSTOM IMPORTS
import Data_structures as DS
from torch.utils.data import Dataset

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

# Loss calculator

# This loss function computes a MSE loss using the SOAP predictions of the final MG object and the MG object
# obtained in the last atom-addition iteration

def SOAP_loss(output_graph, last_hist):
    
    mean_sq_loss = nn.MSELoss(reduction="mean")
    
    chem_env = DS.env_manager(output_graph.species).env_tuples()
    N = len(chem_env)
    M = len(output_graph.atoms)
    
    #arrange all SOAP components into a single tensor
    output_SOAP1 = torch.zeros((int(16*M), int(4*N)), requires_grad = False, device = my_device)
    output_SOAP2 = torch.zeros((int(16*M), int(4*N)), requires_grad = False, device = my_device)
    target_SOAP = torch.zeros((int(16*M), int(4*N)), requires_grad = False, device = my_device)
    
    for m in range(M):
        for c, n in zip(chem_env, range(N)):
            atom_start = int(m*16)
            atom_end = int((m+1)*16)
            env_start = int(n*4)
            env_end = int((n+1)*4)
            
            SOAP1 = output_graph.SOAPs[m][c]
            SOAP2 = last_hist.SOAPs[m][c]
            label = output_graph.SOAP_labels[m][c]
            
            output_SOAP1[atom_start:atom_end, env_start:env_end] = SOAP1
            output_SOAP2[atom_start:atom_end, env_start:env_end] = SOAP2
            target_SOAP[atom_start:atom_end, env_start:env_end] = label
    
    #Combine the tensors 
    comb_SOAP = torch.cat((output_SOAP1, output_SOAP2), dim=0)
    comb_target = torch.cat((target_SOAP, target_SOAP), dim=0)
    
    #compute loss
    loss = mean_sq_loss(comb_SOAP, comb_target)
    
    return loss

def SOAP_loss_eval(output_graph, last_hist):
    
    mean_sq_loss = nn.MSELoss(reduction="mean")
    
    chem_env = DS.env_manager(output_graph.species).env_tuples()
    N = len(chem_env)
    M = len(output_graph.atoms)
    
    #arrange all SOAP components into a single tensor
    output_SOAP1 = torch.zeros((int(16*M), int(4*N)), requires_grad = False, device = my_device)
    output_SOAP2 = torch.zeros((int(16*M), int(4*N)), requires_grad = False, device = my_device)
    target_SOAP = torch.zeros((int(16*M), int(4*N)), requires_grad = False, device = my_device)
    
    for m in range(M):
        for c, n in zip(chem_env, range(N)):
            atom_start = int(m*16)
            atom_end = int((m+1)*16)
            env_start = int(n*4)
            env_end = int((n+1)*4)
            
            SOAP1 = output_graph.SOAPs[m][c]
            SOAP2 = last_hist[m][c]
            label = output_graph.SOAP_labels[m][c]
            
            output_SOAP1[atom_start:atom_end, env_start:env_end] = SOAP1
            output_SOAP2[atom_start:atom_end, env_start:env_end] = SOAP2
            target_SOAP[atom_start:atom_end, env_start:env_end] = label
    
    #Combine the tensors 
    comb_SOAP = torch.cat((output_SOAP1, output_SOAP2), dim=0)
    comb_target = torch.cat((target_SOAP, target_SOAP), dim=0)
    
    #compute loss
    loss = mean_sq_loss(comb_SOAP, comb_target)
    
    return loss

# Dataset

class MG_dataset(Dataset):
    '''
    This class retreives the data files from the data directory and produces a custom dataset with
    len and getitem methods.
    
    Initialisation arguments:
    - target_folder: "train" (training files) or "val" (valdation files)
    
    __getitem__ arguments:
    indx : integer (between 0 and len-1 dataset), which is the index attached to the file
    '''
    
    def __init__(self, target_folder):
        self.target_folder = target_folder
        
    def __len__(self):
        path2dir = DS.dir_fetcher(self.target_folder)
        return len(os.listdir(path2dir))
    
    def __getitem__(self, indx):
        file_name = str(indx) + ".xyz"
        path2file = DS.file_fetcher(file_name, self.target_folder)
        #Extract randomised labels
        molecule = DS.xyz_manager(path2file, shuffle = True)
        atoms_list = molecule.atoms_list
        SOAPs_list = molecule.SOAPs_list
        
        return (atoms_list, SOAPs_list)

# Creates a list of lists that are the integers from 0 to len-1 shuffled at random.
# These will be used for randomly accessing the files

class custom_dataloader:
    '''
    Initialisation argumnets:
    size_dataset : number of files in the dataset
    minibatch : number of files in the minibatch
    '''
    def __init__(self, size_dataset, minibatch):
        self.size = size_dataset
        self.minibatch = minibatch
    
    def create_batches(self):
        '''
        Method that creates a list of lists of random integers whose size is given by the requested minibatch
        size.
        '''
        
        ordering = [n for n in range(self.size)]
        random.shuffle(ordering)
        
        N = int(self.size/self.minibatch)
        
        batches = []
        for i in range(N):
            start = int(i*self.minibatch)
            end = int((i+1)*self.minibatch)
            
            minibatch = ordering[start:end]
            
            batches.append(minibatch)
            
        return batches
