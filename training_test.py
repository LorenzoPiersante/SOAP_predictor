#IMPORTS
import os
import shutil
import pathlib as pl
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

#CUSTOM IMPORTS
import Data_structures as DS
import Train_utils as TU
import SOAP_predictor as SP

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

#Get the Datasets

train_data = TU.MG_dataset("train")
val_data = TU.MG_dataset("val")

len_train = train_data.__len__()
len_val = val_data.__len__()
ratio = int(len_train/len_val) #must be integer multiples!

#Define the model

soap_pred = SP.SOAP_predictor_model().to(my_device)

#Define the optimiser
optimiser = optim.Adam(soap_pred.parameters(), lr = 10**(-5))

epochs = 1

#minibatch size, same ratio in order to have the same number of minibatches
train_minibatch = 10
val_minibatch = int(train_minibatch/ratio)

#used to store historical values for each epoch
hist_training_loss = []
hist_validation_loss = []



for e in range(epochs):
    
    #create randomised access
    train_batches = TU.custom_dataloader(len_train, train_minibatch).create_batches()
    val_batches = TU.custom_dataloader(len_val, val_minibatch).create_batches()
    
    #initial loss over epoch - for historical record
    training_loss = 0
    validation_loss = 0
    
    #batch counter
    B_count = 0
    V_count = 0
    
    for B, V in zip(train_batches, val_batches):
        
        train_loss = torch.tensor(0, dtype=torch.float, device = my_device)
        val_loss = torch.tensor(0, dtype=torch.float, device = my_device)
        
        print("Epoch : " + str(e))
        
        #TRAINING
        print("Training batch : " + str(B_count))
        sample_count = 0
        for b in B: #for sample in minibatch
            
            st = time.time()
            with torch.no_grad():
                #create lables and iter N-1 inputs
                input_molecule = train_data.__getitem__(b)
                #initialise MG object with labels
                input_graph = DS.MoleculeGraph(input_molecule.atoms_list, input_molecule.SOAPs_list)
                #assign iter N-1 inputs
                input_graph.atoms = input_molecule.atoms_list_N_1
                input_graph.species = input_molecule.species
                input_graph.SOAPs = input_molecule.SOAPs_N_1
                input_graph.nodes_by_class = input_molecule.nodes_by_class
                input_graph.init_node_enc = input_molecule.init_node_enc
            et = time.time()
            
            print(str(sample_count) + " Molecule size: " + str(len(input_graph.atom_labels)))
            print("Elapsed time: " + str(et-st))
            
            st = time.time()
            #GO THROUGH LAST ITERATION STEP
            output_graph_N = soap_pred(input_graph, net_state=1)
            
            #GO THROUGH FINALISATION
            output_graph = soap_pred(output_graph_N, net_state=2)
            et = time.time()
            print("Elapsed time: " + str(et-st))
            
            st = time.time()
            #COMPUTE SAMPLE LOSS
            tr_ls = TU.SOAP_loss(output_graph, output_graph_N)
            et = time.time()
            print("Elapsed time: " + str(et-st))

            print("Sample loss : " + "{:.4f}".format(tr_ls.item()))
            
            #ACCUMUlATE LOSS
            train_loss += tr_ls
            
            sample_count += 1
        
        #MEAN LOSS OVER BATCH
        train_loss = train_loss/train_minibatch
        
        B_count += 1
        
        print("Training loss over batch: " + "{:.4f}".format(train_loss.item()))
        
        #MODEL OPTIMISATION
        print("Start BP")
        train_loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        print("Finish BP")
        
        #MODEL CHECKPOINTING
        save_model_name = "soap_pred_weights"+ str(B_count) + ".pth"
        torch.save(soap_pred.state_dict(), save_model_name)
        
        #VALIDATION
        print("Validation batch : " + str(V_count))
        with torch.no_grad():
            sample_count = 0
            for v in V:
                
                #create lables and iter N-1 inputs
                input_molecule = train_data.__getitem__(b)
                #initialise MG object with labels
                input_graph = DS.MoleculeGraph(input_molecule.atoms_list, input_molecule.SOAPs_list)
                #assign iter N-1 inputs
                input_graph.atoms = input_molecule.atoms_list_N_1
                input_graph.species = input_molecule.species
                input_graph.SOAPs = input_molecule.SOAPs_N_1
                input_graph.nodes_by_class = input_molecule.nodes_by_class
                input_graph.init_node_enc = input_molecule.init_node_encan
                
                print(str(sample_count) + " Molecule size: " + str(len(input_graph.atom_labels)))
                
                #GO THROUGH LAST ITERATION STEP
                output_graph_N = soap_pred(input_graph, net_state=1)
            
                #GO THROUGH FINALISATION
                output_graph = soap_pred(output_graph_N, net_state=2)
                
                #COMPUTE SAMPLE LOSS
                val_ls = TU.SOAP_loss(output_graph, output_graph_N)
                
                print("Sample loss : " + "{:.4f}".format(tr_ls.item()))
                
                #ACCUMULATE LOSS
                val_loss += val_ls
                
                sample_count += 1
            
            #MEAN LOSS OVER BATCH
            val_loss = val_loss/val_minibatch
            
            V_count += 1
            
            print("Validation loss over batch: " + "{:.4f}".format(val_loss.item()))
            
            #accumulate epoch values
            training_loss += train_loss.item()
            validation_loss += val_loss.item()
            
        #SAVE EPOCH VALUES
        training_loss = training_loss/(len_train/train_minibatch)
        validation_loss = validation_loss/(len_val/val_minibatch)

        hist_training_loss.append([e, training_loss])
        hist_validation_loss.append([e, validation_loss])
        
        np.savetxt("training_error.txt", hist_training_loss)
        np.savetxt("validation_error.txt", hist_validation_loss)