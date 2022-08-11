import torch
from torch import optim
import torch.nn as nn
import numpy as np
import time

import Model_inputs as Mi
import Model_definition as Md

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

#import the model
SOAP_predictor = Md.SOAP_predictor_model().to(my_device)

#define optimiser and loss
optimiser = optim.Adam(SOAP_predictor.parameters(), lr = 5*10**(-5))
mean_sq_loss = nn.MSELoss(reduction="mean")

start_epochs = time.time()

epochs = 1000

B = 100 #batch size
N = 20 #molecle size
alpha = 0.3 #controls the amount of white noise
no_teacher_forcing_prob = 0.7 

#extent of AFM image
lenght_width = 10
height = 2

optimiser.zero_grad()

#save loss:
#rows: number of nodes in graph
#columns: epochs
loss_lists = [[],[],[],[],[],[],[],[],[],[], [],[],[],[],[],[],[],[],[],[]]
mean_loss_list = [] #over all nodes

for e in range(epochs):
    print("\n", "\n")
    print("Epoch: " + str(e))
    
    print("Generate batch of molecules")
    with torch.no_grad():
        #create input
        #atom class buckets
        coordinate_buckets = Mi.create_buckets(lenght_width, height, B, N)[2]
        number_classes = len(coordinate_buckets)
        #generate input order
        history, molecule_input, molecule_string, number_per_class = Mi.create_input_order(number_classes, coordinate_buckets, B, N)
        #re-order points by Euclidean distance
        molecule_input = Mi.Euclid_reorder(molecule_input)
    
    print("Predict SOAP for all subgraphs from 1 to " + str(N) + " atoms per subgraph")
    #feed atoms one by one to the net
    #model prediction on intermediate graph
    #calculate label and loss
    #BP
    SOAP_n_1 = None
    chem_env_n_1 = []
    for n in range(1, N+1):
        
        print("Number nodes: " + str(n))
        print("Create subgraph input and label")
        with torch.no_grad():
            #net inputs
            input_graph_enc, input_node_centre_enc, input_SOAP_pred, chem_env_n = Mi.make_network_inputs(molecule_input, molecule_string, SOAP_n_1 , chem_env_n_1, n, B)
            sorted_species_list = Mi.env_manager(set(molecule_string[0:n])).species
            #create label
            SOAP_label = Mi.make_label(n, molecule_string, molecule_input, B)[0]
        
        print("Model prediction")
        
        st = time.time()
        SOAP_prediction = SOAP_predictor(input_graph_enc, input_node_centre_enc, input_SOAP_pred, chem_env_n, chem_env_n_1, sorted_species_list, SOAP_n_1, n)
        et = time.time()
        print("Elapsed time: ", et-st)
        
        loss = mean_sq_loss(SOAP_prediction, SOAP_label)
        print("Loss: ", loss.item())
        
        print("start BP")
        st = time.time()
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        et = time.time()
        print("end BP")
        print("Elapsed time: ", et-st)
        
        with torch.no_grad():
            #update loss
            loss_lists[n-1].append(loss.item())
            #prepare inputs from previous iteration for next one
            SOAP_n_1 = SOAP_prediction.detach()
            chem_env_n_1 = chem_env_n
            #apply teacher forcing according to its probability
            if np.random.uniform(0, 1) > no_teacher_forcing_prob:
                SOAP_n_1 = Mi.make_label(n, molecule_string, molecule_input, B)[0]
                noise = torch.randn(SOAP_n_1.size(), requires_grad = False, dtype = torch.float, device=my_device)*alpha
                SOAP_n_1 = SOAP_n_1 + noise*SOAP_n_1
                
        
    with torch.no_grad():
        #compute mean loss
        mean_loss = np.sum(np.array(loss_lists)[:, -1])/N
        mean_loss_list.append(mean_loss)
        print("\n")
        print("Mean Loss Epoch {}: ".format(e), mean_loss)
        #save loss and checkpoint model weights
        if (e+1) % 50 == 0:
            np.savetxt("loss.txt", np.array(loss_lists))
            save_model_name = "SOAP_predictor_weights"+ str(e) + ".pth"
            torch.save(SOAP_predictor.state_dict(), save_model_name)
            np.savetxt("mean_loss.txt", np.array(mean_loss_list))
            
end_epochs = time.time()
time_epochs = end_epochs-start_epochs
print("Time for {} epochs: ".format(epochs), time_epochs)