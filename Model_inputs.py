#02-08-2022
#This file contains the class that creates the input batched molecules of the desired number of atoms.
#This tensor defines the order in which nodes are supplied to the model.

#The "molecules" used for traing will have either 3 or 2 classes.

#03-08-2022
#Add distance to input of nce and SOAP predictor input 
#Add angle to input of nce and SOAP predictor input

import torch
import numpy as np
import random
import ase
import dscribe
from dscribe.descriptors import SOAP

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

#THE FUNCTIONS BELOW ARE NEEDED TO GENERATE THE BATCHED INPUT OF ATOM LOCATIONS BELONGING TO 2 OR 3 DIFFERENT CLASSES

#Hyper-parameters
r_cut = 4
n_max = 4
l_max = 3

SS_l_component = 10
IS_l_component = 16
SOAP_env_vector = 64

#General purpose dictoniaries

#atom dictionary
atomic_dict = {
    "H":1, 
    "C":6,
    "O":8
}

#reverse atom dictionary
reverse_atomic_dict = {
    1: "H",
    6: "C",
    8: "O"
}

#generate random number of atoms per class, they will add up to the molecule size
def random_classes(molecule_size):
    '''
    Args:
    - molecule size: number of atoms in the molecule
    Returns:
    - tuple of number of atoms in given class (class type = tuple index)
    '''
    samples = sorted(random.sample(range(molecule_size), 2)) #sample two random numbers without replacement
    class_0 = samples[0]
    class_1 = samples[1]-samples[0]
    class_2 = molecule_size - samples[1]
    classes = [class_0, class_1, class_2]
    random.shuffle(classes)
    class_0 = classes[0]
    class_1 = classes[1]
    class_2 = classes[2]
    return class_0, class_1, class_2

#generate a random coordinate
def random_coordinate(lenght_width, height):
    '''
    Args:
    - length_width: 2D extent of the molecule, a square 2*length_width Å x 2*length_width Å from -length_width 
    to length_width in the two directions
    - height: extent of the molecule in the z-direction, from -height to height
    Returns:
    - [[x, y, z]]: numpy.ndarray(1, 3), coordinates are uniformly distributed
    '''
    xy = np.random.uniform(-lenght_width, lenght_width, size = (1, 2))
    z = np.random.uniform(-height, height, size = (1, 1))
    coordinate = np.concatenate((xy, z), axis=1)
    return coordinate

#generate coordinates for every atom in each class
def create_buckets(lenght_width, height, B, N):
    '''
    Args:
    - lenght_width
    - height
    - B: number of molecules in the batch
    - N: number of atoms in the molecule
    Retrurns:
    - classes: 3-tuple, each element is the number of atoms in the given class
    - zero_class: class that has 0 atoms either None, 0, 1, 2
    - coordinate_buckets: list of at most 3 tensors of dimension dim(batch size, number of atoms in class, 3-coordinate)
    '''
    
    classes = random_classes(N)
    print("Classes allocation :", classes)

    coordinate_buckets = []

    zero_class = None
    if classes[0] == 0:
        pass
    else:
        class_0_bucket = torch.zeros((B, classes[0], 3), requires_grad=False, device=my_device)
        for i in range(B):
            for j in range(classes[0]):
                class_0_bucket[i, j, :] = torch.tensor(random_coordinate(lenght_width, height), requires_grad = False, dtype = torch.float, device=my_device)
        coordinate_buckets.append(class_0_bucket)

    if classes[1] == 0:
        #do nothing
        zero_class = 1
    else:
        class_1_bucket = torch.zeros((B, classes[1], 3),  requires_grad=False, device=my_device)
        for i in range(B):
            for j in range(classes[1]):
                class_1_bucket[i, j, :] = torch.tensor(random_coordinate(lenght_width, height), requires_grad = False, dtype = torch.float, device=my_device)
        coordinate_buckets.append(class_1_bucket)

    if classes[2] == 0:
        #do nothing
        zero_class = 2
    else:
        class_2_bucket = torch.zeros((B, classes[2], 3), requires_grad=False, device=my_device)
        for i in range(B):
            for j in range(classes[2]):
                class_2_bucket[i, j, :] = torch.tensor(random_coordinate(lenght_width, height), requires_grad = False, dtype = torch.float, device=my_device)
        coordinate_buckets.append(class_2_bucket)

    
    print("Class with zero atoms: ", zero_class)
    
    return classes, zero_class, coordinate_buckets

#generate molecule input that is:
#- history: list of len = number of atoms in molecule of 0, 1 or 0, 1, 2 which defines in what order to access the buckets
#the number of 0s, 1s and 2s is determined by the number of atoms per class
#- tensor of dimension dim(batch size, number of molecules in batch, 3)

def create_input_order(number_classes, coordinate_buckets, B, N):
    '''
    Args:
    - number_classes: number of different atomic classes in the molecule (2 or 3)
    - coordinate_buckets
    - B: batch size
    - N: number of molecules
    Returns:
    - history: order in which buckets are accessed
    - molecule_input: tensor of dimension dim(B, N, 3), which, along with the history, defines
    the order in which new atoms coordinates are added to the system
    - molecule_string: input string for ase.Atoms object, it generates a string of H (0 class), C (1 class), O
    (2 class) that is used to produce the label SOAP (SOAP.create requires an ase molecule input)
    '''
    class_dict = {
        0:"H",
        1:"C",
        2:"O"
    }
    
    if number_classes == 3:
        counter_0 = 0
        counter_1 = 0
        counter_2 = 0
        counter = [counter_0, counter_1, counter_2]
        N_0 = coordinate_buckets[0].size()[1]
        N_1 = coordinate_buckets[1].size()[1]
        N_2 = coordinate_buckets[2].size()[1]
        N_i = [N_0, N_1, N_2]
    else:
        counter_0 = 0
        counter_1 = 0
        counter = [counter_0, counter_1]
        N_0 = coordinate_buckets[0].size()[1]
        N_1 = coordinate_buckets[1].size()[1]
        N_i = [N_0, N_1]

    history = []
    molecule_input = torch.zeros((B, N, 3), requires_grad=False, device="cpu")
    molecule_string = ""

    for n in range(N):

        if number_classes == 3:
            selection = [0, 1, 2]
        else:
            selection = [0, 1]

        pick_class = np.random.randint(number_classes)

        if counter[pick_class] == int(N_i[pick_class]):
            restricted_selection = set(selection) - set([pick_class])
            pick_class = random.sample(restricted_selection, 1)[0]

        if counter[pick_class] == int(N_i[pick_class]):
            restricted_selection = restricted_selection - set([pick_class])
            pick_class = list(restricted_selection)[0]
            
        current_bucket = coordinate_buckets[pick_class]
        new_atoms = current_bucket[:, counter[pick_class], :]
        molecule_input[:, n, :] = new_atoms

        counter[pick_class] += 1
        history.append(pick_class)
        molecule_string = molecule_string + class_dict[pick_class]
    
    print("Final counters: ", counter)
    
    return history, molecule_input, molecule_string, counter

#Starting from 0-point, reorder according to Euclidean distance
def Euclid_reorder(molecule_input):
    '''
    This function reorders the points in the cloud of points (molecules) of the batched molecule input 
    according to Euclidean distance.
    It starts from the first point in the original point cloud tensor, it computes the distance to all other
    points and selects the closest one as next entry in the new point cloud tensor.
    This point then becomes the new reference point, and the successive closest point is evaluated with
    respect to it.
    The process is repeated until the list of new reference points is exhausted.
    '''
    
    #split batch
    molecules = list(torch.split(molecule_input, 1, dim=0))
    #list for ordered molecules
    ordered_molecules = []

    for m in molecules:
        #remove dummy dimension
        m = torch.squeeze(m, dim=0)
        #split into points
        points = list(torch.split(m, 1, dim=0))
        #remove dummy dimensions from points
        for p in range(len(points)):
            points[p] = torch.squeeze(points[p], dim=0)
        
        #list of ordered points 
        ordered_points = []
        #start point
        p_ref = points[0]
        
        for n in range(m.size()[0]):
            #list of distances
            distance_list = []
            for q in range(len(points)):
                #calculate distance of all points from reference point
                distance = torch.sum(torch.sqrt((points[q]-p_ref)**2)).item()
                distance_list.append(distance)
            #find index of point at minimum distance from reference
            index_min = distance_list.index(min(distance_list))
            #make it new reference
            p_ref = points.pop(index_min)
            #add it to list of ordered points
            ordered_points.append(p_ref)
        
        #create tensor for ordered molecule
        ordered_molecule = torch.stack(ordered_points, dim=0)
        #add it to list of ordered molecules
        ordered_molecules.append(ordered_molecule)
    
    #create new batched input
    new_molecule_input = torch.stack(ordered_molecules, dim=0)
    
    return new_molecule_input

#Since the input is decomposed into lists, and then recomposed, this is the function that takes the longest
#to execute
#Space complexity: B*N*N ~ N^3

#ENVIRONMENT GENERATOR CLASS
#These functions are needed to create the environment tuples
#They are overly general so that they can be applied to the graph building iterative process
class env_manager():
    '''
    Initialisation args:
    - species: set of all the atomic sts(symbols) in the molecule
    '''
    
    def __init__(self, species):
        
        #This function sorts the input species according to atomic number
        def sorter(species):
            
            species_list = [atomic_dict[j] for j in species] #list of atomic numbers as given by the input
            species_list.sort() #sort them
            sorted_species_list = [reverse_atomic_dict[j] for j in species_list] #convert back to symbols
            
            return sorted_species_list
            
        self.species = sorter(species)
    
    #This function creates a list of environment tuples according to the standard SOAP ordering
    def env_tuples(self):
        
        envs = []
        
        n = 0
        for i in self.species:
            a = i
            
            for j in range(n, len(self.species)):
                b = self.species[j]
                SOAP_env = (a, b)
                envs.append(SOAP_env)
            
            n += 1
            
        return envs

#SELF-SOAP RESHAPING FUNCTION
#In the same-species SOAP, the soap.create() method generates a vector with only non-repeated elements, thus we
#need a function to reshape it into the complete symmetric power vector
#The general SOAP for a given chemical env and l-component is arrange as: 
#for n from 1 to n_max
#   for n' from 1 to n_max
#if the chemical environment involves like species, symmetric components are stored once only
#so taking a (H, H) env the components will be 1-1, 1-2, 1-3, 1-4, 2-2, 2-3, 2-4, 3-3, 3-4, 4-4
#(this is the output of soap.create() where soap = SOAP object of dscribe library of molecular descriptors)
#the target vector is then:
#1-1, 1-2, 1-3, 1-4, 1-2, 2-2, 2-3, 2-4, 1-3, 2-3, 3-3, 3-4, 1-4, 2-4, 3-4, 4-4
#this will ensure that the label SOAP power vectors all have the same size
def SSOAP_reshaping(SS_vector):
    '''
    Casts the self-SOAP vector to the same shape of the inter-SOAP vector.
    It adds the symmetric components that are otherwise missing.
    '''
    num_atoms, num_entries = np.shape(SS_vector)
    #split the vector into the l components
    l_components = np.split(SS_vector, int(l_max+1), axis=1)
    
    #identify the n-value split locations
    split_locations = []
    loc = 0
    for n in reversed(range(1, n_max+1)):
        loc += n
        split_locations.append(loc)
    
    #reshape all the components one at a time
    for l in range(l_max+1):
        #divide the l-component according to n
        current_l = np.split(l_components[l], split_locations, axis = 1)
        for i in range(1, n_max):
            for j in reversed(range(i)):
                #add the missing nn' values to the given n
                current_l[i] = np.concatenate((current_l[j][:, i].reshape((num_atoms, 1)), current_l[i]), axis=1)
        
        #reform the l-component
        reshaped_l_component = np.concatenate(current_l, axis=1)
        #add it to original list
        l_components[l] = reshaped_l_component
    
    #generate the new SS_vector
    new_SS_vector = np.concatenate(l_components, axis=1)
    
    return new_SS_vector

#interim_size: size of the output
#iteration: size of the output, goes from 1 to num atoms
#case iteration = 1 is a special case, the SOAP_n_1 input of make_network_inputs() is not used in this case

#FUNCTION TO GENERATE THE LABELS - labels for the current iteration 
#number classes must be given by number of classes seen by the system this far

def make_label(interim_size, molecule_string, molecule_input, B):
    '''
    This function generates the batched SOAP labels for the current iteration, that is the
    interim molecule, the intermediate SOAP at iteration n.
    This will be the learning target.
    
    Args:
    - interim_size: number of atoms classified at iteration n
    - molecule_string: string of H, C, O for the input molecule
    - molecule_input: tensor of dim(B, N, 3), which contains the coordinates of all atoms
    for all the molecules
    - B: number of molecules in the batch
    
    Returns:
    - SOAP_labels: tensor of dim(B, num_env, interim_size, 64), that is the SOAP env vectors for all atoms in the
    molecules in the batch
    - chem_envs: chemical environments present in the current molecule
    '''
    
    #interim ase string
    interim_string = molecule_string[0:interim_size]
    
    #species in molecule
    species_in_mol = set(interim_string)
    
    #create environment tuples
    chem_envs = env_manager(species_in_mol).env_tuples()
    
    #instantiate SOAP
    soap = SOAP(
                species=species_in_mol,
                periodic=False,
                #the one below are hyper parameters
                rcut=r_cut,
                nmax=n_max,
                lmax=l_max
    )

    #create dictionary access to SOAP components
    SOAP_dict = {c:soap.get_location(c) for c in chem_envs}
    
    print(chem_envs)
    #print(SOAP_dict)
    
    #allocate memory for batched label
    SOAP_labels = torch.zeros((B, len(chem_envs), interim_size, 64), requires_grad=False, device=my_device)
    
    #generate list of samples for entire batch
    samples = [ase.Atoms(interim_string, np.array(molecule_input[b, 0:interim_size, :])) for b in range(B)]
    #gnerate SOAP labels, list of SOAPs for each saple
    samples_labels = soap.create(samples, n_jobs = 4)
    #samples_labels = np.split(samples_labels, B, axis=0) #if error is risen, remove this line
        
    #assing SOAP labels to batched label
    for b in range(B):
        SOAP_vector = samples_labels[b]
        
        if len(chem_envs)==6:
            HH_vect = torch.tensor(SSOAP_reshaping(SOAP_vector[:, SOAP_dict[chem_envs[0]]]), requires_grad = False, dtype=torch.float, device=my_device)
            HC_vect = torch.tensor(SOAP_vector[:, SOAP_dict[chem_envs[1]]], requires_grad = False, dtype=torch.float, device=my_device)
            HO_vect = torch.tensor(SOAP_vector[:, SOAP_dict[chem_envs[2]]], requires_grad = False, dtype=torch.float, device=my_device)
            CC_vect = torch.tensor(SSOAP_reshaping(SOAP_vector[:, SOAP_dict[chem_envs[3]]]), requires_grad = False, dtype=torch.float, device=my_device)
            CO_vect = torch.tensor(SOAP_vector[:, SOAP_dict[chem_envs[4]]], requires_grad = False, dtype=torch.float, device=my_device)
            OO_vect = torch.tensor(SSOAP_reshaping(SOAP_vector[:, SOAP_dict[chem_envs[5]]]), requires_grad = False, dtype=torch.float, device=my_device)

            SOAP_labels[b, 0, :, :] = HH_vect
            SOAP_labels[b, 1, :, :] = HC_vect
            SOAP_labels[b, 2, :, :] = HO_vect
            SOAP_labels[b, 3, :, :] = CC_vect
            SOAP_labels[b, 4, :, :] = CO_vect
            SOAP_labels[b, 5, :, :] = OO_vect
        elif len(chem_envs)==3:
            vect_0 = torch.tensor(SSOAP_reshaping(SOAP_vector[:, SOAP_dict[chem_envs[0]]]), requires_grad = False, dtype=torch.float, device=my_device)
            vect_1 = torch.tensor(SOAP_vector[:, SOAP_dict[chem_envs[1]]], requires_grad = False, dtype=torch.float, device=my_device)
            vect_2 = torch.tensor(SSOAP_reshaping(SOAP_vector[:, SOAP_dict[chem_envs[2]]]), requires_grad = False, dtype=torch.float, device=my_device)

            SOAP_labels[b, 0, :, :] = vect_0
            SOAP_labels[b, 1, :, :] = vect_1
            SOAP_labels[b, 2, :, :] = vect_2
        elif len(chem_envs)==1:
            vect_0 = torch.tensor(SSOAP_reshaping(SOAP_vector[:, SOAP_dict[chem_envs[0]]]), requires_grad = False, dtype=torch.float, device=my_device)
            
            SOAP_labels[b, 0, :, :] = vect_0
            
    return SOAP_labels, chem_envs

#generate inputs for network

def make_network_inputs(molecule_input, molecule_string, SOAP_outputs_n_1, chem_envs_n_1, iteration, B):
    '''
    This function produces the bathched inputs for the various parts of the network.
    Args:
    - molecule_input
    - molecule_string
    - SOAP_outputs_n_1: tensor of dim(B, env, iteration-1, 64), SOAP outputs at the previous iteration
    - chem_envs_n_1: list of chemical environments at the previous iteration
    - iteration: iteration number
    - B: number of molecules in the batch
    Returns:
    - input_graph_enc: tensor of dim(B, env, iteration-1, pos+SOAP), input of graph encoding part of the net
    - input_node_centre_enc: tensor of dim(B, iteration, iteration, 8), input of global node-centre encoding
    part of the net
    - input_SOAP_pred: list of at most 3 tensors of dim(B, iteration, N_i, 8), they are the inputs of the SOAP
    predictor heads
    - new_chem_env: list of new chemical environments
    
    The entries in the input_node_centre_enc and input_SOAP_pred are: [position_P, centre, distance, angle P-C-Origin]
    [3, 3, 1, 1]
    '''
    #INPUT 1
    if iteration == 1:
        #This input is not used when the first atom is added
        input_graph_enc = None
    else:
        #generate graph-encoding input
        #extract locations at iteration n-1
        nodes_graph_info = molecule_input[:, 0:iteration-1, :]
        #allocate memory for chem_env-position 
        positions_graph_enc = torch.zeros((B, len(chem_envs_n_1), iteration-1, 3), requires_grad = False, dtype=torch.float, device=my_device)
        for c in range(len(chem_envs_n_1)):
            positions_graph_enc[:, c, :, :] = nodes_graph_info

        #concatenate previous SOAP output and positions
        input_graph_enc = torch.cat((positions_graph_enc, SOAP_outputs_n_1), dim=3)
    
    #FIRSTLY ALLOCATE MEMORY FOR INPUTS 2 AND 3
    
    ####################################
    #INPUT 2
    #generate node-centre encoding input
    #extract locations at iteration n
    nodes_centre_info = molecule_input[:, 0:iteration, :]
    #allocate memory for input
    input_node_centre_enc = torch.zeros((B, iteration, iteration, 8), requires_grad = False, dtype=torch.float, device=my_device)
    ####################################
    
    ####################################
    #INPUT 3
    #generate node-centre encoding for each class
    #returned as a list, list order is determied by Z of the species
    interim_string = molecule_string[0:iteration]
    
    #species in current molecule
    species_in_mol = set(interim_string)
    #sort them according to Z
    sorted_species = env_manager(species_in_mol).species
    #produce new env list
    new_chem_env = env_manager(species_in_mol).env_tuples()
    #produce list of class input history
    input_history = list(interim_string)
    
    #generate the appropriate SOAP predictor input according to the 
    if len(species_in_mol) == 1:
        num0 = len(input_history)
        input_SOAP_pred = [torch.zeros((B, num0, num0, 8), requires_grad = False, dtype=torch.float, device=my_device)]
        count0 = 0
        counter = [count0]
    elif len(species_in_mol) == 2:
        num0 = input_history.count(sorted_species[0])
        num1 = input_history.count(sorted_species[1])
        input_SOAP_pred = [torch.zeros((B,iteration, num0, 8), requires_grad = False, dtype=torch.float, device=my_device), 
                         torch.zeros((B, iteration, num1, 8), requires_grad = False, dtype=torch.float, device=my_device)]
        count0 = 0
        count1 = 0
        counter = [count0, count1]
    elif len(species_in_mol) == 3:
        num0 = input_history.count(sorted_species[0])
        num1 = input_history.count(sorted_species[1])
        num2 = input_history.count(sorted_species[2])
        input_SOAP_pred = [torch.zeros((B, iteration, num0, 8), requires_grad = False, dtype=torch.float, device=my_device), 
                         torch.zeros((B, iteration, num1, 8), requires_grad = False, dtype=torch.float, device=my_device),
                          torch.zeros((B, iteration, num2, 8), requires_grad = False, dtype=torch.float, device=my_device)]
        count0 = 0
        count1 = 0
        count2 = 0
        counter = [count0, count1, count2]
    ####################################
    
    #assign entries to tensors
    for c, i in zip(input_history, range(iteration)):
        #produce node-centre combination of correct shape
        nodes_centre = torch.zeros((B, iteration, 8), requires_grad = False, dtype=torch.float, device=my_device)
        centre = nodes_centre_info[:, i, :]
        nodes_centre[:, :, 0:3] = nodes_centre_info
        for j in range(iteration):
            nodes_centre[:, j, 3:6] = centre
            
            #add euclidean distance 
            P = torch.clone(centre).to(my_device)
            Q = nodes_centre[:, j, 0:3].to(my_device)
            distance = torch.sqrt(torch.sum((Q-P)**2, dim=-1))
            
            #assign distance
            nodes_centre[:, j, 6] = distance
            
            #add angle
            #need vectors wrt to centre
            r_1 = Q - P
            r_2 = -P
            distance1 = torch.clone(distance)
            distance2 = torch.sqrt(torch.sum((r_2)**2, dim=-1))
            if Q[0, 0] == P[0, 0] and Q[0, 1] == P[0, 1] and Q[0, 2] == P[0, 2]:
                normalised_inner_product = torch.sum((r_1*r_2), dim=-1)
                angle = torch.acos(normalised_inner_product)*(180/3.14159265359)
            else:
                normalised_inner_product = torch.sum((r_1*r_2), dim=-1)/(distance1*distance2)
                
                s1 = torch.tensor(1, requires_grad = False, dtype=torch.float, device=my_device)
                s2 = torch.tensor(-1, requires_grad = False, dtype=torch.float, device=my_device)
                #ensure that all entries remain within interval definition of arcos argument
                normalised_inner_product = torch.where(normalised_inner_product<1, normalised_inner_product, s1)
                normalised_inner_product = torch.where(normalised_inner_product>-1, normalised_inner_product, s2)
                
                angle = torch.acos(normalised_inner_product)*(180/3.14159265359)
            
            
            #assign angle
            nodes_centre[:, j, -1] = angle
            
        #assign to global node_centre enc
        input_node_centre_enc[:, i, :, :] = nodes_centre
        
        #turn chem symbol info into index - assign to SOAP input
        index_class = sorted_species.index(input_history[i])
        loc = counter[index_class]
        input_SOAP_pred[index_class][:, :, loc, :] = nodes_centre
        counter[index_class] += 1
                
    return input_graph_enc, input_node_centre_enc, input_SOAP_pred, new_chem_env