#IMPORTS
import os
import shutil
import pathlib as pl

import numpy as np
import pandas as pd

import ase
import ase.io
import dscribe
from dscribe.descriptors import SOAP

#General purpose dictoniaries

#atom dictionary
atomic_dict = {
    "H":1,
    "C":6,
    "N":7,
    "O":8,
    "F":9,
    "Si":14,
    "P":15,
    "S":16,
    "Cl":17,
    "Br":35
}

#reverse atom dictionary
reverse_atomic_dict = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br"
}

#Initialise SOAP PARAMETERS- the SOAP parameters are hyperparameters that will be used throughout the model

l_max = 3 #starts counting from 0
n_max = 4 #starts counting from 1
r_cut = 4 #std dev of Gaussian distribution

#number or SOAP entries for each l
size_l_SS = 10 #self-SOAP
size_l_IS = 16 #inter-SOAP


# SUPPORT FUNCTIONS AND CLASSES

# PRODUCE PATH TO FILE
# This function returns the path to the desired file

def file_fetcher(file_name, target_folder):
    '''
    Args:
    - file_name: name of the file, NOTE IT MUST ESIST IN target_folder
    - target_folder: select if you want the train data folder or the validation data folder
    '''

    local_path = os.path.abspath("./")  # get absolute path to current directory
    parent_path = pl.PurePath(local_path).parent  # move to parent directory

    # new directories
    molecule_folder = "xyz_molecules/SOAP_data"
    # available target_folder options
    folders = {
        "train": "SOAP_train",
        "val": "SOAP_val"
    }

    path2file = parent_path.joinpath(molecule_folder, folders[target_folder], file_name)

    return path2file


# ENVIRONMENT GENERATOR CLASSES
# These functions are needed to create the environment tuples
# They are overly general so that they can be applied to the graph building iterative process

class env_manager():
    '''
    Initialisation args:
    - species: set of all the atomic sts(symbols) in the molecule
    '''

    def __init__(self, species):

        # This function sorts the input species according to atomic number
        def sorter(species):
            species_list = [atomic_dict[j] for j in species]  # list of atomic numbers as given by the input
            species_list.sort()  # sort them
            sorted_species_list = [reverse_atomic_dict[j] for j in species_list]  # convert back to symbols

            return sorted_species_list

        self.species = sorter(species)

    # This function creates a list of environment tuples according to the standard SOAP ordering
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

#THE CLASSES and OBJECT BELOW ARE DESIGNED TO BE THE NETWORKS DATASTRUCTURES

class Atom:
    '''
    Simple object that stores atom location and atom class.

    Args:
    - atom position [x, y, z]
    - species

    to get classes or position use .attributes
    '''

    def __init__(self, position, species):
        self.position = position
        self.atom_class = species


class xyz_manager:
    '''
    This class contains a set of methods to operate on the xyz file

    Args:
    - xyz_file: .xyz file path, ASE compatible .xyz format
    - shuffle: bool, if True, a random shuffle of the final lists is performed, match between indeces in atoms_list
    and SOAPs_list is maintained
    '''

    def __init__(self, xyz_file, shuffle):
        import random

        # just store the file path
        self.file = xyz_file

        def atoms_list(self):
            '''
            Takes the .xyz file in the self variable and outputs a list of Atoms objects
            '''

            atoms_list = []

            fl = open(self.file)
            fl = fl.readlines()
            fl = fl[2:-1]

            for i in fl:
                i = i.split()

                # extract atom symbol
                symbol = i[0]

                # remove it to just have atom positions
                del i[0]
                # generate xyz list
                xyz = [float(j) for j in i]

                new_atom = Atom(xyz, symbol)
                atoms_list.append(new_atom)

            return atoms_list

        self.atoms_list = atoms_list(self)

        # ordering list, SAME FOR ATOMS AND SOAPs
        ordering = [x for x in range(len(self.atoms_list))]
        random.shuffle(ordering)

        if shuffle:
            new_atoms_list = [self.atoms_list[i] for i in ordering]
            self.atoms_list = new_atoms_list

        def SOAPs(self):
            '''
            This function takes the .xyz files, computes SOAP for this specific molecule, and then creates a
            list of dictionaries containing the SOAP vector associated with each atomic environment.
            The SOAP for each environment is reshaped to a nx4 array s.t. each column correspond to a
            different l.

            Details of the SOAP container:
            - list of dictionaries, one dictionary per atom
            - dictionaries are formatted as: { (env tuple) : SOAP_array((n, 4))}
            '''

            SOAP_vectors = []

            # create a molecule object
            mol = ase.io.read(self.file)
            # get the atomic species
            species_in_mol = set(mol.get_chemical_symbols())  # not sorted according to Z
            # initialise SOAP object
            soap = SOAP(
                species=species_in_mol,
                periodic=False,
                rcut=r_cut,
                nmax=n_max,
                lmax=l_max
            )

            # compute SOAP, output is an array, each row corresponding to a different atom
            SOAP_array = soap.create(mol)

            # now extract the various SOAP components for each atom and arrange them
            # in a dictionary data structure

            # create the environment tuples for this molecule
            envs = env_manager(species_in_mol).env_tuples()
            # extract loactions, arranged as  in a dictionary
            SOAP_locations = {x: soap.get_location(x) for x in envs}

            for n in range(len(self.atoms_list)):  # loop over all atoms
                SOAP_atom_vect = SOAP_array[n]
                SOAP_atom_dict = {}

                for x in envs:  # loop over all environments

                    if x[0] == x[1]:  # SS
                        env_component = SOAP_atom_vect[SOAP_locations[x]]
                        N = int(len(env_component) / 4)

                        V = np.zeros((N, 4))
                        for m in range(4):
                            start = int(m * N)
                            end = int((m + 1) * N)
                            l_component = env_component[start:end]
                            V[:, m] = l_component

                        SOAP_atom_dict[x] = V

                    else:  # IS
                        env_component = SOAP_atom_vect[SOAP_locations[x]]
                        N = int(len(env_component) / 4)

                        V = np.zeros((N, 4))
                        for m in range(4):
                            start = int(m * N)
                            end = int((m + 1) * N)
                            l_component = env_component[start:end]
                            V[:, m] = l_component

                        SOAP_atom_dict[x] = V

                SOAP_vectors.append(SOAP_atom_dict)

            return SOAP_vectors

        self.SOAPs_list = SOAPs(self)

        if shuffle:
            new_SOAPs_list = [self.SOAPs_list[i] for i in ordering]
            self.SOAPs_list = new_SOAPs_list

# 20-07-2022: for now there is no mapping between labels and active network variables, I am assuming that the
# nodes are fed in the label order
# This class contains a group of attributes which are acted upon by the network, and another groupss of attributes
# that constitute the labels
class MoleculeGraph:
    '''
    Object to store molecule-wide information.

    NETWORK objects
    atoms: atoms in molecule are stored as a list of Atom objetcs.
    species : the atomic species are stored in a set.
    SOAPs: SOAP for each atom is stored as a list of dictionaries.
    node_cloud: list of all the atomic positions in the graph, atom class is not specified.

    LABEL objects
    List of all the label atoms.
    List of all the label SOAPs, for data structure details see xyz_manager.SOAPs

    Args:
    - reference-molecule: xyz_manager object that contains a list of Atoms objects and a list of SOAPs, for
    each atom. They are used to create the label variables.
    '''

    def __init__(self, reference_molecule):
        # initialise the object as an empty container.
        # NETWORK objects
        self.atoms = []
        self.species = set()
        self.SOAPs = []

        # LABEL objects
        self.atom_labels = reference_molecule.atoms_list
        self.SOAP_labels = reference_molecule.SOAPs_list

        node_cloud = [x.position for x in self.atom_labels]
        self.node_cloud = node_cloud

    def add_atom(self, new_atom):
        '''
        This function takes either a [x, y, z, "symbol"] input or an Atom input (new atom added to the graph)
        and returns:
        - updated atom list, with new Atom object
        - uptated species set, with eventual new symbol
        - updated list of SOAP dictionaries
        '''

        # convert to Atom datatype if not already
        if not isinstance(new_atom, Atom):
            new_atom = Atom(new_atom[0:3], new_atom[-1])

        # create current env tuples
        if bool(self.species):
            current_envs = set(env_manager(self.species).env_tuples())
        else:
            current_envs = set()  # handle addition of first atom

        # append new atom and update species set
        self.atoms.append(new_atom)
        self.species.add(new_atom.atom_class)

        # create updated env tuples
        new_envs = set(env_manager(self.species).env_tuples())

        # set of new environments
        new_env_tuples = new_envs - current_envs

        if bool(new_env_tuples):
            if len(self.atoms) == 1:  # handle addition of first atom
                self.SOAPs.append({})
                for x in new_env_tuples:
                    if x[0] == x[1]:
                        self.SOAPs[0][x] = np.zeros((size_l_SS, 4))
                    else:
                        self.SOAPs[0][x] = np.zeros((size_l_IS, 4))
            else:
                # first add new environemnts to pre-existing atom disctionaries
                for n in range(len(self.atoms) - 1):
                    for x in new_env_tuples:
                        if x[0] == x[1]:
                            self.SOAPs[n][x] = np.zeros((size_l_SS, 4))
                        else:
                            self.SOAPs[n][x] = np.zeros((size_l_IS, 4))
                # add new atom dictionary
                self.SOAPs.append({})
                for x in new_envs:
                    if x[0] == x[1]:
                        self.SOAPs[-1][x] = np.zeros((size_l_SS, 4))
                    else:
                        self.SOAPs[-1][x] = np.zeros((size_l_IS, 4))

    def update_SOAP(self, atom, env, l_component, new_SOAP_vector):
        '''
        Args:
        atom: Atom object to be updated or Index in the list of objects
        env: chemical environment tuple
        l_component: 0, 1, 2, 3. l_value to be updated.
        new_SOAP_vector: output of NN nodel.
        '''
        if isinstance(atom, Atom):
            index = self.atoms.index(atom)
        else:
            index = atom

        self.SOAPs[atom][env][:, l_component] = new_SOAP_vector