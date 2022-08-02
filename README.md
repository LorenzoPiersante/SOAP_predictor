# test_SOAP_predictor

N: number of atoms in the molecule

This is a test run of a ML framework needed to predict the SOAP molecular descriptor out of the graph of a molecule.

The train routine developed here is similar to the one implemented in the branch train-routine-1. The main difference lies in the preparation of the imput for the model in mode 1 and 2 of the training phase.
In order to speed up the execution time of the model on a training sample, the first N-1 iterations (graph building process) are skipped and their outputs approximated by a noisy SOAP corresponding to the current shape of the molecular graph.
This output (computed using the SOAP object of the dscribe package) is the input for the Nth iteration (mode 1) of the model, which is then fed to the finalisation mode of the model.

This allowed to gain a substantial speed-up of the training phase, which, despite the improvemnet, is still too slow to ecexute to be of practical use.


This branch presents some substantial modifications in the DataStructure.py file, especially the xyz_manager() file, which has been modified to produce the noisy intermediate SOAP at iteration N-1 (when the molecular graph is composed of N-1 nodes).
These are assigned to the MoleculeGraph objects before it is input to the model.

Lastly, the MoleculeGraph object, in the part that generates the GRU input, has been modified to keep the tensor on the utilized device (cuda) in order to avoid a performace loss due to the data transfer beteween cpu and cuda.
