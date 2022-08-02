# test_SOAP_predictor

This is a test run of a ML framework needed to predict the SOAP molecular descriptor out of the graph of a molecule.

In this branch I have implemented a naive training procedure of the model:

1) Create randomised batched access to a dataset of 5000 training samples and 1000 evaluation samples;
2) Apply the model to each sample in the minibatch and accumulate the loss;
3) Backpropagate and optimise the model parameters;
4) Apply the model to an evaluation minibatch.

I checkpoint the model after each minibatch and save a history of the training and evaluation losses for each epoch.

The model has been modified to operate in 4 different modes depending on the training or evaluation stage:

Trainig modes
Mode 0: execute the model up until iteraion N-1 (N-1 atoms have been added to the graph)
Mode 1: execute N-th iteration from N-1th iteration output
Mode 2: finalisation

Evaluation modes:
Mode 3: the forward goes through Modes 0, 1, 2 in one go.

There are two major issues with the current implementation:
- excessive execution time which make the model unpractical;
- the ipossibility to use a batched input;
- input vectorisation is not fully taken advantage of (that is we are not using the GPU computational capacity in full);
- excessive memory consumption in the storage of gradients.
