# SOAP_predictor - final version

This is a test run of a ML framework needed to predict the SOAP molecular descriptor out of the graph of a molecule.

This branch develops a different model architecture (still based on attention) which takes full advantage of vectorisation of operations. The RNN part part of the model that was used to combine the SOAP predictions to the hidden state vectors of the graph has been removed in order to make the net more shallow and prevent the problem of exploding gradients.

The training procedure is based on 3 different atomic classes only (arbitrarily called H, C and O) and on batched of fixed size, where the fixed size is the number of atoms in the molecule.

Molecules are not drawn from xyz files, but they are generated on the fly as clouds of randomly distributed points belonging to at least two classes, and at most three. Since molecules are generated, the number data is not a limitation and we can train the model on the desired molecule size.
Having a fixed sized batch was essential to vectorise all operations.

The prediction of the SOAP descriptor is here regarded as a regression problem on a mathematical operator.
