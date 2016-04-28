Pretraining and fine-tuning steps for Neon can be performed using a couple of tricks.
We can set the learning rate of some of the layers that we should not update to zero using multiple optimizers for different layers (see https://github.com/NervanaSystems/neon/blob/master/examples/multi_optimizer.py)
The gradient will be computed for all the layers, but the update is performed with learning rate of zero on some layers. 

Regarding the transfer of the learned weights to the next AE: this can be done using some tricks. See https://groups.google.com/forum/#!topic/neon-users/2RGMsBNOjx4

For timinig we do not need to worry about these issues.
