# Deep-fractional-BP-networks
There exists four files.

## BP_main.m
BP_main.m is the main function of BP network. Here we define 5 layers (inputSize, 64, 64, 64, numClass), and you can define 
your layers and nodes. When you run *BP_main.m*, first you should choose the run pattern.
"1" means train mode, and "0" means test mode.

If you select train mode, then you can set an arbitrary fractional order *v* (0<*v*<2) to train.
Otherwise, function will load the weights.mat which contains all pretrained weights parameters.

## bpNN.m
We define a class of fractional back propagation neural network.

## mnist_small_matlab.mat
This is the mnist dataset. Because of the limitation of file size, we just upload smaller dataset (10, 000 train images).

## weights.mat
It contains pretrain weights parameters (60000 train images, 500 epochs, fractional order = 1.1).
