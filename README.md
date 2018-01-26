# NeuralNetwork_4Layers
A neural network with 2 hidden layers using the backpropagation algorithm to learn.
started: december 2017

some documentation of the code:
- new neuralNet(sizes) can be used to create a new net. sizes is an array of length 4 with the sizes of the layers.
- train(input, desired) can be called to perform backpropagation over all the weights, where all the parameters are expected to be arrays.
- fullTrain(avrError, runtime, data, log) can be called to train the network on the specified data, untill the average error is below avrError (parameter). runtime is the maximum amount of times the network learns, so if the network cannot perform optimally, we won't get stuk in a loop. (there is a while loop in there which we want to break). Log can be used to log the errors and the avrErrors to the console, as well as the runtime. This is set to be false at default, but can be set true as a parameter
- run(input) can be used to run the network on a specified input. input is expected to be an array. 

in the example file (p5main.js) mousePressed() is used. this is a p5.js function, and so all the libraries are linked in the index.html.
Also at the end neuralNetwork.js, I have put some functions to calculate the sigmoid of x, the derivative sigmoid of x and te reverse sigmoid of x. The training process is logged to the console. 

A working model can be found at https://luukfroling.github.io/NeuralNetwork_4Layers/ .
