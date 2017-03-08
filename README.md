# Neural Networks

Student project in data mining. Contains classes for creating and storing a neural network model. The program uses them to analyze data taken from http://yann.lecun.com/exdb/mnist.

This neural network outputs confidences as a number from 0 to 1 for each possible output label.

If you want to use the neural network for yourself you'll need to include the following files into your project: Matrix.h, Matrix.cpp, NeuralNetwork.h and NeuralNetwork.cpp.

A pseudo-code for initializing and training the NN would look like this:

	NeuralNetwork *NN = new NeuralNetwork(inSize, hiddenLayerSizesVector, labelCnt);
	
	shuffleTrainSamples();

	repeat (cycleCnt) {
		// you may use batch, mini-batch or stochastic gradient descent as you will
		
		foreach (TrainSample T) {
			for (i : 0 to inSize) NN->setInput(i, T.input[i]);
			
			// labels as bool values
			NN->backpropagation(T.labels);
		}
		
		NN->gradientDescentStep();
	}

Running this NN can then be done as:

	for (i : 0 to inSize) NN->setInput(i, input[i]);
	NN->feedforward();
	
	for (i : 0 to labelCnt) labelConfidence[i] = NN->getOutput(i);
