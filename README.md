# Neural Networks

An implementation of feedforward neural networks in C++. NeuralNetwork class can be initiated with a desired architecture, trained on a set of data and then used on real examples.

This implementation uses neural networks for categorization problems. It outputs a number from 0 to 1 for each category, representing a confidence for that category. Sigmoid function is used as activation function and gradient descent as optimizer. Regularization is used as a measure against overfitting.

Also included is an example usage for analyzing the MNIST database. MNIST is a popular dataset of handwritten digits. You can learn more about it here: http://yann.lecun.com/exdb/mnist

This project is primarily intended for educational purposes, as real machine learning applications use enormous amounts of data and need infrastructures with good performance to support them.

## How to use

All needed files are under *source/neural_networks*. Include these in your project. What you will get are two classes: Matrix (for linear algebra calculations) and NeuralNetwork (which you will use).

NeuralNetwork can then be initialized and trained like this:

	NeuralNetwork *NN = new NeuralNetwork(inputLayerSize, vectorOfHiddenLayerSizes, outputCategoryCount);
		
	shuffleTrainSamplesSet();

	for (int i = 0; i < numberOfTrainEpochs; ++i) {
		for (TrainSample trainSample : trainSet) {
			for (int j = 0; j < inputLayerSize; ++j)
				NN->setInput(j, trainSample.inputVector[j]);
			
			// output labels as bool values
			NN->backpropagation(trainSample.correctOutputVector);
		}
		
		NN->gradientDescentStep();
	}

After training, use them on real or test samples like this:

	for (int i = 0; i < inputLayerSize; ++i)
		NN->setInput(i, sampleInputVector[i]);
	NN->feedforward();
	
	for (int i = 0; i < outputCategoryCount; ++i)
		categoryConfidence[i] = NN->getOutput(i);

## MNIST example

Under *source/example_usage* is a program which uses this implementation of neural networks on MNIST database (files in *MNIST* folder). This program will read train and test sets, train a neural network and test and print its accuracy achieved on the test set.
