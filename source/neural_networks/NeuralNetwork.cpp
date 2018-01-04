#include "NeuralNetwork.h"
#include <cstdlib>
#include <cmath>
#include <fstream>

NeuralNetwork::NeuralNetwork(const std::string &fileName) {
	readFrom(fileName);
}

NeuralNetwork::NeuralNetwork(unsigned numOfInputs, 
	std::vector<unsigned> hiddenLayerSizes, 
	unsigned numOfOutputs) {
	inputSize = numOfInputs;
	labelNum = numOfOutputs;

	hiddenSize = hiddenLayerSizes;

	init(true);
}

void NeuralNetwork::init(bool randWeights) {
	if (inputSize == 0 || labelNum == 0) return;

	initWeights(randWeights);

	// input
	activ.push_back(Matrix(inputSize + 1));

	// hidden layers
	for (unsigned i = 0; i < hiddenSize.size(); ++i)
		activ.push_back(Matrix(hiddenSize[i] + 1));

	// output
	activ.push_back(Matrix(labelNum));

	// bias units in input and hidden layers
	for (unsigned i = 0; i < 1 + hiddenSize.size(); ++i)
		activ[i](0) = 1.0;

	cumulCost = 0.0;
	trainSetSize = 0;

	lambda = 0;
	alpha = 0.01;

	// delta 'errors' for hidden layers
	for (unsigned i = 0; i < hiddenSize.size(); ++i)
		delta.push_back(Matrix(hiddenSize[i]));

	// and for the output layer
	delta.push_back(Matrix(labelNum));
}

double randForInit(double eps) {
	return (((double)rand()) / (RAND_MAX)) * 2 * eps - eps;
}

void NeuralNetwork::initWeights(bool randWeights) {
	if (inputSize == 0 || labelNum == 0) return;

	double epsilon;
	if (randWeights) epsilon = sqrt(6.0 / (inputSize + labelNum));

	// weight of connections
	for (unsigned i = 0; i < hiddenSize.size() + 1; ++i) {
		theta.push_back(Matrix(i < hiddenSize.size() ? hiddenSize[i] : labelNum, 
			(i > 0 ? hiddenSize[i - 1] : inputSize) + 1));
	}

	if (randWeights) {
		for (unsigned i = 0; i < theta.size(); ++i)
			for (unsigned j = 0; j < theta[i].getRows(); ++j)
				for (unsigned k = 0; k < theta[i].getCols(); ++k)
					theta[i](j, k) = randForInit(epsilon);
	}

	// first derivatives of cost function of theta weights
	for (unsigned i = 0; i < theta.size(); ++i) {
		deriv.push_back(Matrix(theta[i].getRows(), theta[i].getCols()));
		deriv[deriv.size() - 1].allTo(0.0);
	}
}

double sigm(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double sigmGrad(double sigma) {
	return sigma * (1.0 - sigma);
}

void NeuralNetwork::feedforward() {
	if (inputSize == 0 || labelNum == 0) return;

	Matrix z;
	
	for (unsigned i = 1; i < activ.size(); ++i) {
		z = theta[i - 1] * activ[i - 1];

		if (i + 1 < activ.size()) {
			for (unsigned j = 1; j <= hiddenSize[i - 1]; ++j)
				activ[i](j) = sigm(z(j - 1));
		} else {
			for (unsigned j = 0; j < labelNum; ++j)
				activ[i](j) = sigm(z(j));
		}
	}
}

void NeuralNetwork::setInput(unsigned index, double value) {
	// skip the 0th which is the +1 bias
	activ[0](index + 1) = value;
}

double NeuralNetwork::getOutput(unsigned ind) const {
	return activ[activ.size() - 1](ind);
}

void NeuralNetwork::addCost(std::vector<bool> labels) {
	if (labelNum == 0) return;

	for (unsigned i = 0; i < labelNum; ++i) {
		if (labels[i]) {
			cumulCost += -log(getOutput(i));
		} else {
			cumulCost += -log(1 - getOutput(i));
		}
	}
}

void NeuralNetwork::backpropagation(std::vector<bool> labels) {
	if (inputSize == 0 || labelNum == 0) return;

	feedforward();

	addCost(labels);
	++trainSetSize;

	for (unsigned i = 0; i < labelNum; ++i)
		delta[delta.size() - 1](i) = getOutput(i) - (labels[i] ? 1.0 : 0.0);

	Matrix tmp;
	if (hiddenSize.size() > 0) {
		for (unsigned i = hiddenSize.size() - 1; i >= 0; --i) {
			tmp = theta[i + 1].trans() * delta[i + 1];
			for (unsigned j = 0; j < hiddenSize[i]; ++j) {
				delta[i](j) = tmp(j + 1) * sigmGrad(activ[i + 1](j + 1));
			}

			// cuz of unsigned arithmetic the loop condition is always correct
			if (i == 0) break;
		}
	}

	for (unsigned i = 0; i < deriv.size(); ++i) {
		deriv[i] += delta[i] * activ[i].trans();
	}
}

void NeuralNetwork::gradientDescentStep() {
	// add regularization cost derivatives, not applied to bias connections
	for (unsigned i = 0; i < deriv.size(); ++i) {
		deriv[i] += theta[i] * lambda;

		for (unsigned j = 0; j < deriv[i].getRows(); ++j)
			deriv[i](j, 0) -= theta[i](j, 0) * lambda;

		deriv[i] /= trainSetSize;
	}

	// descent
	for (unsigned i = 0; i < theta.size(); ++i)
		theta[i] -= deriv[i] * alpha;

	cumulCost = 0.0;
	trainSetSize = 0;
	for (unsigned i = 0; i < deriv.size(); ++i) deriv[i].allTo(0.0);
}

void NeuralNetwork::readFrom(const std::string &fileName) {
	hiddenSize.clear();
	theta.clear();
	activ.clear();
	delta.clear();
	deriv.clear();

	std::ifstream file(fileName);

	if (!file.is_open()) throw "Neural network file not found: " + fileName;

	file >> inputSize >> labelNum;

	unsigned hiddenLayers;
	file >> hiddenLayers;
	for (unsigned i = 0; i < hiddenLayers; ++i) {
		unsigned size;
		file >> size;
		hiddenSize.push_back(size);
	}

	init(false);

	for (unsigned i = 0; i < theta.size(); ++i)
		file >> theta[i];

	file.close();
}

void NeuralNetwork::writeTo(const std::string &fileName) const {
	std::ofstream file(fileName);

	file << inputSize << ' ' << labelNum << std::endl;

	file << hiddenSize.size();
	for (unsigned i = 0; i < hiddenSize.size(); ++i)
		file << ' ' << hiddenSize[i];

	for (unsigned i = 0; i < theta.size(); ++i)
		file << std::endl << theta[i];

	file.close();
}

// for debugging
/*double NeuralNetwork::getTotalCost() const {
	if (trainSetSize == 0) return 0.0;

	// add regularization cost, which is not applied to bias connections
	double sum = 0.0;
	
	for (unsigned i = 0; i < theta.size(); ++i)
		for (unsigned j = 0; j < theta[i].getRows(); ++j)
			for (unsigned k = 1; k < theta[i].getCols(); ++k)
				sum += theta[i](j, k) * theta[i](j, k);

	sum *= lambda / 2;

	return (cumulCost + sum) / trainSetSize;
}*/

/*void NeuralNetwork::gradientChecking(std::vector<bool> labels) {
	double epsilon = 0.02;

	unsigned l = 0, x = 15, y = 0;

	std::vector<Matrix> th = theta;

	backpropagation(labels);
	double derivative = (deriv[l](x, y) + theta[l](x, y) * lambda) / trainSetSize;
	std::cout << derivative;

	gradientDescentStep();
	theta = th;

	theta[l](x, y) -= epsilon;
	backpropagation(labels);
	double costMinus = getTotalCost();

	gradientDescentStep();
	theta = th;

	theta[l](x, y) += epsilon;
	backpropagation(labels);
	double costPlus = getTotalCost();

	double aproxDerivative = (costPlus - costMinus) / (2 * epsilon);
	std::cout << ' ' << aproxDerivative << std::endl;
}*/