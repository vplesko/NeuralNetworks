#ifndef _NeuralNetwork_h_
#define _NeuralNetwork_h_

#include "Matrix.h"
#include <string>
#include <vector>

class NeuralNetwork {
	unsigned inputSize;
	std::vector<unsigned> hiddenSize;
	unsigned labelNum;

	double lambda, alpha;

	std::vector<Matrix> theta;
	std::vector<Matrix> activ;
	std::vector<Matrix> delta;
	std::vector<Matrix> deriv;

	double cumulCost;
	unsigned trainSetSize;

	void init(bool randWeights);
	void initWeights(bool randWeights);

	void addCost(std::vector<bool> labels);

	// for debugging
	//double getTotalCost() const;
	//void gradientChecking(std::vector<bool> labels);

public:
	// it is allowed to have no hidden layers
	NeuralNetwork(unsigned numOfInputs, std::vector<unsigned> hiddenLayerSizes, unsigned numOfOutputs);
	explicit NeuralNetwork(const std::string &fileName);

	unsigned getInputSize() const { return inputSize; }
	unsigned getOutputSize() const { return labelNum; }

	void setRegulCoef(double lam) { lambda = lam; }
	void setLearningRate(double alph) { alpha = alph; }

	void setInput(unsigned index, double value);
	double getOutput(unsigned ind) const;

	// set input before calling this
	void feedforward();

	// set input before calling
	// labels present whether the correct output is 1 (on true) or 0 (on false)
	void backpropagation(std::vector<bool> labels);
	
	// pseudo code for batch gradient descent training:
	/*
		repeat (10)
			for each train example
				setNeuralInput;
				neuro->backpropagation(expectedOutputs);
			end for;

			neuro->gradientDescentStep();
		end repeat;
	*/
	void gradientDescentStep();

	void readFrom(const std::string &fileName);
	void writeTo(const std::string &fileName) const;
};

#endif