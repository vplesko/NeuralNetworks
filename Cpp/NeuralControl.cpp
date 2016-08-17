#include "NeuralControl.h"
#include "NeuralNetwork.h"
#include "ReaderImage.h"
#include "ReaderLabel.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>

NeuralControl::NeuralControl(unsigned percentage) {
	hiddenArchitecture.push_back(400);

	nn = new NeuralNetwork(28 * 28, hiddenArchitecture, 10);
	nn->setLearningRate(alpha = 0.3);
	nn->setRegulCoef(lambda = 0);

	cycles = 10;
	batch = 1000;
	
	if (percentage == 0) percentage = 1;
	else if (percentage > 100) percentage = 100;

	trainSize = (60000 / 100) * percentage;
	testSize = (10000 / 100) * percentage;

	trainImages = new ReaderImage(trainSize);
	trainLabels = new ReaderLabel(trainSize);

	testImages = new ReaderImage(testSize);
	testLabels = new ReaderLabel(testSize);
}

void NeuralControl::readTrain() {
	if (trainLabels->isRead()) return;

	std::cout << "Reading train labels...";
	trainLabels->read("train-labels.idx1-ubyte");
	std::cout << "\tDone." << std::endl;

	std::cout << "Reading train images...";
	trainImages->read("train-images.idx3-ubyte");
	std::cout << "\tDone." << std::endl;
}

void NeuralControl::readTest() {
	if (testLabels->isRead()) return;

	std::cout << "Reading test labels...";
	testLabels->read("t10k-labels.idx1-ubyte");
	std::cout << "\tDone." << std::endl;

	std::cout << "Reading test images...";
	testImages->read("t10k-images.idx3-ubyte");
	std::cout << "\tDone." << std::endl;
}

void NeuralControl::setNeuralInput(unsigned char *image) {
	for (unsigned k = 0; k < nn->getInputSize(); ++k)
		nn->setInput(k, (double)(image[k]));
}

void NeuralControl::train() {
	readTrain();

	unsigned char *image = new unsigned char[trainImages->getImageWidth() * trainImages->getImageHeight()];
	unsigned label;

	std::vector<unsigned> order;
	for (unsigned i = 0; i < trainImages->getImageCnt(); ++i) order.push_back(i);
	std::random_shuffle(order.begin(), order.end());

	std::vector<bool> labels;
	for (unsigned i = 0; i < nn->getOutputSize(); ++i) labels.push_back(false);
	
	unsigned writeProgressAt = (unsigned)(pow(10, floor(log10(order.size()))));
	unsigned prevWriteOn = 0;

	for (unsigned c = 0; c < cycles; ++c) {
		std::cout << "Cycle " << c << ":\n";
		
		prevWriteOn = 0;

		for (unsigned i = 0; i < order.size(); i += batch) {
			if (i - prevWriteOn >= writeProgressAt && i > 0) {
				std::cout << "\t" << i << "/" << order.size() << "\n";
				prevWriteOn = i;
			}

			for (unsigned j = 0; j < batch && i + j < order.size(); ++j) {
				trainImages->getImage(order[i + j], image);

				label = trainLabels->getLabel(order[i + j]);
				labels[label] = true;

				setNeuralInput(image);
				nn->backpropagation(labels);

				labels[label] = false;
			}

			nn->gradientDescentStep();
		}
	}

	delete [] image;
}

unsigned NeuralControl::run(unsigned char *image, bool draw) {
	if (draw) {
		std::cout << "Image:"<< std::endl;

		for (unsigned i = 0; i < trainImages->getImageHeight(); ++i) {
			for (unsigned j = 0; j < trainImages->getImageWidth(); ++j) {
				double pixel = image[i * trainImages->getImageWidth() + j];
				char ch;
				if (pixel > 127) ch = '#';
				else if (pixel > 63) ch = '@';
				else if (pixel > 31) ch = '%';
				else if (pixel > 15) ch = '+';
				else if (pixel > 7) ch = ':';
				else if (pixel > 3) ch = '\'';
				else if (pixel > 1) ch = '`';
				else ch = ' ';
				std::cout << ch;
			}

			std::cout << std::endl;
		}
	}

	setNeuralInput(image);
	nn->feedforward();

	double maxConf = 0.0;
	unsigned bestLabel = 0;

	if (draw) {
		std::cout << "Confidences:" << std::endl;
	}

	for (unsigned i = 0; i <= 9; ++i) {
		if (draw) {
			std::cout << "\t[" << i << "]=\t" << nn->getOutput(i) << "          \t (";

			int lines = (int)((nn->getOutput(i) + 0.05) * 10);
			if (lines < 0) lines = 0;
			else if (lines > 10) lines = 10;

			for (int j = 0; j < 10; ++j) std::cout << (j < lines ? '-' : ' ');

			std::cout << ")" << std::endl;
		}

		if (nn->getOutput(i) > maxConf) {
			maxConf = nn->getOutput(i);
			bestLabel = i;
		}
	}

	if (draw) {
		std::cout << "Label: " << bestLabel << std::endl;
	}

	return bestLabel;
}

double NeuralControl::test(ReaderImage *img, ReaderLabel *lab) {
	unsigned correct = 0;

	unsigned char *image = new unsigned char[img->getImageWidth() * img->getImageHeight()];
	unsigned label;

	for (unsigned i = 0; i < lab->getLabelCnt(); ++i) {
		img->getImage(i, image);
		label = lab->getLabel(i);

		if (run(image) == label) ++correct;
	}

	delete [] image;

	return ((double)correct) / lab->getLabelCnt();
}

double NeuralControl::testOnTrain() {
	readTrain();

	double ret = test(trainImages, trainLabels);
	std::cout << "Train set accuracy: " << ret << std::endl;
	return ret;
}

double NeuralControl::testOnTest() {
	readTest();
	
	double ret = test(testImages, testLabels);
	std::cout << "Test set accuracy: " << ret << std::endl;
	return ret;
}

void NeuralControl::readFrom(const std::string &fileName) {
	nn->readFrom(fileName);
}

void NeuralControl::writeTo(const std::string &fileName) const {
	nn->writeTo(fileName);
}

void NeuralControl::outputConfig(std::ofstream &file) const {
	file << "alpha=" << alpha << "\tlambda=" << lambda 
		<< "\tcycles=" << cycles << "\tbatch=" << batch;

	file << "\t{" << hiddenArchitecture.size();
	if (!hiddenArchitecture.empty()) file << ":";
	for (unsigned i = 0; i < hiddenArchitecture.size(); ++i) {
		if (i > 0) file << ",";
		file << hiddenArchitecture[i];
	}
	file << "}";
}

void NeuralControl::testTrainImage(unsigned index) {
	readTrain();

	unsigned char *image = new unsigned char[trainImages->getImageWidth() * trainImages->getImageHeight()];
	trainImages->getImage(index, image);

	if (run(image, true) == trainLabels->getLabel(index)) std::cout << "Correct!";
	else std::cout << "Wrong!";

	std::cout << std::endl;

	delete [] image;
}

void NeuralControl::testTestImage(unsigned index) {
	readTest();

	unsigned char *image = new unsigned char[testImages->getImageWidth() * testImages->getImageHeight()];
	testImages->getImage(index, image);

	if (run(image, true) == testLabels->getLabel(index)) std::cout << "Correct!";
	else std::cout << "Wrong!";

	std::cout << std::endl;

	delete [] image;
}

NeuralControl::~NeuralControl() {
	delete nn;
	delete trainImages;
	delete trainLabels;
	delete testImages;
	delete testLabels;
}