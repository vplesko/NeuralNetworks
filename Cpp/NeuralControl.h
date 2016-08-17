#ifndef _NeuralControl_h_
#define _NeuralControl_h_

#include <string>
#include <vector>

class NeuralNetwork;
class ReaderImage;
class ReaderLabel;

class NeuralControl {
	unsigned trainSize, testSize;

	NeuralNetwork *nn;

	ReaderImage *trainImages, *testImages;
	ReaderLabel *trainLabels, *testLabels;

	double alpha, lambda;
	std::vector<unsigned> hiddenArchitecture;
	unsigned cycles, batch;

	void setNeuralInput(unsigned char *image);
	
	double test(ReaderImage *img, ReaderLabel *lab);

public:
	NeuralControl(unsigned percentage);

	void readTrain();
	void readTest();

	void train();

	unsigned run(unsigned char *image, bool draw = false);

	double testOnTrain();
	double testOnTest();

	void readFrom(const std::string &fileName);
	void writeTo(const std::string &fileName) const;

	void outputConfig(std::ofstream &file) const;
	
	void testTrainImage(unsigned index);
	void testTestImage(unsigned index);

	~NeuralControl();
};

#endif