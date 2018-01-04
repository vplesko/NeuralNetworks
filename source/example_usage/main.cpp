#include "NeuralControl.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
	srand((unsigned)(time(0)));

	high_resolution_clock::time_point t1, t2;

	unsigned percent = 100;
	if (argc > 1) percent = atoi(argv[1]);

	NeuralControl cont(percent);

	cont.readTrain();
	cont.readTest();

	t1 = high_resolution_clock::now();
	cont.train();
	t2 = high_resolution_clock::now();
	cout << "TRAIN TIME: " << duration_cast<milliseconds>(t2 - t1).count() << "ms" << endl;

	cont.writeTo(argc > 2 ? argv[2] : "NN.txt");

	ofstream res("results.txt", ios::app);

	t1 = high_resolution_clock::now();
	double trainAcc = cont.testOnTrain();
	double testAcc = cont.testOnTest();
	t2 = high_resolution_clock::now();
	cout << "TEST TIME: " << duration_cast<milliseconds>(t2 - t1).count() << "ms" << endl;

	res << "train_acc=" << trainAcc << "\ttest_acc=" << testAcc << "\t";
	cont.outputConfig(res);
	res << endl;

	res.close();
	
	return 0;
}