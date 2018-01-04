#include "ReaderLabel.h"
#include <fstream>

ReaderLabel::ReaderLabel(unsigned cnt) {
	this->cnt = cnt;

	ready = false;
}

void ReaderLabel::read(const std::string &fileName) {
	std::ifstream in(fileName, std::ios::binary);
	in.unsetf(std::ios::skipws);

	unsigned char uc;

	// skip header
	for (unsigned i = 0; i < 8; ++i) in >> uc;

	labels.clear();
	for (unsigned i = 0; i < getLabelCnt(); ++i) {
		in >> uc;
		labels.push_back(uc);
	}

	in.close();

	ready = true;
}