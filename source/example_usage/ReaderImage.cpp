#include "ReaderImage.h"
#include <fstream>

ReaderImage::ReaderImage(unsigned cnt) {
	this->cnt = cnt;

	ready = false;
}

void ReaderImage::read(const std::string &fileName) {
	std::ifstream in(fileName, std::ios::binary);
	in.unsetf(std::ios::skipws);

	unsigned char uc;

	// skip header
	for (unsigned i = 0; i < 16; ++i) in >> uc;

	pixels.clear();
	for (unsigned i = 0; i < getPixelCnt(); ++i) {
		in >> uc;
		pixels.push_back(uc);
	}

	in.close();

	ready = true;
}

unsigned ReaderImage::getPixelCnt() const {
	return getImageCnt() * getImageWidth() * getImageHeight();
}

unsigned ReaderImage::getPixel(unsigned image, unsigned row, unsigned column) const {
	return pixels[image * getImageWidth() * getImageHeight() + row * getImageWidth() + column];
}

void ReaderImage::getImage(unsigned image, unsigned char *dst) const {
	for (unsigned i = 0; i < getImageHeight(); ++i)
		for (unsigned j = 0; j < getImageWidth(); ++j)
			dst[i * getImageWidth() + j] = getPixel(image, i, j);
}