#ifndef _ReaderImage_h_
#define _ReaderImage_h_

#include <string>
#include <vector>

class ReaderImage {
	unsigned cnt;
	std::vector<unsigned char> pixels;

	bool ready;

public:
	ReaderImage(unsigned cnt);

	void read(const std::string &fileName);

	bool isRead() const { return ready; }

	unsigned getImageCnt() const { return cnt; }
	unsigned getImageWidth() const { return 28; }
	unsigned getImageHeight() const { return 28; }
	unsigned getPixelCnt() const;

	unsigned getPixel(unsigned image, unsigned row, unsigned column) const;
	void getImage(unsigned image, unsigned char *dst) const;
};

#endif