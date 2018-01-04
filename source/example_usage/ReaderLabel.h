#ifndef _ReaderLabel_h_
#define _ReaderLabel_h_

#include <string>
#include <vector>

class ReaderLabel {
	unsigned cnt;
	std::vector<unsigned char> labels;

	bool ready;

public:
	ReaderLabel(unsigned cnt);

	void read(const std::string &fileName);

	bool isRead() const { return ready; }

	unsigned getLabelCnt() const { return cnt; }

	unsigned getLabel(unsigned index) const { return labels[index]; }
};

#endif