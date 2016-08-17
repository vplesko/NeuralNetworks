#include "Matrix.h"

Matrix::Matrix() {
	r = c = 0;
	vals = 0;
}

Matrix::Matrix(unsigned r, unsigned c) {
	this->r = r;
	this->c = c;

	vals = new double*[r];
	for (unsigned i = 0; i < r; ++i) {
		vals[i] = new double[c];
	}
}

void Matrix::copy(const Matrix &M) {
	clear();

	r = M.r;
	c = M.c;

	vals = new double*[r];
	for (unsigned i = 0; i < r; ++i) {
		vals[i] = new double[c];

		for (unsigned j = 0; j < c; ++j)
			vals[i][j] = M.vals[i][j];
	}
}

void Matrix::clear() {
	if (vals) {
		for (unsigned i = 0; i < r; ++i) delete [] vals[i];
		delete [] vals;

		r = c = 0;
		vals = 0;
	}
}

Matrix::Matrix(const Matrix &M) {
	vals = 0;
	copy(M);
}

Matrix& Matrix::operator=(const Matrix &M) {
	if (this == &M) return *this;

    if (vals) clear();
	copy(M);

    return *this;
}

void Matrix::setSize(unsigned r, unsigned c) {
	clear();

	this->r = r;
	this->c = c;

	vals = new double*[r];
	for (unsigned i = 0; i < r; ++i) {
		vals[i] = new double[c];
	}
}

void Matrix::allTo(double v) {
	for (unsigned i = 0; i < r; ++i)
		for (unsigned j = 0; j < c; ++j)
			vals[i][j] = v;
}

Matrix Matrix::operator*(double x) const {
	Matrix ret(r, c);
	
	for (unsigned i = 0; i < r; ++i) {
		for (unsigned j = 0; j < c; ++j) {
			ret(i, j) = vals[i][j] * x;
		}
	}

	return ret;
}

Matrix operator*(double x, const Matrix &M) {
	return M * x;
}

Matrix Matrix::operator/(double x) const {
	Matrix ret(r, c);
	
	for (unsigned i = 0; i < r; ++i) {
		for (unsigned j = 0; j < c; ++j) {
			ret(i, j) = vals[i][j] / x;
		}
	}

	return ret;
}

Matrix& Matrix::operator*=(double x) {
	(*this) = (*this) * x;

	return *this;
}

Matrix& Matrix::operator/=(double x) {
	(*this) = (*this) / x;

	return *this;
}

Matrix Matrix::operator+(const Matrix &M) const {
	if (r != M.r || c != M.c) throw "Mismatching dimensions on addition!";

	Matrix ret(r, c);
	
	for (unsigned i = 0; i < r; ++i) {
		for (unsigned j = 0; j < c; ++j) {
			ret(i, j) = vals[i][j] + M(i, j);
		}
	}

	return ret;
}

Matrix Matrix::operator-(const Matrix &M) const {
	if (r != M.r || c != M.c) throw "Mismatching dimensions on subtraction!";

	Matrix ret(r, c);
	
	for (unsigned i = 0; i < r; ++i) {
		for (unsigned j = 0; j < c; ++j) {
			ret(i, j) = vals[i][j] - M(i, j);
		}
	}

	return ret;
}

Matrix Matrix::operator*(const Matrix &M) const {
	if (c != M.r) throw "Mismatching dimensions on multiplication!";

	Matrix ret(r, M.c);

	for (unsigned i = 0; i < r; ++i) {
		for (unsigned j = 0; j < M.c; ++j) {
			ret(i, j) = 0.0;

			for (unsigned k = 0; k < c; ++k)
				ret(i, j) += (*this)(i, k) * M(k, j);
		}
	}

	return ret;
}

Matrix& Matrix::operator+=(const Matrix &M) {
	(*this) = (*this) + M;

	return *this;
}

Matrix& Matrix::operator-=(const Matrix &M) {
	(*this) = (*this) - M;

	return *this;
}

Matrix Matrix::elemWiseMult(const Matrix &M) const {
	if (r != M.r || c != M.c) throw "Mismatching dimensions on element-wise multiplication!";

	Matrix ret(r, c);
	
	for (unsigned i = 0; i < r; ++i) {
		for (unsigned j = 0; j < c; ++j) {
			ret(i, j) = vals[i][j] * M(i, j);
		}
	}

	return ret;
}

Matrix Matrix::trans() const {
	Matrix ret(c, r);

	if (r == 0 || c == 0) return ret;

	for (unsigned i = 0; i < r; ++i) {
		for (unsigned j = 0; j < c; ++j) {
			ret(j, i) = vals[i][j];
		}
	}

	return ret;
}

std::ostream& operator<<(std::ostream &out, const Matrix &M) {
	out << M.getRows() << ' ' << M.getCols();

	for (unsigned i = 0; i < M.getRows(); ++i) {
		out << std::endl;
		for (unsigned j = 0; j < M.getCols(); ++j) {
			if (j > 0) out << '\t';
			out << M(i, j);
		}
	}

	return out;
}

std::istream& operator>>(std::istream &in, Matrix &M) {
	M.clear();

	unsigned w, h;
	in >> w >> h;

	M.setSize(w, h);

	for (unsigned i = 0; i < M.getRows(); ++i) {
		for (unsigned j = 0; j < M.getCols(); ++j) {
			in >> M(i, j);
		}
	}

	return in;
}

Matrix::~Matrix() {
	clear();
}