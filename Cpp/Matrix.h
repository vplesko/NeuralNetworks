#ifndef _Matrix_h_
#define _Matrix_h_

#include <iostream>

class Matrix {
	unsigned r, c;
	double **vals;

	void copy(const Matrix &M);
	void clear();

public:
	Matrix();
	explicit Matrix(unsigned r, unsigned c = 1);

	Matrix(const Matrix &M);
	Matrix& operator=(const Matrix &M);

	unsigned getRows() const { return r; }
	unsigned getCols() const { return c; }

	void setSize(unsigned r, unsigned c = 1);

	void allTo(double v = 0.0);

	// it's possible that there's a bug with these two!
	double operator()(unsigned i, unsigned j = 0) const { return vals[i][j]; }
	double& operator()(unsigned i, unsigned j = 0) { return vals[i][j]; }
	
	Matrix operator*(double x) const;
	friend Matrix operator*(double x, const Matrix &M);
	Matrix operator/(double x) const;
	
	Matrix& operator*=(double x);
	Matrix& operator/=(double x);
	
	Matrix operator+(const Matrix &M) const;
	Matrix operator-(const Matrix &M) const;
	Matrix operator*(const Matrix &M) const;

	Matrix& operator+=(const Matrix &M);
	Matrix& operator-=(const Matrix &M);

	Matrix elemWiseMult(const Matrix &M) const;

	Matrix trans() const;

	friend std::ostream& operator<<(std::ostream &out, const Matrix &M);
	friend std::istream& operator>>(std::istream &in, Matrix &M);

	~Matrix();
};

#endif