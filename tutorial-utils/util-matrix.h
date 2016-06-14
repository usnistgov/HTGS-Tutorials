//
// Created by tjb3 on 1/15/16.
//

#ifndef HTGS_UTIL_MATRIX_H_H
#define HTGS_UTIL_MATRIX_H_H

#include <complex>
#include <iostream>
#include <fstream>
#include "enums/MatrixType.h"

double *allocMatrix(int width, int height);
void initMatrix(double *matrix, int width, int height);
std::string generateFilename(std::string directory, MatrixType type, int blockRow, int blockCol, std::string suffix);
std::string generateDirectoryName(std::string basePath, int totalWidth, int totalHeight, int blockSize);
std::string generateDirectoryName(std::string basePath, int totalWidth, int totalHeight);
int generateMatrixBlockFiles(std::string path, MatrixType type, int totalWidth, int totalHeight, int blockSize);
int generateFullMatrixFile(std::string path, MatrixType type, int totalWidth, int totalHeight);

double *readMatrix(std::string path,
                  MatrixType type,
                  int totalWidth,
                  int totalHeight,
                  int blockSize,
                  int blockRow,
                  int blockCol,
                  std::string suffix);

size_t readMatrix(std::string path,
                  MatrixType type,
                  int totalWidth,
                  int totalHeight,
                  int blockSize,
                  int blockRow,
                  int blockCol,
                  double *&matrix,
                  std::string suffix);

size_t writeMatrix(std::string file, int matrixWidth, int matrixHeight, double *&matrix, bool silent);
bool checkMatrixBlockFiles(std::string path, MatrixType type, int totalWidth, int totalHeight, int blockSize);
void checkAndValidateMatrixBlockFiles(std::string directory, int widthA, int heightA, int widthB, int heightB, int blockSize);
void checkAndValidateMatrixFiles(std::string directory, int widthA, int heightA, int widthB, int heightB);

#endif //HTGS_UTIL_MATRIX_H_H
