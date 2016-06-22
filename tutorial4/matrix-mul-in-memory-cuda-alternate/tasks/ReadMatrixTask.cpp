//
// Created by tjb3 on 6/22/16.
//

#include "ReadMatrixTask.h"


ReadMatrixTask::ReadMatrixTask(int numThreads, MatrixType type, int blockSize, long fullMatrixWidth, long fullMatrixHeight, double *matrix, std::string matrixName) :
    ITask(numThreads), blockSize(blockSize), fullMatrixHeight(fullMatrixHeight), fullMatrixWidth(fullMatrixWidth), matrix(matrix), matrixName(matrixName)
{
  this->type = type;
  numBlocksRows = (int)ceil((double)fullMatrixHeight / (double)blockSize);
  numBlocksCols = (int)ceil((double)fullMatrixWidth / (double)blockSize);
}

void ReadMatrixTask::executeTask(std::shared_ptr<MatrixRequestData> data) {

  int row = data->getRow();
  int col = data->getCol();

  long matrixWidth;
  long matrixHeight;

  if (col == numBlocksCols-1 && fullMatrixWidth % blockSize != 0)
    matrixWidth = fullMatrixWidth % blockSize;
  else
    matrixWidth = blockSize;


  if (row == numBlocksRows-1 && fullMatrixHeight % blockSize != 0)
    matrixHeight = fullMatrixHeight % blockSize;
  else
    matrixHeight = blockSize;

  // compute starting location of pointer
  double *memPtr = &matrix[IDX2C(blockSize*row, blockSize*col, fullMatrixHeight)];

  addResult(new MatrixBlockData<double *>(data, memPtr, matrixWidth, matrixHeight));

}