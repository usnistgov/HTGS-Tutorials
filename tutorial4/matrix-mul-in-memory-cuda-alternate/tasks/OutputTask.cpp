//
// Created by tjb3 on 6/22/16.
//


#include "OutputTask.h"



OutputTask::OutputTask(double *matrix, long fullMatrixWidth, long fullMatrixHeight, int blockSize) :
    matrix(matrix), fullMatrixWidth(fullMatrixWidth), fullMatrixHeight(fullMatrixHeight), blockSize(blockSize) {
  numBlocksRows = (int)ceil((double)fullMatrixHeight / (double)blockSize);
  numBlocksCols = (int)ceil((double)fullMatrixWidth / (double)blockSize);
}

void OutputTask::executeTask(std::shared_ptr<MatrixBlockData<double *>> data) {
  int col = data->getRequest()->getCol();
  int row = data->getRequest()->getRow();

  double *startLocation = &this->matrix[IDX2C(blockSize*row, blockSize*col, fullMatrixHeight)];

  long dataWidth = data->getMatrixWidth();
  long dataHeight = data->getMatrixHeight();
  double *matrixData = data->getMatrixData();

  for (long c = 0; c < dataWidth; c++)
  {
    for (long r = 0; r < dataHeight; r++)
    {
      startLocation[IDX2C(r, c, fullMatrixHeight)] = matrixData[IDX2C(r, c, dataHeight)];
    }
  }

  delete [] matrixData;
  matrixData = nullptr;

  addResult(data->getRequest());
}