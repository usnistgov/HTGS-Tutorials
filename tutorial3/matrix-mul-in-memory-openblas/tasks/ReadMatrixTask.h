//
// Created by tjb3 on 2/23/16.
//
#ifndef HTGS_READMATRIXTASK_H
#define HTGS_READMATRIXTASK_H

#include <htgs/api/ITask.hpp>
#include <cmath>
#include "../memory/MatrixMemoryRule.h"

class ReadMatrixTask : public htgs::ITask<MatrixRequestData, MatrixBlockData<double *>>
{

 public:

  ReadMatrixTask(int numThreads, MatrixType type, int blockSize, int fullMatrixWidth, int fullMatrixHeight, double *matrix, std::string matrixName) :
      ITask(numThreads), blockSize(blockSize), fullMatrixHeight(fullMatrixHeight), fullMatrixWidth(fullMatrixWidth), matrix(matrix), matrixName(matrixName)
  {
    this->type = type;
    numBlocksRows = (int)ceil((double)fullMatrixHeight / (double)blockSize);
    numBlocksCols = (int)ceil((double)fullMatrixWidth / (double)blockSize);
  }

  virtual ~ReadMatrixTask() {
  }
  virtual void initialize(int pipelineId,
                          int numPipeline) {
  }
  virtual void shutdown() {
  }

  virtual void executeTask(std::shared_ptr<MatrixRequestData> data) {
    std::string matrixName;

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
    double *memPtr = &matrix[blockSize*col+blockSize*row*fullMatrixWidth];

    addResult(new MatrixBlockData<double *>(data, memPtr, matrixWidth, matrixHeight));

  }
  virtual std::string getName() {
    return "ReadMatrixTask(" + matrixName + ")";
  }
  virtual ReadMatrixTask *copy() {
    return new ReadMatrixTask(this->getNumThreads(), this->type, blockSize, fullMatrixWidth, fullMatrixHeight, matrix, matrixName);
  }
  virtual bool isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
    return inputConnector->isInputTerminated();
  }

  int getNumBlocksRows() const {
    return numBlocksRows;
  }
  int getNumBlocksCols() const {
    return numBlocksCols;
  }
 private:
  MatrixType type;
  double *matrix;
  int blockSize;
  int fullMatrixWidth;
  int fullMatrixHeight;
  int numBlocksRows;
  int numBlocksCols;
  std::string matrixName;


};

#endif //HTGS_GENERATEMATRIXTASK_H
