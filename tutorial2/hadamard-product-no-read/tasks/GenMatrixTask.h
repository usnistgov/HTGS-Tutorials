//
// Created by tjb3 on 2/23/16.
//

#include <htgs/api/ITask.hpp>
#include <cmath>
#include "../memory/MatrixMemoryRule.h"
#ifndef HTGS_READMATRIXTASK_H
#define HTGS_READMATRIXTASK_H

class GenMatrixTask : public htgs::ITask<MatrixRequestData, MatrixBlockData<double *>> {

 public:

  GenMatrixTask(int numThreads, int blockSize, int fullMatrixWidth, int fullMatrixHeight) :
      ITask(numThreads), blockSize(blockSize), fullMatrixHeight(fullMatrixHeight), fullMatrixWidth(fullMatrixWidth) {
    numBlocksRows = (int) ceil((double) fullMatrixHeight / (double) blockSize);
    numBlocksCols = (int) ceil((double) fullMatrixWidth / (double) blockSize);
  }

  virtual ~GenMatrixTask() {

  }
  virtual void initialize(int pipelineId,
                          int numPipeline) {

  }
  virtual void shutdown() {}

  virtual void executeTask(std::shared_ptr<MatrixRequestData> data) {
    int row = data->getRow();
    int col = data->getCol();

    int matrixWidth;
    int matrixHeight;

    if (col == numBlocksCols - 1 && fullMatrixWidth % blockSize != 0)
      matrixWidth = fullMatrixWidth % blockSize;
    else
      matrixWidth = blockSize;

    if (row == numBlocksRows - 1 && fullMatrixHeight % blockSize != 0)
      matrixHeight = fullMatrixHeight % blockSize;
    else
      matrixHeight = blockSize;

    // Allocate matrix Memory
    double *matrixData = new double[matrixHeight * matrixWidth];

    // Initialize with a simple value
    for (int i = 0; i < matrixWidth * matrixHeight; i++)
      matrixData[i] = 2.0;

    addResult(new MatrixBlockData<double *>(data, matrixData, matrixWidth, matrixHeight));

  }
  virtual std::string getName() {
    return "GenMatrixTask";
  }
  virtual htgs::ITask<MatrixRequestData, MatrixBlockData<double *>> *copy() {
    return new GenMatrixTask(this->getNumThreads(), blockSize, fullMatrixWidth, fullMatrixHeight);
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
  int blockSize;
  int fullMatrixWidth;
  int fullMatrixHeight;
  int numBlocksRows;
  int numBlocksCols;
  std::string directory;

};

#endif //HTGS_GENERATEMATRIXTASK_H
