//
// Created by tjb3 on 2/23/16.
//
#ifndef HTGS_READMATRIXTASK_H
#define HTGS_READMATRIXTASK_H

#include <htgs/api/ITask.hpp>
#include <cmath>
#include "../memory/MatrixMemoryRule.h"

class ReadMatrixTask : public htgs::ITask<MatrixRequestData, MatrixBlockData<MatrixMemoryData_t>> {

 public:

  ReadMatrixTask(int numThreads, int blockSize, int fullMatrixWidth, int fullMatrixHeight, std::string directory) :
      ITask(numThreads),
      blockSize(blockSize),
      fullMatrixHeight(fullMatrixHeight),
      fullMatrixWidth(fullMatrixWidth),
      directory(directory) {
    numBlocksRows = (int) ceil((double) fullMatrixHeight / (double) blockSize);
    numBlocksCols = (int) ceil((double) fullMatrixWidth / (double) blockSize);
  }

  virtual ~ReadMatrixTask() {}
  virtual void initialize(int pipelineId,
                          int numPipeline) {

  }
  virtual void shutdown() {}

  virtual void executeTask(std::shared_ptr<MatrixRequestData> data) {
    std::string matrixName;

    switch (data->getType()) {
      case MatrixType::MatrixA: matrixName = "matrixA";
        break;
      case MatrixType::MatrixB: matrixName = "matrixB";
        break;
      case MatrixType::MatrixC: return;
    }
    MatrixMemoryData_t matrixData = this->memGet<double *>(matrixName, new MatrixMemoryRule(1));

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

    std::string fileName(directory + "/" + matrixName + "/" + std::to_string(row) + "_" + std::to_string(col));

    // Read data
    std::ifstream file(fileName, std::ios::binary);

    file.read((char *) matrixData->get(), sizeof(double) * matrixWidth * matrixHeight);

    addResult(new MatrixBlockData<MatrixMemoryData_t>(data, matrixData, matrixWidth, matrixHeight));

  }
  virtual std::string getName() {
    return "ReadMatrixTask";
  }
  virtual htgs::ITask<MatrixRequestData, MatrixBlockData<MatrixMemoryData_t>> *copy() {
    return new ReadMatrixTask(this->getNumThreads(), blockSize, fullMatrixWidth, fullMatrixHeight, directory);
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
