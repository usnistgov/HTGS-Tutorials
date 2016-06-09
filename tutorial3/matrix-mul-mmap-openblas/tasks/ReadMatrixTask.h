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

  ReadMatrixTask(int numThreads, MatrixType type, int blockSize, int fullMatrixWidth, int fullMatrixHeight, std::string directory, std::string matrixName) :
      ITask(numThreads), blockSize(blockSize), fullMatrixHeight(fullMatrixHeight), fullMatrixWidth(fullMatrixWidth), directory(directory), matrixName(matrixName)
  {
    this->type = type;
    numBlocksRows = (int)ceil((double)fullMatrixHeight / (double)blockSize);
    numBlocksCols = (int)ceil((double)fullMatrixWidth / (double)blockSize);
  }

  virtual ~ReadMatrixTask() {
    munmap(this->mmapMatrix, sizeof(double)*fullMatrixHeight*fullMatrixWidth);
  }
  virtual void initialize(int pipelineId,
                          int numPipeline) {
    std::string matrixName;
    switch(type)
    {
      case MatrixType::MatrixA: matrixName = "matrixA"; break;
      case MatrixType::MatrixB: matrixName = "matrixB"; break;
      case MatrixType::MatrixC:matrixName = "matrixC"; break;
    }

    std::string fileName(directory + "/" + matrixName);
//    std::ifstream inFile(fileName);
//    this->mmapMatrix = new double[fullMatrixWidth*fullMatrixHeight];
//
//    inFile.read((char *)mmapMatrix, sizeof(double) *fullMatrixWidth*fullMatrixHeight);

    int fd = -1;
    if ((fd = open(fileName.c_str(), O_RDONLY)) == -1) {
      err(1, "open failed");
    }

    this->mmapMatrix = (double *)mmap(NULL, fullMatrixWidth*fullMatrixHeight*sizeof(double), PROT_READ, MAP_SHARED, fd, 0);

    if (this->mmapMatrix == MAP_FAILED)
    {
      close(fd);
      err(2, "Error mmapping file");
    }

    close(fd);
  }
  virtual void shutdown() {
  }

  virtual void executeTask(std::shared_ptr<MatrixRequestData> data) {
    std::string matrixName;

    int row = data->getRow();
    int col = data->getCol();

    int matrixWidth;
    int matrixHeight;

    if (col == numBlocksCols-1 && fullMatrixWidth % blockSize != 0)
      matrixWidth = fullMatrixWidth % blockSize;
    else
      matrixWidth = blockSize;


    if (row == numBlocksRows-1 && fullMatrixHeight % blockSize != 0)
      matrixHeight = fullMatrixHeight % blockSize;
    else
      matrixHeight = blockSize;

    // compute starting location of pointer
    double *memPtr = &mmapMatrix[blockSize*col+blockSize*row*fullMatrixWidth];

    addResult(new MatrixBlockData<double *>(data, memPtr, matrixWidth, matrixHeight));

  }
  virtual std::string getName() {
    return "ReadMatrixTask(" + matrixName + ")";
  }
  virtual ReadMatrixTask *copy() {
    return new ReadMatrixTask(this->getNumThreads(), this->type, blockSize, fullMatrixWidth, fullMatrixHeight, directory, matrixName);
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
  double *mmapMatrix;
  int blockSize;
  int fullMatrixWidth;
  int fullMatrixHeight;
  int numBlocksRows;
  int numBlocksCols;
  std::string directory;
  std::string matrixName;

};

#endif //HTGS_GENERATEMATRIXTASK_H
