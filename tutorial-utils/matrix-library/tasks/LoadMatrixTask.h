
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//
#ifndef HTGS_LOADMATRIXTASK_H
#define HTGS_LOADMATRIXTASK_H

#include <htgs/api/ITask.hpp>
#include <cmath>
#include "../data/MatrixBlockData.h"
#include "../data/MatrixRequestData.h"
#include "../../util-matrix.h"

class LoadMatrixTask : public htgs::ITask<MatrixRequestData, MatrixBlockData<double *>> {

 public:

  LoadMatrixTask(double *matrix, size_t numThreads, MatrixType matrixType, size_t blockSize, size_t fullMatrixWidth, size_t fullMatrixHeight, bool colMajor) :
      ITask(numThreads),
      matrix(matrix),
      blockSize(blockSize),
      fullMatrixHeight(fullMatrixHeight),
      fullMatrixWidth(fullMatrixWidth),
      matrixType(matrixType),
      colMajor(colMajor)
  {
    numBlocksRows = (size_t) ceil((double) fullMatrixHeight / (double) blockSize);
    numBlocksCols = (size_t) ceil((double) fullMatrixWidth / (double) blockSize);
  }

  virtual ~LoadMatrixTask() {}

  virtual void executeTask(std::shared_ptr<MatrixRequestData> data) {
    size_t row = data->getRow();
    size_t col = data->getCol();

    size_t matrixWidth;
    size_t matrixHeight;

    if (col == numBlocksCols - 1 && fullMatrixWidth % blockSize != 0)
      matrixWidth = fullMatrixWidth % blockSize;
    else
      matrixWidth = blockSize;

    if (row == numBlocksRows - 1 && fullMatrixHeight % blockSize != 0)
      matrixHeight = fullMatrixHeight % blockSize;
    else
      matrixHeight = blockSize;

    double *memPtr;

    // compute starting location of pointer
    if (colMajor)
      memPtr = &matrix[IDX2C(blockSize*row, blockSize*col, fullMatrixHeight)];
    else
      memPtr = &matrix[blockSize * col + blockSize * row * fullMatrixWidth];

    if (colMajor)
      addResult(new MatrixBlockData<double *>(data, memPtr, matrixWidth, matrixHeight, fullMatrixHeight));
    else
      addResult(new MatrixBlockData<double *>(data, memPtr, matrixWidth, matrixHeight, fullMatrixWidth));

  }
  virtual std::string getName() {
    return "LoadMatrixTask(" + matrixTypeToString(matrixType) + ")";
  }
  virtual LoadMatrixTask *copy() {
    return new LoadMatrixTask(matrix, this->getNumThreads(), matrixType, blockSize, fullMatrixWidth, fullMatrixHeight, colMajor);
  }

  size_t getNumBlocksRows() const {
    return numBlocksRows;
  }

  size_t getNumBlocksCols() const {
    return numBlocksCols;
  }
 private:
  double *matrix;
  size_t blockSize;
  size_t fullMatrixWidth;
  size_t fullMatrixHeight;
  size_t numBlocksRows;
  size_t numBlocksCols;
  MatrixType matrixType;
  bool colMajor;
};

#endif //HTGS_GENERATEMATRIXTASK_H
